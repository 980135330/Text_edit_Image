import torch
from inspect import isfunction
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from ..builder import ATTENTION

# helpers

def exists(val):
    return val is not None

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    return (t * alpha).softmax(dim = dim)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max
# classes

@ATTENTION.register_module()
class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., stable = False,
                 static_mask = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal
        self.register_buffer('static_mask', static_mask, persistent=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None, cache = None, cache_key = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax
        offset = cache.get('offset', 0) if exists(cache) else 0

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)



        q = q * self.scale

        if offset > 0:
            k_top, v_top = cache[cache_key]
            k = torch.cat([k_top, k], dim=-2)
            v = torch.cat([v_top, v], dim=-2)
        if exists(cache):
            cache[cache_key] = k, v

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal and offset == 0:  # causality is naturally enforced for the cached inference
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        if exists(self.static_mask):
            dots.masked_fill_(~self.static_mask[offset:offset + n, :offset + n], mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

#  cross attention  该模块用于测试图文cross attention的效果、
#  kv 为原图片，q为文字
@ATTENTION.register_module()
class CrossAttention(nn.Module):
    def __init__(self,
                 dim=768,
                 seq_len=196,
                 causal=True,
                 heads=8,
                 dim_head=64,
                 attn_dropout_ratio=0.,
                 proj_dropout_ratio=0.,
                 stable=False,
                 qk_scale=None,
                 static_mask=None):
        super().__init__()
        # qkv 的 dim
        inner_dim = dim_head *  heads

        self.heads = heads
        self.seq_len = seq_len

        # qk 计算的scale 值，若为Nne则使用dim_head计算出的scale
        self.scale = dim_head ** -0.5 if not qk_scale else qk_scale

        self.stable = stable
        self.causal = causal
        self.register_buffer('static_mask', static_mask, persistent=False)

        # qkv 的投影层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)


        # 输出层投影
        self.proj = nn.Linear(inner_dim, dim, bias=False)

        # drop out 设置
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)
        self.proj_dropout = nn.Dropout(proj_dropout_ratio)

    def forward(self,q,k,v):
        b,n,c = q.shape[0],q.shape[1],q.shape[2]

        # qkv 统一投影到 inner_dim
        q = self.to_q(q) 
        k = self.to_k(k)
        v = self.to_v(v)

        # qk attn 操作,得到注意力矩阵
        attn = (q@k.transpose(-2,-1))/self.scale
        # 对 attn 进行 softmax,这里不需要进行mask操作
        attn = attn.softmax(dim=-1)

        # 对 attn 进行 dropout
        attn = self.attn_dropout(attn)

        x = attn@v
        x = self.proj(x)

        # proj 进行drop out
        x = self.proj_dropout(x)


        return x 









# sparse attention with convolutional pattern, as mentioned in the blog post. customizable kernel size and dilation

@ATTENTION.register_module()
class SparseConvCausalAttention(nn.Module):
    def __init__(self, dim, seq_len, image_size = 32, kernel_size = 5, dilation = 1, heads = 8, dim_head = 64, dropout = 0., stable = False, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be odd'

        inner_dim = dim_head *  heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.stable = stable

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None):
        b, n, _, h, img_size, kernel_size, dilation, seq_len, device = *x.shape, self.heads, self.image_size, self.kernel_size, self.dilation, self.seq_len, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        img_seq_len = img_size ** 2
        text_len = seq_len + 1 - img_seq_len

        # padding

        padding = seq_len - n + 1
        mask = default(mask, lambda: torch.ones(b, text_len, device = device).bool())

        x = F.pad(x, (0, 0, 0, padding), value = 0)
        mask = mask[:, :text_len]

        # derive query / keys / values

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        q *= self.scale

        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # text attention

        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # image attention

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding = effective_kernel_size // 2

        k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h = img_size), (k_img, v_img))
        k_img, v_img = map(lambda t: F.unfold(t, kernel_size, padding = padding, dilation = dilation), (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = kernel_size ** 2), (k_img, v_img))

        # let image attend to all of text

        dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
        dots_image_to_text = einsum('b i d, b j d -> b i j', q_img, k_text)

        # calculate causal attention for local convolution

        i, j = dots_image.shape[-2:]
        img_seq = torch.arange(img_seq_len, device = device)
        k_img_indices = rearrange(img_seq.float(), '(h w) -> () () h w', h = img_size)
        k_img_indices = F.pad(k_img_indices, (padding,) * 4, value = img_seq_len) # padding set to be max, so it is never attended to
        k_img_indices = F.unfold(k_img_indices, kernel_size, dilation = dilation)
        k_img_indices = rearrange(k_img_indices, 'b j i -> b i j')

        # mask image attention

        q_img_indices = rearrange(img_seq, 'i -> () i ()')
        causal_mask =  q_img_indices < k_img_indices

        # concat text mask with image causal mask

        causal_mask = repeat(causal_mask, '() i j -> b i j', b = b * h)
        mask = repeat(mask, 'b j -> (b h) i j', i = i, h = h)
        mask = torch.cat((~mask, causal_mask), dim = -1)

        # image can attend to all of text

        dots = torch.cat((dots_image_to_text, dots_image), dim = -1)
        dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim = -1)

        # aggregate

        attn_image_to_text, attn_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b i j, b i j d -> b i d', attn_image, v_img)
        out_image_to_text = einsum('b i j, b j d -> b i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out =  self.to_out(out)
        return out[:, :n]
