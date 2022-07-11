from re import X
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from ..builder import BACKBONE

import clip


# 使用MAE的decoder的思想，尝试利用cross attention 解决图片编辑问题
@BACKBONE.register_module()
class MAE_decoder(nn.Module):
    def __init__(self, 
                    dim=768,
                    seq_len=196,
                    num_heads=8,
                    dim_head=64,
                    depth=12,
                    qk_scale=None,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    attn_dropout_ratio=0.,
                    proj_dropout_ratio=0.,
    ):
        super(MAE_decoder, self).__init__()

        # seq_len 即输入的token数量
        self.num_tokens = seq_len
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, dim))
        #初始化位置编码 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


        # 按深度叠加block
        self.blocks = nn.ModuleList([
            Block(  dim=dim,
                    seq_len=seq_len,
                    heads=num_heads,
                    dim_head=dim_head,
                    qk_scale=qk_scale,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attn_dropout_ratio=attn_dropout_ratio,
                    proj_dropout_ratio=proj_dropout_ratio,) for i in range(depth)
        ])

        self.norm = norm_layer(dim)

        # 使用clip 的 tokenizer 和 encoder 提取text特征
        self.text_token_model,_ =  clip.load("ViT-L/14")
        self.text_encoder = self.text_token_model.encode_text






    def forward(self, x, image):

        # 将位置编码加入输入，这里因为pos_embed第一维是1，所以会自动广播
        # 现在的策略暂时不需要对位置编码
        # x = x + self.pos_embed


        # clip encoder 抽取文本特征
        b,n,d = x.shape
        #  先去掉中中间的维度,过clip抽取特征
        x = x.view(b,-1)
        x = self.text_encoder(x)
        # 恢复原本的形状，第二维增加维度，变为 bx1xd
        # 由于clip抽出的特征是float16,所以要先.float()转换成32位
        x = x.unsqueeze(1).float()

        # 直接将预处理后的图片展开为BxNxC形式 作为q
        image = image.view(image.shape[0], self.num_tokens, -1)

        # 过block,kv都设置为image_gt
    
        for block in self.blocks:
            x = block(image,x)
       
        # 最后的norm
        x = self.norm(x)

        return x

# 组合MLP 和 attention 为一个transformer block
class Block(nn.Module):
    def __init__(self,
                    dim=768,
                    seq_len=196,
                    heads=8,
                    dim_head=64,
                    qk_scale=None,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    attn_dropout_ratio=0.,
                    proj_dropout_ratio=0.,
                    ):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.cross_attn = CrossAttention(dim=dim,
                                  seq_len=seq_len,
                                  heads=heads,
                                  dim_head=dim_head,
                                  qk_scale=qk_scale,
                                  attn_dropout_ratio=attn_dropout_ratio,
                                  proj_dropout_ratio=proj_dropout_ratio)
        self.mlp = MLP(in_channel=dim,
                       hidden_channel=dim * 2,
                       act_layer=act_layer,
                       dropout_ratio=proj_dropout_ratio)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, image,text_features):

        x = self.norm1(image)
        
        # cross attention操作，kv都设置为image_gt
        x = x + self.cross_attn(x,text_features,text_features)

        x = self.norm2(x)
        x = x + self.mlp(x)
        

        return x
        




#  transformer中的 MLP层，线性投影放大为hidden_channel后放回来
class MLP(nn.Module):
    def __init__(self, in_channel=768, hidden_channel=1024, act_layer=nn.GELU, dropout_ratio=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, in_channel)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)

        # 按照原始transformer，第二个线性层没有激活函数
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# cross attention 层
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


    # 要求每次输入的q,k,v都是batch_size * seq_len * dim
    # kv 由于 每批数据都不同，所以需要单独输入
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
