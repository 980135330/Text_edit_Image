import torch.nn as nn
from ..builder import LC_HEAD
from einops import rearrange
import ipdb
# MAE最后的分类头，接受transformer输出的token，将token重新变成像素展示
@LC_HEAD.register_module()
class Pixel_head(nn.Module):

    def __init__(self, in_channels=768, out_channels=768, image_size=224,patch_size=16):
        super(Pixel_head, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)


        self.image_size = image_size
        self.patch_size = patch_size


    def forward(self, x):
        B, N ,C = x.shape
        #重新将输出范围压缩到-1到1
        # 重新展开成图片
        # 试了很多次才得到的正确展开
        # 先取出c
        x = rearrange(x,'b n (c p) ->b c n p',c=3, p=self.patch_size**2)
        # 从seq_len中分解出长宽
        x = rearrange(x,"b c (h w) p ->b c h w  p ",h=self.image_size//self.patch_size,w=self.image_size//self.patch_size)
        # 最后展开为原图
        x = rearrange(x,"b c h w (p1 p2) ->b c (h p1) (w p2)",p1=self.patch_size,p2=self.patch_size)

        return x