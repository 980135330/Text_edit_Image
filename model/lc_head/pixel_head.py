import torch.nn as nn
from ..builder import LC_HEAD
import ipdb
# MAE最后的分类头，接受transformer输出的token，将token重新变成像素展示
@LC_HEAD.register_module()
class Pixel_head(nn.Module):

    def __init__(self, in_channels=768, out_channels=768, image_size=224):
        super(Pixel_head, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.act_layer = nn.Tanh()


        self.image_size = image_size


    def forward(self, x):
        B, N ,C = x.shape
        x = self.fc1(x)
        #重新将输出范围压缩到-1到1
        x = self.act_layer(x)

        ipdb.set_trace()


        # 重新展开成图片
        x = x.view(B, 3, self.image_size, self.image_size)

        return x