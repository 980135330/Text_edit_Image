from torch import nn
from ..builder import LC_HEAD

@LC_HEAD.register_module()
class Avg_lc_head(nn.Module):

    def __init__(self,output_channel = 512,num_class = 100) -> None:
        super().__init__()
        # 自适应平均池化层，将BxCxHxW的输入转换为BxCx1x1的输出
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(output_channel,num_class)

    def forward(self,x):
        x = self.avg(x)
        # 将BxCx1x1的输出转换为BxC的输出
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x