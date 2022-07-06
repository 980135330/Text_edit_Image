import torch.nn as nn
from ..builder import BACKBONE

@BACKBONE.register_module()
class ResNet50(nn.Module):
    def __init__(self,
                 input_channel=3,
                 num_blocks=[3, 4, 6, 3],
                 channels = [64,128,256,512]
                 ):
        super(ResNet50, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel,channels[0],kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        # 按照ResNet50 的设计 分别生成4个resnet块
        self.conv2 = self.make_layer(channels[0],channels[0],num_blocks[0])
        self.conv3 = self.make_layer(channels[0],channels[1],num_blocks[1])
        self.conv4 = self.make_layer(channels[1],channels[2],num_blocks[2])
        self.conv5 = self.make_layer(channels[2],channels[3],num_blocks[3])

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x



    # 根据输入输出通道数和block数量生成对应的resnet块
    def make_layer(self,input_channel,output_channel,num_block):
        layers = []
        strides = [2] + [1]*(num_block-1)
        for stride in strides:
            layers.append(ResBlock(input_channel,output_channel,stride))
            input_channel = output_channel
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):

        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,
                               output_channel // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.conv2 = nn.Conv2d(output_channel // 4,
                               output_channel // 4,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        
        self.conv3 = nn.Conv2d(output_channel // 4,
                               output_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.block = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(output_channel // 4),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm2d(output_channel // 4),
            nn.ReLU(),
            self.conv3,
            nn.BatchNorm2d(output_channel),
        )
        
        self.shortcut = nn.Sequential()
        if stride!=1 or input_channel != output_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel,
                          output_channel,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(output_channel),
            )
    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out