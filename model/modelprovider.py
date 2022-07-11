import torch.nn as nn
from .builder import build_patch_embed,build_backbone,build_lc_head,MODEL

# 用于模型的搭建
@MODEL.register_module()
class  Classification(nn.Module):
    # 将分类模型分为 预处理、backbone、最后的输出映射 三部分
    def __init__(self,
                 patch_embed_cfg=None,
                 backbone_cfg=None,
                 lc_head_cfg=None):
        super(Classification, self).__init__()

        # 根据传入的cfg 获取对应的模块
        if not backbone_cfg:
            raise ValueError("backbone_cfg is None")


        self.patch_embed = build_patch_embed(patch_embed_cfg)

        self.backbone = build_backbone(backbone_cfg)
        self.lc_head = build_lc_head(lc_head_cfg)

    def forward(self,x):

        #  forward 过程
        # patch_embed为可选模块
        if self.patch_embed:
            x = self.patch_embed(x)
        x = self.backbone(x)
        x = self.lc_head(x)

        return x

@MODEL.register_module()
class  MAE_IMAGE_EDIT(nn.Module):
    def __init__(self,
                 patch_embed_cfg=None,
                 backbone_cfg=None,
                 lc_head_cfg=None):
        super(MAE_IMAGE_EDIT, self).__init__()

        # 根据传入的cfg 获取对应的模块
        if not backbone_cfg:
            raise ValueError("backbone_cfg is None")
        if not lc_head_cfg:
            raise ValueError("lc_head_cfg is None")

        # 输入文本的方案暂不对图片进行patch处理
        # self.patch_embed = build_patch_embed(patch_embed_cfg)
        self.backbone = build_backbone(backbone_cfg)
        self.lc_head = build_lc_head(lc_head_cfg)

    def forward(self,text,image):
            
        #  forward 过程
        # x = self.patch_embed(x)

        x = self.backbone(text,image)
        x = self.lc_head(x)

        return x
