import torch 
import torch.nn as nn

from model import build_model
from data import build_dataprovider
import clip

from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

import ipdb
# 加载模型路径
epoch_path = "/mnt/datasets/tgif/mae_checkpoint/epoch_10.pth"
# 初始化dataloader
batch_size = 16
epoch = 10
mode = "train"
# MAE transformer 设置
img_size = 224
embed_dim = 1024
patch_size = 8
num_heads=16
dim_head=embed_dim//num_heads
depth=32
num_worker = 16
dist = False
seq_len=(img_size//patch_size)**2

dataprovider_cfg=dict(
        type="TGIF",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        dist=dist,
        # 使用的数据集
        dataset=dict(
            dict(
                #root='/content/drive/MyDrive/Colab Notebooks/Basic Network/my_code/data/',
            data_json_path = "/mnt/datasets/tgif/",
            type='TGIFDataSet',
        )),
        # 数据要使用的预处理
        piplines=[dict(type="Resize",resize_size=img_size),dict(type="Normal")],
    )

data_loader = build_dataprovider(dataprovider_cfg).get_loaders()

# 初始化模型
model_cfg=dict(type="MAE_IMAGE_EDIT",
                    backbone_cfg=dict(
                        type="MAE_decoder",
                        in_channels=3,
                        dim=embed_dim,
                        seq_len=seq_len,
                        patch_size=patch_size,
                        num_heads=num_heads,
                        dim_head=dim_head,
                        depth=depth,
                        qk_scale=None,
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        attn_dropout_ratio=0.,
                        proj_dropout_ratio=0.,
                    ),
                    lc_head_cfg=dict(
                        type="Pixel_head",
                        in_channels=embed_dim,
                        out_channels=(patch_size**2)*3, 
                        image_size=img_size,
                    ))
mae = build_model(model_cfg)


def load_checkpoint(epoch_path,model):
    check_point = torch.load(epoch_path)
    model.load_state_dict( {k.replace('module.', ''):v for k, v in check_point['model_state_dict'].items()})




@torch.no_grad()
def inference_debug(model):
    # 测试模型
    load_checkpoint(epoch_path,model)
    model = model.cuda()
    
    model.eval()
    for idx, (text, img,img_gt) in enumerate(data_loader):

        ipdb.set_trace()
        img = img.cuda()
        img_gt = img_gt.cuda()
        text = text.cuda()
        output = model(text,img)
        ipdb.set_trace()


if __name__ == "__main__":
    inference_debug(mae)
