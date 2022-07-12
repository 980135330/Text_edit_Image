from matplotlib import image
import torch.nn as nn
work_dir = "./"

use_wandb = False


batch_size = 16
epoch = 10
mode = "train"
# MAE transformer 设置
img_size = 224
embed_dim = 1024
patch_size = 8
num_heads=16
dim_head=embed_dim//num_heads
depth=24

num_worker = 16

seq_len=(img_size//patch_size)**2
# lr和warmup_lr的比例
lr = 1e-4
warm = 0.2

# 是否使用多机训练
dist = True

# 恢复训练和模型初始化设置,如果不为None则读取模型
#resume_from = work_dir + "checkpoint/epoch_3.pth"
# resume_from = "/home/chenghua.zhou/shiyan/Text2Img/classification/checkpoint/epoch_3.pth"
resume_from = None
init_model_from = None

# 储存checkpoint的频率
checkpoint_freq = 5



exp = dict(

    type="EXP_MAE_edit",
    work_dir = work_dir,
    epoch=epoch,
    mode=mode,
    use_wandb=use_wandb,
    dist=dist,

    resume_from = resume_from,
    init_model_from = init_model_from,

    checkpoint_freq = checkpoint_freq,

    # dataprovider 设置
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
    ),

    # 模型设置
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
                    )),
    optimizer_cfg=dict(
        type="Adam",
        lr=lr,
    ),
    lr_scheduler_cfg=dict(
        type="Cos_LR_Scheduler",
        warm=warm,
    ),
    loss_cfg=dict(type="MSE_Loss", ),

    dist_cfg=dict(  backend='nccl', 
                    
                    ),

    logger_cfg = dict(
        filename="log.txt",
        format='%(asctime)s - %(message)s',
        filemode='w',
    ),
)
