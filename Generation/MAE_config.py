import torch.nn as nn
work_dir = "./"

use_wandb = False


batch_size = 64
epoch = 50
mode = "train"
img_size = 224
patch_size = 16
embed_dim = 768
num_worker = 8

# lr和warmup_lr的比例
lr = 1e-4
warm = 0.2

# 是否使用多机训练
dist = False

# 恢复训练和模型初始化设置,如果不为None则读取模型
#resume_from = work_dir + "checkpoint/epoch_3.pth"
# resume_from = "/home/chenghua.zhou/shiyan/Text2Img/classification/checkpoint/epoch_3.pth"
resume_from = None
init_model_from = None

# 储存checkpoint的频率
checkpoint_freq = 10



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
        piplines=[dict(type="RandomCrop",crop_size=img_size),dict(type="Normal")],
    ),

    # 模型设置
    model_cfg=dict(type="MAE_IMAGE_EDIT",
                    backbone_cfg=dict(
                        type="MAE_decoder",
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
                    ),
                    lc_head_cfg=dict(
                        type="Pixel_head",
                        in_channels=768, 
                        out_channels=768, 
                        image_size=224
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
