work_dir = "./"

use_wandb = False


batch_size = 64
epoch = 200
mode = "train"
img_size = 32
patch_size = 8
embed_dim = 768
num_worker = 8
num_class = 100

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
checkpoint_freq = 10



exp = dict(

    type="EXP_Classification",
    work_dir = work_dir,
    epoch=epoch,
    mode=mode,
    use_wandb=use_wandb,
    dist=dist,

    resume_from = resume_from,
    init_model_from = init_model_from,

    checkpoint_freq = checkpoint_freq,

    dataprovider_cfg=dict(
        type="CIFAR100",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        dist=dist,
        dataset=dict(
            dict(
                #root='/content/drive/MyDrive/Colab Notebooks/Basic Network/my_code/data/',
                root='../dataset/',
                download=True,
                train=True),
            type='CIFAR100_train',
        ),
        piplines=[dict(type="Normal")],
    ),
    model_cfg=dict(type="Classification",
                   patch_embed_cfg=dict(
                       type="PatchEmbed",
                       img_size=img_size,
                       patch_size=patch_size,
                       in_c=3,
                       embed_dim=embed_dim,
                   ),
                   backbone_cfg=dict(
                       type="VisionTransformer",
                       num_classes=num_class,
                       embed_dim=embed_dim,
                       depth=12,
                       num_heads=12,
                       mlp_ratio=4.0,
                       qkv_bias=True,
                       qk_scale=None,
                       representation_size=None,
                       distilled=False,
                       drop_ratio=0.,
                       attn_drop_ratio=0.,
                       drop_path_ratio=0.,
                       num_patches=(img_size // patch_size)**2,
                   ),
                   lc_head_cfg=dict(
                       type="Cls_head",
                       embed_dim=embed_dim,
                       num_class=num_class,
                   )),
    optimizer_cfg=dict(
        type="Adam",
        lr=lr,
    ),
    lr_scheduler_cfg=dict(
        type="Cos_LR_Scheduler",
        warm=warm,
    ),
    loss_cfg=dict(type="CrossEntropyLoss", ),

    dist_cfg=dict(  backend='nccl', 
                    
                    ),

    logger_cfg = dict(
        filename="log.txt",
        format='%(asctime)s - %(message)s',
        filemode='w',
    ),
)
