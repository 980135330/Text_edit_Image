exp = dict(

    type = "EXP_Classification",
    epoch = 10,
    mode = "train",
    use_wandb = False,

    dataprovider_cfg = dict(
            type = "CIFAR100",
            batch_size = 64,
            shuffle = True,
            num_workers = 8,
            dataset = dict(
            dict(
            root='/content/drive/MyDrive/Colab Notebooks/Basic Network/my_code/data/',
            train=True),
            type = 'CIFAR100_train',
            ),
            piplines = [
                dict(type = "Normal")
            ],
    ),   

    model_cfg = dict(
            type = "Classification",
            backbone_cfg = dict(
                type  = "ResNet50",
                input_channel=3,
                num_blocks=[3, 4, 6, 3],
                channels = [64,128,256,512]
            ),
            lc_head_cfg = dict(
                type = "Avg_lc_head",
                output_channel = 512,
                num_class = 100
            )
    ),
    optimizer_cfg = dict(
        type = "Adam",
        lr = 0.001,
    ),
    lr_scheduler_cfg = dict(
        type = "Cos_LR_Scheduler",
        warm = 0.2,
    ),
    loss_cfg = dict(
        type = "CrossEntropyLoss",
        
    ),

)