a = 8888
dataprovider = dict(

        type = "CIFAR100",
        batch_size = 64,
        shuffle = True,
        num_workers = 8,
        dataset = dict(
        dict(
        root='C:/Users/lenovo/Desktop/代码复现/基础卷积模块/my_code/data',
        train=True),
        type = 'CIFAR100_train',
        ),
        piplines = [
            dict(type = "Normal")
        ],
    
)
