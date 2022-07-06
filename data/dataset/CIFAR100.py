import torchvision
from ..builder import DATASET

DATASET._register_module(
    # torchvision.datasets.CIFAR100(
    #     root='C:/Users/lenovo/Desktop/代码复现/基础卷积模块/my_code/data', train=True),
    torchvision.datasets.CIFAR100,
    name='CIFAR100_train',
)
DATASET._register_module(
    # torchvision.datasets.CIFAR100(
    #     root='C:/Users/lenovo/Desktop/代码复现/基础卷积模块/my_code/data', train=False),
    torchvision.datasets.CIFAR100,
    name='CIFAR100_test',
)
