# 用于控制data模块的Registry类注册
from .builder import ( DATAPROVIDER, DATASET, PIPLINE,
                       build_dataset)
from torch.utils.data import DataLoader
from .pipline import PiplineDataSet,GeneratonPiplineDataSet
from torch.distributed import get_rank, get_world_size

import sys 
sys.path.append('..')
from utils import DistributedSampler


#注册TGIF的dataprovider
@DATAPROVIDER.register_module()
class TGIF:
    def __init__(
        self,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        dataset=None,
        piplines=None,
        dist = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset = dataset
        self.piplines = piplines
        self.dist = dist


    # 所有的参数都用cfg代替
    # 获取dataset,然后根据选项是否对dataset piplines处理
    # 最后返回dataloader
    def get_loaders(self):
        if not self.dataset:
            raise ValueError("dataset is None")


        dataset = build_dataset(self.dataset)
        if self.piplines:

            dataset = GeneratonPiplineDataSet(dataset,self.piplines,self.dist)
            
        sampler = None
        if self.dist:
            sampler = DistributedSampler(dataset)
            self.shuffle = False

        # # 多机训练配置
        # if self.dist:
        #     rank = get_rank()
        #     world_size = get_world_size()

        #     # 多机训练的采样器
        #     sampler = DistributedSampler(
        #         dataset, world_size, rank, shuffle=self.shuffle)
        #     self.shuffle = False

        # 设置多机训练的dataloader
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                sampler=sampler if self.dist else None,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers)

        return dataloader

# 注册CIFAR100的dataprovider
@DATAPROVIDER.register_module()
class CIFAR100:
    def __init__(
        self,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        dataset=None,
        piplines=None,
        dist = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset = dataset
        self.piplines = piplines
        self.dist = dist


    # 所有的参数都用cfg代替
    # 获取dataset,然后根据选项是否对dataset piplines处理
    # 最后返回dataloader
    def get_loaders(self):
        if not self.dataset:
            raise ValueError("dataset is None")


        dataset = build_dataset(self.dataset)
        if self.piplines:

            dataset = PiplineDataSet(dataset,self.piplines,self.dist)
            
        sampler = None
        if self.dist:
            sampler = DistributedSampler(dataset)
            self.shuffle = False

        # # 多机训练配置
        # if self.dist:
        #     rank = get_rank()
        #     world_size = get_world_size()

        #     # 多机训练的采样器
        #     sampler = DistributedSampler(
        #         dataset, world_size, rank, shuffle=self.shuffle)
        #     self.shuffle = False

        # 设置多机训练的dataloader
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                sampler=sampler if self.dist else None,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers)

        return dataloader
