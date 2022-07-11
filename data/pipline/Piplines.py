from ..builder import PIPLINE, build_pipline
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# 用于pipline处理的基类
class PiplineDataSet(Dataset):
    def __init__(self,dataset,piplines,dist):
        self.dataset = dataset
        self.piplines = piplines
        self.dist = dist

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        # 多机训练放入对应GPU

        data,label = self.dataset[index]

        # 遍历每个pipline，分别初始化对数据预处理的单元对数据进行操作
        for pipline_cfg in self.piplines:
            each = build_pipline(pipline_cfg)
            data = each(data)

        return data,label

# 用于生成任务  pipline处理的基类
class GeneratonPiplineDataSet(Dataset):
    def __init__(self,dataset,piplines,dist):
        self.dataset = dataset
        self.piplines = piplines
        self.dist = dist

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        # 多机训练放入对应GPU

        text,img,img_gt = self.dataset[index]

        # 遍历每个pipline，分别初始化对数据预处理的单元对数据进行操作
        for pipline_cfg in self.piplines:
            each = build_pipline(pipline_cfg)
            img = each(img)
            img_gt = each(img_gt)

        return text,img,img_gt

# 不同的pipline操作，对数据进行预处理
@PIPLINE.register_module()
class RandomCrop:
    def __init__(self,crop_size=None):
        assert crop_size is not None
        self.tf = transforms.Compose([
            transforms.RandomCrop(crop_size),
        ])
    def __call__(self,data):

        return self.tf(data)

@PIPLINE.register_module()
class Normal:

    def __init__(self):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __call__(self,data):

        return self.tf(data)

@PIPLINE.register_module()
class Resize:

    def __init__(self,resize_size=None):
        assert resize_size is not None
        self.tf = transforms.Compose([
            transforms.Resize((resize_size,resize_size))
        ])


    def __call__(self,data):

        return self.tf(data)