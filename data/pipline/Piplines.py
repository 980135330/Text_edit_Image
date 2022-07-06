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

# 不同的pipline操作，对数据进行预处理
@PIPLINE.register_module()
class Normal:

    def __init__(self):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __call__(self,data):

        return self.tf(data)
