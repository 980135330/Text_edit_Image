import json
import os

import clip
import numpy as np
import torch
from PIL import Image, ImageSequence
from torch.utils.data import DataLoader, Dataset
from ..builder import DATASET


@DATASET.register_module()
class TGIFDataSet(Dataset):
    def __init__(self,data_json_path=None):
        assert data_json_path is not None
        self.data_path = data_json_path
        with open(data_json_path+"train_data.json",'r') as file:
            content = file.read()
        self.data_dict = data_dict = json.loads(content)
        self.data_list = list(data_dict.keys())
        self.data_len = len(self.data_list)
    
    def __len__(self):
        return self.data_len 
    
    def __getitem__(self,idx):
        
        gif_path = self.data_path + self.data_list[idx][2:]
        
        caption = self.data_dict[self.data_list[idx]]

        # 在数据集时直接对caption进行tokenizer
        caption = clip.tokenize(caption)
        gif = Image.open(gif_path)
        
        frames = ImageSequence.all_frames(gif)
        
        img = frames[0]
        gt_idx = np.random.randint(int(len(frames)*0.7),len(frames))
        img_gt = frames[gt_idx]
        
        return  caption,img,img_gt
        
