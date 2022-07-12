import os
import sys
from numpy import NaN

import torch
from data import dataprovider

from .builder import EXPS

sys.path.append('..')
import logging
import time

import torch
from data import build_dataprovider
from loss import build_loss
from LRScheduler import build_lr_scheduler
from model import build_model
from optimizer import build_optimizer
from torch.distributed import get_rank
from torch.nn.parallel import DistributedDataParallel
from utils import init_dist


def reduce_tenosr(tensor):
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        
        return tensor

@EXPS.register_module()
class EXP_MAE_edit:

    def __init__(self,
                 work_dir = None,
                 epoch=10,
                 mode="train",
                 use_wandb=False,
                 dataprovider_cfg=None,
                 model_cfg=None,
                 optimizer_cfg=None,
                 lr_scheduler_cfg=None,
                 loss_cfg=None,
                 dist_cfg=None,
                 logger_cfg=None,
                 use_GPU=True,
                 dist=False,

                 resume_from = None,
                 init_model_from = None,
                 checkpoint_freq = 10,
                 ):

        self.epoch = epoch
        self.mode = mode
        self.use_GPU  = use_GPU
        self.use_wandb = use_wandb
        self.dist = dist

        self.model = None
        self.dataprovider = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss = None

        self.dataprovider_cfg = dataprovider_cfg
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.loss_cfg = loss_cfg

        self.checkpoint_freq = checkpoint_freq
        self.device = None

        self.resume_from = resume_from
        self.init_model_from = init_model_from

        # checkpoint 保存路径
        self.checkpoint_path = os.path.join(work_dir, "checkpoint")
        # 若不存在，则创建checkpoint 文件夹
        if not self.dist or get_rank()==0:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
        
        # 恢复训练的epoch，默认从1开始
        self.resume_epoch = 0

        # logger 相关设置
        logging.basicConfig(**logger_cfg)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)


    def build_model(self):
        assert self.model_cfg is not None
        self.model = build_model(self.model_cfg)

        # 如果有传入模型的初始化路径，则加载对应的模型初始化权重
        # if self.init_model_from is not None:
        #     if not os.path.exists(self.init_model_from):
        #         raise Exception("init_model_from path not exists")
            
        #     self.model.load_state_dict(torch.load(self.init_model_from)['model_state_dict'])
        # if self.resume_from is not None:
        #     if not os.path.exists(self.resume_from):
        #         raise Exception("resume_from path not exists")
        #     self.model.load_state_dict(torch.load(self.resume_from)['model_state_dict'])


    def build_dataprovider(self):
        assert self.dataprovider_cfg is not None
        self.dataprovider = build_dataprovider(self.dataprovider_cfg).get_loaders()

    def build_optimizer(self,model):
        assert self.optimizer_cfg is not None
        self.optimizer = build_optimizer(model.parameters(),self.optimizer_cfg)

    def build_lr_scheduler(self):
        self.lr_scheduler = build_lr_scheduler(self.lr_scheduler_cfg)

    def build_loss(self):
        assert self.loss_cfg is not None
        self.loss = build_loss(self.loss_cfg)

    def save_checkpoint(self,epoch):

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'warmup_scheduler': self.lr_scheduler.state_dict(),
            }, os.path.join(self.checkpoint_path, "epoch_{}.pth".format(epoch)))

    # 同步多机训练数据，这里用来累加数据

    # def build_parallel_model(self):
    # 多机训练初始化模型
    # device = torch.device("cuda",get_rank())
    # self.build_model()
    # if isinstance(self.model,DistributedDataParallel):
    #     return
    # print("*"*20)
    # print(torch.cuda.current_device())
    # self.model = DistributedDataParallel(self.model.to(device),device_ids=[0,1,2,3,4,5,6,7])
    # print("*"*20)
    # print(type(self.model))
    # print(isinstance(self.model,DistributedDataParallel))






    def train(self):

        # if self.dist:
        # assert dist_cfg is not None
        # init_dist('pytorch', **dist_cfg)

        # torch.distributed.init_process_group(backend='nccl')
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")

        # local_rank = get_rank()
        # torch.cuda.set_device(local_rank)
        # self.device = torch.device("cuda", local_rank)


        # if self.dist:
        #     device = torch.device("cuda",get_rank())
        
        # 先释放显存
        torch.cuda.empty_cache()


        # 多机训练的模型初始化
        if self.dist:
            torch.distributed.init_process_group(backend='nccl')
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")

            local_rank = get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            self.build_model()
            self.model = self.model.to(device)

            if not self.resume_from:
                self.model = DistributedDataParallel(self.model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters = True)
            
            print("current GPU",{local_rank},"\n")

        else:
            self.build_model()
            if self.use_GPU:
                self.model = self.model.cuda()


        self.build_dataprovider()

        self.build_optimizer(self.model)


        self.lr_scheduler_cfg["optimizer"] = self.optimizer
        self.lr_scheduler_cfg["max_epoch"] = self.epoch
        self.lr_scheduler_cfg["iter_num"] = len(self.dataprovider)

        self.build_lr_scheduler()
        self.build_loss()

        # 是否恢复训练，如果设置了恢复训练的选项，则初始化model、epoch、optimizer、lr_scheduler的状态
        if self.resume_from:
    
            if not os.path.exists(self.resume_from):
                raise FileNotFoundError("resume_from path not exists")
            
            checkpoint = torch.load(self.resume_from)


            self.logger.info("loading model from {}".format(self.resume_from))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['warmup_scheduler'])
            self.resume_epoch = checkpoint['epoch'] 
            # if self.dist:
            #     self.model = DistributedDataParallel(self.model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters = True)

            print("Resume_Epoch", self.resume_epoch)

        
        # 模型训练过程
        for epoch in range(1,self.epoch+1):
            
            # 小于resume_epoch，则直接跳过
            if epoch<= self.resume_epoch:
                continue

            
            # dataprovider提供text和image,生成image_gt
            for step,(text,image,image_gt) in enumerate(self.dataprovider):
                start_time = time.time()
                if self.use_GPU:
                    if self.dist:
                        text = text.to(device)
                        image = image.to(device)
                        image_gt = image_gt.to(device)
                    else:
                        text = text.cuda()
                        image = image.cuda()
                        image_gt = image_gt.cuda()

                self.optimizer.zero_grad()
                self.model.train()

                output = self.model(text,image)

                loss = self.loss(output,image_gt)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                end_time = time.time()
                if not self.dist or get_rank()==0:
                    self.logger.info(
                        "Epoch [{epoch}/{max_epoch}], Step [{step}/{iter_num}], Loss: {loss:.7f}, lr: {lr:.7f}, time: {time}\n".format(
                            epoch=epoch,
                            max_epoch=self.epoch,
                            step=step+1,
                            iter_num=len(self.dataprovider),
                            loss=loss.item(),
                            lr=self.optimizer.param_groups[0]['lr'],
                            time=end_time - start_time)
                            )

  

            # 在对应的epoch结束后，保存checkpoint
            if epoch % self.checkpoint_freq == 0:
                # 只让一个GPU保存
                if not self.dist or get_rank() == 0:
                    self.save_checkpoint(epoch)


    def run(self):
        if self.mode == "train":
            self.train()
