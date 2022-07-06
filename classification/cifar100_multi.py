import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import wandb
import Vit_config as cfg
sys.path.append("..")
from exps  import build_exps

if __name__ == '__main__':

    # recive args
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', 
                        type=int, 
                        default=128, 
                        help='input batch size')

    parser.add_argument('-num_workers',
                        type=int,
                        default=None,
                        help='number of workers')
    parser.add_argument('-lr',
                        type=float,
                        default=None,
                        help='initial learning rate')
    parser.add_argument('-warm',
                        type=int,
                        default=None,
                        help='warm up training phase')
    parser.add_argument('-epochs',
                        type=int,
                        default=None,
                        help='number of epochs to train')
    parser.add_argument('-gpu',
                        action='store_true',
                        default=False,
                        help='use gpu or not')
    parser.add_argument('-resume',
                        action='store_true',
                        default=False,
                        help='resume training or not')
    parser.add_argument('-wandb',
                        action='store_true',
                        default=False,
                        help='use wandb or not')
    parser.add_argument('-experiment_name',
                        type=str,
                        help='experiment name')
    parser.add_argument('--local_rank', type=int, default=0) 

    args = parser.parse_args()

    # 多机训练分配RANK号，如果没有则使用torch.launch 为我们分配的rank号
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 如果没有checkpoint文件夹，则创建
    checkpoint_path = os.path.join(cfg.work_dir, "checkpoint")
    if not os.path.exists(checkpoint_path):
        try:
            os.makedirs(checkpoint_path) 
        except OSError as e:
            print("mkdir failed",e)
                   


    exp = build_exps(cfg.exp)
    print("exp:  ",{cfg.exp['resume_from']})
    exp.run()

    





