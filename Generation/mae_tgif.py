import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import wandb
import MAE_config as cfg
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

    parser.add_argument('-dist',
                        action='store_true',
                        default=False,
                        help='use dist or not')

    parser.add_argument('--local_rank',
                        type=int,
                        default=None,
                        help='local rank')
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

    args = parser.parse_args()
    # 通过 arg调整是否使用多卡训练
    cfg.dist = args.dist
                                
    exp = build_exps(cfg.exp)
    
    exp.run()

    



