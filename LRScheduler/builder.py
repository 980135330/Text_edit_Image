import sys
sys.path.append("..")
from utils import Registry,build_from_cfg

LRSCHEDULER = Registry('lrscheduler')

def  build_lr_scheduler(cfg):
    return build_from_cfg(cfg,LRSCHEDULER)