import sys
sys.path.append("..")
from utils import Registry,build_from_cfg

LOSS = Registry('loss')

def build_loss(cfg):
    return build_from_cfg(cfg,LOSS)



