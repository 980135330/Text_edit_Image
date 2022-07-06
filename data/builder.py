# 用于控制data模块的Registry类注册
import sys
sys.path.append("..")
from utils import Registry,build_from_cfg

DATASET = Registry('dataset')
PIPLINE = Registry('pipeline')
DATAPROVIDER = Registry('dataprovider')

def build_dataprovider(cfg):
    return build_from_cfg(cfg,DATAPROVIDER)

def build_dataset(cfg):
    return build_from_cfg(cfg,DATASET)

def build_pipline(cfg):
    return build_from_cfg(cfg,PIPLINE)
    


