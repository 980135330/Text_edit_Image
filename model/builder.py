# 用于控制model模块的Registry类注册
import sys

from torch import rad2deg
sys.path.append("..")
from utils import Registry,build_from_cfg


PATCH_EMBED = Registry('patch_embed')
BACKBONE = Registry('backbone')
LC_HEAD = Registry('lc_head')
VAE = Registry('vae')
AR_GEN = Registry('ar_gen')

MODEL = Registry('model')
ATTENTION = Registry('attention')

def build_patch_embed(cfg):
    if not cfg:
        return None
    return build_from_cfg(cfg,PATCH_EMBED)

def build_backbone(cfg):
    return build_from_cfg(cfg,BACKBONE)

def build_lc_head(cfg):
    if not cfg:
        return None
    return build_from_cfg(cfg,LC_HEAD)
 
# 返回vae模型的配置
def build_vae(cfg):
    return build_from_cfg(cfg,VAE)

# attention build
def build_attention(cfg):
    if not cfg:
        return None
    return build_from_cfg(cfg,ATTENTION)

def build_model(cfg):
    return build_from_cfg(cfg,MODEL)