import sys
sys.path.append("..")
from utils import Registry,build_from_cfg

EXPS = Registry("exps")

def build_exps(cfg):
    return build_from_cfg(cfg,EXPS)