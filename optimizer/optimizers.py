import torch 
from .builder import OPTIMIZER

# 注册optimizer组件
OPTIMIZER._register_module(name="Adam", module_class = torch.optim.Adam)
OPTIMIZER._register_module(name="SGD", module_class = torch.optim.SGD)

