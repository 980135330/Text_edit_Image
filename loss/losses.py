import torch.nn as nn
from .builder import LOSS

LOSS._register_module(name = "CrossEntropyLoss",module_class = nn.CrossEntropyLoss)
LOSS._register_module(name = "MSE_Loss",module_class = nn.MSELoss)