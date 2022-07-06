import torch.nn as nn
from .builder import LOSS

LOSS._register_module(name = "CrossEntropyLoss",module_class = nn.CrossEntropyLoss)