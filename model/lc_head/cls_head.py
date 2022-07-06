import torch.nn as nn 
from ..builder import LC_HEAD

@LC_HEAD.register_module()
class  Cls_head(nn.Module):
    def __init__(self,embed_dim = None,num_class = None):
        super().__init__()

        assert embed_dim is not None and num_class is not None
        assert num_class is not None

        self.head = nn.Linear(
            embed_dim,
            num_class
            ) if num_class > 0 else nn.Identity()
    def forward(self,x):
        x = self.head(x)
        return x

