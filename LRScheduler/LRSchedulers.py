import math
from .builder import LRSCHEDULER
from torch.optim.lr_scheduler import _LRScheduler

@LRSCHEDULER.register_module()
class Cos_LR_Scheduler(_LRScheduler):
    def __init__(self,
                 optimizer=None,
                 max_epoch=None,
                 iter_num = None,
                 warm=0.2,
                 last_epoch=-1,):

        assert optimizer is not None
        assert max_epoch is not None
        assert iter_num is not None

        self.max_iter = max_epoch * iter_num
        self.warmup_iter = int(self.max_iter * warm)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iter) /
                          (self.max_iter - self.warmup_iter))) / 2
            for base_lr in self.base_lrs
        ]
