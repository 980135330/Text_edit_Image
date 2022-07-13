#!/bin/bash

python -m torch.distributed.launch  --nproc_per_node=2  --master_port=29523   mae_tgif.py 