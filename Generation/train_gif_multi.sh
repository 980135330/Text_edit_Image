#!/bin/bash

python -m torch.distributed.launch  --nproc_per_node=8  --master_port=29523   mae_tgif.py -dist