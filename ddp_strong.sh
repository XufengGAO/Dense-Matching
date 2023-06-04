#!/usr/bin/env bash


benchmark="pfpascal"
backbone="resnet50"
nnodes=1
master_addr="10.233.66.69" #1
# master_addr="10.233.90.176" #2
# master_addr="10.233.70.135" #3
# master_addr="10.233.87.139" #4


master_port=12367

# CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1 \
                                    --nnodes=${nnodes} --node_rank=$1 \
                                    --master_addr=${master_addr} \
                                    train.py \
                                    --config './configs/train_strongce.py'