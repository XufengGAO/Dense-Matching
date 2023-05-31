#!/usr/bin/env bash


benchmark="pfpascal"
backbone="resnet50"
nnodes=1
master_addr="10.233.114.2"
master_port=12364

# CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1 \
                                    --nnodes=${nnodes} --node_rank=$1 \
                                    --master_addr=${master_addr} \
                                    ddp_train.py \
                                    --benchmark $benchmark \
                                    --backbone $backbone \
                                    --pretrain 'imagenet' \
                                    --alpha 0.1 \
                                    --criterion 'weak'\
                                    --lr 0.0001 \
                                    --freeze_backbone True \
                                    --epochs 100 \
                                    --batch_size 8 \
                                    --backbone_path './backbone/dino_resnet50.pth' \
                                    --temp 0.05 \
                                    --weak_lambda '[0.0, 0.0, 1.0]' \
                                    --collect_grad False \
                                    --entropy_func 'info_entropy'\
                                    --use_negative False \
                                    --match_norm_type 'l1' \
                                    --momentum 0.9 \
                                    --optimizer 'sgd' \
                                    --exp2 0.5 \
                                    --use_wandb True \
                                    --wandb_proj 'ddp_scot' \
                                    --loss_stage "sim" \
                                    --cam "mask/resnet101/200_300" \
                                    --output_image_size '(200,300)' 