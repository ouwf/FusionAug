#!/bin/bash

seed=1
dataset=FVUSM
data="/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-processed"
network=resnet18
loss=fusion
python3 -u ./train.py \
    --seed $seed \
    --dataset $dataset --data $data --network $network --loss ${loss} \
    --pretrained \
    --inter_aug "TB" \
    --intra_aug \
    2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}.txt
