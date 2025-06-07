#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/hdd/andrew/zhiyong.ma/work/duc
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model srresnet \
    --batch_size 32 \
    --nb_epochs 50 \
    --image_size 224 \
    --lr 0.01 \
    --loss mse \
    --source_noise_model gaussian,0,50 \
    --target_noise_model clean \
    --val_noise_model duc,10 \
    --output_path output/srresnet_duc_10_noise_clean
