#!/bin/bash
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model unet \
    --batch_size 128 \
    --nb_epochs 100 \
    --image_size 224 \
    --lr 0.01 \
    --loss mse \
    --source_noise_model gaussian,0,50 \
    --target_noise_model clean \
    --val_noise_model gaussian,10,10 \
    --output_path output/unet_gaussian_0_50_noise_clean