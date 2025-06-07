#!/bin/bash
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model srresnet \
    --batch_size 64 \
    --nb_epochs 50 \
    --image_size 128 \
    --lr 0.01 \
    --loss mse \
    --source_noise_model gaussian,0,50 \
    --target_noise_model clean \
    --val_noise_model gaussian,10,10 \
    --output_path output/srresnet_gaussian_0_50_noise_clean_128
