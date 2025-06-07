# Noise2Noise PyTorch Implementation

A PyTorch implementation of the Noise2Noise denoising method. This project has been completely rewritten from Keras/TensorFlow to PyTorch for better performance and modern deep learning practices.

## Overview

Noise2Noise is a technique for training image denoising models without clean target images. Instead, it trains on pairs of noisy images, learning to map from one noisy version to another noisy version of the same clean image.

## Features

- **Two Model Architectures**: SRResNet and U-Net
- **Multiple Noise Types**: Gaussian, impulse, text overlay, and clean (identity)
- **Flexible Loss Functions**: MSE, MAE, and L0 loss with annealing
- **PyTorch DataLoader**: Efficient data loading with multi-processing
- **TensorBoard Integration**: Real-time training monitoring
- **Comprehensive Testing**: Model evaluation with visual results
- **Training Visualization**: Plot training history and metrics

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.3
- opencv-python >= 4.5.0
- Pillow >= 8.0.0
- tqdm >= 4.60.0
- matplotlib >= 3.5.0

## Project Structure

```
noise2noise/
├── model.py           # Model architectures (SRResNet, U-Net)
├── train.py           # Training script
├── test_model.py      # Testing/inference script
├── generator.py       # PyTorch Dataset and DataLoader
├── noise_model.py     # Noise generation functions
├── plot_history.py    # Training history visualization
├── requirements.txt   # Dependencies
├── README.md         # This file
└── dataset/          # Training and validation data
    ├── 291/          # Training images
    ├── Set5/         # Validation set
    ├── Set14/        # Test set
    └── ...
```

## Usage

### 1. Training

Train a model using the training script:

```bash
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model srresnet \
    --batch_size 16 \
    --nb_epochs 60 \
    --lr 0.01 \
    --loss mse \
    --source_noise_model gaussian,0,50 \
    --target_noise_model gaussian,0,50 \
    --val_noise_model gaussian,25,25 \
    --output_path checkpoints
```

#### Training Parameters

- `--image_dir`: Directory containing training images
- `--test_dir`: Directory containing validation images
- `--model`: Model architecture (`srresnet` or `unet`)
- `--batch_size`: Training batch size (default: 16)
- `--nb_epochs`: Number of training epochs (default: 60)
- `--lr`: Learning rate (default: 0.01)
- `--loss`: Loss function (`mse`, `mae`, or `l0`)
- `--source_noise_model`: Noise model for source images
- `--target_noise_model`: Noise model for target images
- `--val_noise_model`: Noise model for validation
- `--output_path`: Directory to save checkpoints

#### Noise Models

Noise models are specified as strings with parameters:

- **Gaussian**: `gaussian,min_std,max_std` (e.g., `gaussian,0,50`)
- **Impulse**: `impulse,min_occupancy,max_occupancy` (e.g., `impulse,5,20`)
- **Text**: `text,min_occupancy,max_occupancy` (e.g., `text,10,30`)
- **Clean**: `clean` (no noise, identity function)

### 2. Testing

Test a trained model on images:

```bash
python test_model.py \
    --image_dir dataset/Set14 \
    --model srresnet \
    --weight_file checkpoints/weights.059-0.001-35.12345.pth \
    --test_noise_model gaussian,25,25 \
    --output_dir results
```

#### Testing Parameters

- `--image_dir`: Directory containing test images
- `--model`: Model architecture (must match training)
- `--weight_file`: Path to trained model weights
- `--test_noise_model`: Noise model for test images
- `--output_dir`: Directory to save results (optional, shows images if not specified)

### 3. Visualizing Training History

Plot training metrics from saved history:

```bash
python plot_history.py \
    --history_file checkpoints/history.npz \
    --output_dir plots \
    --show
```

### 4. Testing Noise Models

Test different noise models visually:

```bash
python noise_model.py \
    --noise_model gaussian,0,50 \
    --image_size 256
```

## Model Architectures

### SRResNet
- Based on Super-Resolution ResNet
- 16 residual blocks with skip connections
- PReLU activation and batch normalization
- Suitable for general denoising tasks

### U-Net
- Encoder-decoder architecture with skip connections
- Configurable depth and channel progression
- Optional dropout, batch normalization, and residual connections
- Better for preserving fine details

## Training Features

### Learning Rate Scheduling
Automatic learning rate decay:
- Epochs 0-25%: Full learning rate
- Epochs 25-50%: 0.5× learning rate
- Epochs 50-75%: 0.25× learning rate
- Epochs 75-100%: 0.125× learning rate

### L0 Loss Annealing
When using L0 loss, the gamma parameter is automatically annealed:
```
gamma = 2.0 * (total_epochs - current_epoch) / total_epochs
```

### Checkpointing
- Automatic saving of best model (highest validation PSNR)
- Periodic checkpoints every N epochs
- Full checkpoint includes model state, optimizer state, and metrics

### Monitoring
- TensorBoard logging for real-time monitoring
- Progress bars with loss and learning rate
- Validation metrics after each epoch

## Example Training Commands

### Basic Gaussian Denoising
```bash
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model srresnet \
    --source_noise_model gaussian,0,50 \
    --target_noise_model gaussian,0,50 \
    --val_noise_model gaussian,25,25
```

### Text Removal
```bash
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model unet \
    --source_noise_model text,10,30 \
    --target_noise_model clean \
    --val_noise_model text,20,20
```

### Impulse Noise Removal
```bash
python train.py \
    --image_dir dataset/291 \
    --test_dir dataset/Set5 \
    --model srresnet \
    --source_noise_model impulse,5,20 \
    --target_noise_model clean \
    --val_noise_model impulse,10,10
```

## Performance Tips

1. **GPU Usage**: The code automatically detects and uses CUDA if available
2. **Data Loading**: Adjust `--num_workers` based on your CPU cores
3. **Batch Size**: Increase batch size if you have sufficient GPU memory
4. **Image Size**: Larger patches (64-128) generally work better
5. **Steps per Epoch**: Adjust `--steps` to control epoch length

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir checkpoints/logs
```

### Real-time Metrics
The training script displays:
- Training loss per batch
- Validation loss and PSNR per epoch
- Learning rate updates
- Best model saves

## File Formats

- **Model Checkpoints**: `.pth` files containing full training state
- **Training History**: `.npz` files with numpy arrays
- **Images**: Supports JPEG, PNG, BMP formats

## Migration from Keras Version

This PyTorch implementation maintains compatibility with the original Keras version:

1. **Same CLI Arguments**: Most command-line arguments are identical
2. **Same Noise Models**: All noise generation functions preserved
3. **Same Model Architectures**: Faithful PyTorch translations
4. **Improved Performance**: Better GPU utilization and data loading

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Slow Data Loading**: Increase `--num_workers` or use SSD storage
3. **Poor Convergence**: Try different learning rates or loss functions
4. **Validation Issues**: Ensure validation images are large enough (>64px)

### Debug Mode
Add `--device cpu` to force CPU usage for debugging.

## License

This project maintains the same license as the original implementation.

## Citation

If you use this code, please cite the original Noise2Noise paper:

```
@inproceedings{lehtinen2018noise2noise,
  title={Noise2noise: Learning image restoration without clean data},
  author={Lehtinen, Jaakko and Munkberg, Jacob and Hasselgren, Jon and Laine, Samuli and Karras, Tero and Aittala, Miika and Aila, Timo},
  booktitle={International Conference on Machine Learning},
  pages={2965--2974},
  year={2018}
}
