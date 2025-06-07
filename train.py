import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import get_model, L0Loss, psnr
from generator import create_data_loaders
from noise_model import get_noise_model


class LearningRateScheduler:
    def __init__(self, optimizer, nb_epochs, initial_lr):
        self.optimizer = optimizer
        self.nb_epochs = nb_epochs
        self.initial_lr = initial_lr

    def step(self, epoch):
        if epoch < self.nb_epochs * 0.25:
            lr = self.initial_lr
        elif epoch < self.nb_epochs * 0.50:
            lr = self.initial_lr * 0.5
        elif epoch < self.nb_epochs * 0.75:
            lr = self.initial_lr * 0.25
        else:
            lr = self.initial_lr * 0.125
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=64,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; 'mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for source images")
    parser.add_argument("--target_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for target images")
    parser.add_argument("--val_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for validation source images")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of data loading workers")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="save model every N epochs")
    parser.add_argument("--device", type=str, default="auto",
                        help="device to use ('cuda', 'cpu', or 'auto')")
    args = parser.parse_args()
    return args


def validate_model(model, val_loader, criterion, device):
    """Validate the model and return average loss and PSNR"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            
            # Calculate loss
            loss = criterion(output, clean)
            total_loss += loss.item()
            
            # Calculate PSNR (convert back to [0, 255] range)
            output_255 = torch.clamp(output * 255.0, 0, 255)
            clean_255 = clean * 255.0
            batch_psnr = psnr(output_255, clean_255)
            total_psnr += batch_psnr.item()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    
    return avg_loss, avg_psnr


def save_checkpoint(model, optimizer, epoch, loss, psnr_val, output_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr_val
    }
    
    filename = f"weights.{epoch:03d}-{loss:.3f}-{psnr_val:.5f}.pth"
    filepath = output_path / filename
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filename}")


def main():
    args = get_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=output_path / "logs")
    
    # Get noise models
    source_noise_model = get_noise_model(args.source_noise_model)
    target_noise_model = get_noise_model(args.target_noise_model)
    val_noise_model = get_noise_model(args.val_noise_model)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        args.image_dir, args.test_dir,
        source_noise_model, target_noise_model, val_noise_model,
        batch_size=args.batch_size, image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(args.model)
    if "duc" in args.source_noise_model:
        tokens = args.source_noise_model.split(",")
        if len(tokens) > 2 and tokens[2] == "w":
            weights = np.load("../duc/output/demo/resnet50_layer_weight.npy")
            model = get_model(args.model, duc_weights=weights)
        else:
            model = get_model(args.model)
    else:
        model = get_model(args.model)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Load weights if specified
    start_epoch = 0
    if args.weight is not None:
        print(f"Loading weights from {args.weight}")
        checkpoint = torch.load(args.weight, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            model.load_state_dict(checkpoint)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup loss function
    if args.loss.lower() == "mse":
        criterion = nn.MSELoss()
    elif args.loss.lower() == "mae":
        criterion = nn.L1Loss()
    elif args.loss.lower() == "l0":
        criterion = L0Loss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    # Setup learning rate scheduler
    lr_scheduler = LearningRateScheduler(optimizer, args.nb_epochs, args.lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'lr': []
    }
    
    print("Starting training...")
    best_psnr = 0.0
    
    for epoch in range(start_epoch, args.nb_epochs):
        # Update learning rate
        current_lr = lr_scheduler.step(epoch)
        
        # Update L0 loss gamma if using L0 loss
        if args.loss.lower() == "l0":
            new_gamma = 2.0 * (args.nb_epochs - epoch) / args.nb_epochs
            criterion.update_gamma(new_gamma)
            print(f'Epoch {epoch + 1}: UpdateAnnealingParameter reducing gamma to {new_gamma}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.nb_epochs}')
        
        for batch_idx, (source, target) in enumerate(pbar):
            if batch_idx >= args.steps:
                break
                
            source = source.to(device)
            target = target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(source)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        print("Validating...")
        val_loss, val_psnr = validate_model(model, val_loader, criterion, device)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('PSNR/Validation', val_psnr, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['lr'].append(current_lr)
        
        # Print epoch results
        print(f'Epoch {epoch + 1}/{args.nb_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  Val PSNR: {val_psnr:.6f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, val_loss, val_psnr, output_path)
        
        # Save model every N epochs
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, val_psnr, output_path)
        
        print('-' * 50)
    
    # Save final model
    save_checkpoint(model, optimizer, args.nb_epochs - 1, val_loss, val_psnr, output_path)
    
    # Save training history
    np.savez(output_path / "history.npz", **history)
    
    # Close tensorboard writer
    writer.close()
    
    print("Training completed!")
    print(f"Best validation PSNR: {best_psnr:.6f}")


if __name__ == '__main__':
    main()
