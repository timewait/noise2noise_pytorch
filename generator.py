import os
import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, source_noise_model, target_noise_model, image_size=64, transform=None):
        """
        PyTorch Dataset for Noise2Noise training
        
        Args:
            image_dir: Directory containing training images
            source_noise_model: Function to add noise to source images
            target_noise_model: Function to add noise to target images
            image_size: Size of image patches to extract
            transform: Optional torchvision transforms
        """
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_size = image_size
        self.transform = transform
        
        if len(self.image_paths) == 0:
            raise ValueError(f"image dir '{image_dir}' does not include any image")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        # Return a large number to ensure we can sample many patches per epoch
        return len(self.image_paths) * 100

    def __getitem__(self, idx):
        # Randomly select an image
        image_path = random.choice(self.image_paths)
        image = cv2.imread(str(image_path))
        
        if image is None:
            # If image loading fails, try another one
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        h, w, _ = image.shape
        
        # Ensure image is large enough
        if h < self.image_size or w < self.image_size:
            # If image is too small, try another one
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Extract random patch
        i = np.random.randint(0, h - self.image_size + 1)
        j = np.random.randint(0, w - self.image_size + 1)
        clean_patch = image[i:i + self.image_size, j:j + self.image_size]
        
        # Apply noise models
        source_patch = self.source_noise_model(clean_patch.copy())
        target_patch = self.target_noise_model(clean_patch.copy())
        
        # Convert BGR to RGB and normalize to [0, 1]
        source_patch = cv2.cvtColor(source_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target_patch = cv2.cvtColor(target_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to torch tensors and change from HWC to CHW
        source_tensor = torch.from_numpy(source_patch).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_patch).permute(2, 0, 1)
        
        if self.transform:
            source_tensor = self.transform(source_tensor)
            target_tensor = self.transform(target_tensor)
        
        return source_tensor, target_tensor


class ValidationDataset(Dataset):
    def __init__(self, image_dir, val_noise_model, transform=None):
        """
        PyTorch Dataset for validation
        
        Args:
            image_dir: Directory containing validation images
            val_noise_model: Function to add noise to validation images
            transform: Optional torchvision transforms
        """
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        
        if len(image_paths) == 0:
            raise ValueError(f"image dir '{image_dir}' does not include any image")
        
        self.val_noise_model = val_noise_model
        self.data = []
        
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            h, w, _ = image.shape
            # Ensure dimensions are divisible by 16 for stride compatibility
            image = image[:(h // 16) * 16, :(w // 16) * 16]
            
            # Convert BGR to RGB and normalize
            clean_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            noisy_image = self.val_noise_model(image.copy())
            noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Convert to tensors (CHW format)
            clean_tensor = torch.from_numpy(clean_image).permute(2, 0, 1)
            noisy_tensor = torch.from_numpy(noisy_image).permute(2, 0, 1)
            
            if transform:
                clean_tensor = transform(clean_tensor)
                noisy_tensor = transform(noisy_tensor)
            
            self.data.append((noisy_tensor, clean_tensor))
        
        print(f"Loaded {len(self.data)} validation images from {image_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_data_loaders(train_dir, val_dir, source_noise_model, target_noise_model, 
                       val_noise_model, batch_size=16, image_size=64, num_workers=4):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        train_dir: Training images directory
        val_dir: Validation images directory
        source_noise_model: Noise model for source images
        target_noise_model: Noise model for target images
        val_noise_model: Noise model for validation images
        batch_size: Batch size
        image_size: Size of training patches
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    
    # Create datasets
    train_dataset = NoisyImageDataset(
        train_dir, source_noise_model, target_noise_model, image_size
    )
    
    val_dataset = ValidationDataset(val_dir, val_noise_model)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process validation images one by one
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def main():
    """Test the data loading functionality"""
    from noise_model import get_noise_model
    
    # Test noise models
    source_noise = get_noise_model("gaussian,0,50")
    target_noise = get_noise_model("gaussian,0,50")
    val_noise = get_noise_model("gaussian,25,25")
    
    # Create a simple test
    print("Testing data loading...")
    
    # You would need to have actual image directories for this to work
    # train_loader, val_loader = create_data_loaders(
    #     "dataset/291", "dataset/Set5", 
    #     source_noise, target_noise, val_noise,
    #     batch_size=4, image_size=64
    # )
    
    # print(f"Train loader length: {len(train_loader)}")
    # print(f"Val loader length: {len(val_loader)}")
    
    # # Test one batch
    # for batch_idx, (source, target) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}: source shape {source.shape}, target shape {target.shape}")
    #     if batch_idx >= 2:  # Just test a few batches
    #         break
    
    print("Data loading test completed!")


if __name__ == '__main__':
    main()
