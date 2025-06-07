import argparse
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from model import get_model
from noise_model import get_noise_model


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    parser.add_argument("--device", type=str, default="auto",
                        help="device to use ('cuda', 'cpu', or 'auto')")
    args = parser.parse_args()
    return args


def tensor_to_image(tensor):
    """Convert tensor to numpy image array"""
    # Clamp values to [0, 1] and convert to [0, 255]
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
    image = (image * 255).astype(np.uint8)
    return image


def image_to_tensor(image, device):
    """Convert numpy image to tensor"""
    # Convert BGR to RGB and normalize to [0, 1]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def main():
    args = get_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load noise model
    val_noise_model = get_noise_model(args.test_noise_model)
    
    # Create and load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model)
    
    # Load weights
    print(f"Loading weights from {args.weight_file}")
    checkpoint = torch.load(args.weight_file, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'psnr' in checkpoint:
            print(f"Checkpoint PSNR: {checkpoint['psnr']:.6f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Get image paths
    image_dir = Path(args.image_dir)
    image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
    image_paths = [p for p in image_dir.glob("**/*") if p.suffix.lower() in image_suffixes]
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load {image_path}")
            continue
        
        h, w, _ = image.shape
        print(f"Original size: {w}x{h}")
        
        # Ensure dimensions are divisible by 16 for stride compatibility
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        image = image[:new_h, :new_w]
        
        if new_h != h or new_w != w:
            print(f"Cropped to: {new_w}x{new_h}")
        
        # Add noise to create test input
        noise_image = val_noise_model(image.copy())
        
        # Convert to tensor
        noisy_tensor = image_to_tensor(noise_image, device)
        
        # Denoise with model
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)
        
        # Convert back to images
        clean_img = image
        noisy_img = noise_image
        denoised_img = tensor_to_image(denoised_tensor.squeeze(0))
        
        # Convert RGB back to BGR for OpenCV
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)
        
        # Create comparison image (clean | noisy | denoised)
        h, w = clean_img.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = clean_img
        comparison[:, w:w*2] = noisy_img
        comparison[:, w*2:] = denoised_img
        
        if args.output_dir:
            # Save result
            output_filename = f"{image_path.stem}_result.png"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), comparison)
            print(f"Saved: {output_filename}")
            
            # Also save individual images
            cv2.imwrite(str(output_dir / f"{image_path.stem}_clean.png"), clean_img)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_noisy.png"), noisy_img)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_denoised.png"), denoised_img)
        else:
            # Show result
            # Resize for display if too large
            display_height = 600
            if h > display_height:
                scale = display_height / h
                new_width = int(w * 3 * scale)
                comparison = cv2.resize(comparison, (new_width, display_height))
            
            cv2.imshow("Result (Clean | Noisy | Denoised)", comparison)
            print("Press any key to continue, 'q' to quit...")
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
    
    if not args.output_dir:
        cv2.destroyAllWindows()
    
    print("Testing completed!")


if __name__ == '__main__':
    main()
