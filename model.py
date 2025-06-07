import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class L0Loss(nn.Module):
    def __init__(self, gamma=2.0):
        super(L0Loss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        loss = torch.pow(torch.abs(y_true - y_pred) + 1e-8, self.gamma)
        return torch.mean(loss)

    def update_gamma(self, new_gamma):
        self.gamma = new_gamma


def psnr(y_pred, y_true, max_pixel=255.0):
    """Calculate PSNR between predicted and true images"""
    y_pred = torch.clamp(y_pred, 0.0, max_pixel)
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class SRResNet(nn.Module):
    def __init__(self, input_channels=3, feature_dim=64, num_residual_blocks=16, duc_weights=None):
        super(SRResNet, self).__init__()
        
        # Initial convolution
        if duc_weights is not None:
            self.conv_input = nn.Conv2d(input_channels, feature_dim, kernel_size=(7,7), padding=(3,3), bias=False)
            self.conv_input.weight.data = torch.tensor(duc_weights, dtype=torch.float32)
            self.conv_input.weight.requires_grad = False
        else:
            self.conv_input = nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1)
        self.prelu_input = nn.PReLU()
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(feature_dim) for _ in range(num_residual_blocks)
        ])
        
        # Final convolution after residual blocks
        self.conv_mid = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(feature_dim)
        
        # Output convolution
        self.conv_output = nn.Conv2d(feature_dim, input_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        out = self.conv_input(x)
        out = self.prelu_input(out)
        initial = out
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Final convolution after residual blocks
        out = self.conv_mid(out)
        out = self.bn_mid(out)
        out += initial
        
        # Output convolution
        out = self.conv_output(out)
        
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, batch_norm=False, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
        
        if residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            residual = self.residual_conv(x)
            out = torch.cat([residual, out], dim=1)
        return out


class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, start_channels=64, depth=4, 
                 inc_rate=2.0, dropout=0.5, batch_norm=False, max_pool=True, 
                 up_conv=True, residual=False):
        super(UNet, self).__init__()
        
        self.depth = depth
        self.max_pool = max_pool
        self.up_conv = up_conv
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_ch = input_channels
        for i in range(depth + 1):
            out_ch = int(start_channels * (inc_rate ** i))
            do = dropout if i == depth else 0  # Only apply dropout at the bottom
            self.encoder_blocks.append(DoubleConv(in_ch, out_ch, do, batch_norm, residual))
            
            if i < depth:
                if max_pool:
                    self.pool_layers.append(nn.MaxPool2d(2))
                else:
                    self.pool_layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            
            in_ch = out_ch * 2 if residual else out_ch
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for i in range(depth):
            in_ch = int(start_channels * (inc_rate ** (depth - i)))
            out_ch = int(start_channels * (inc_rate ** (depth - i - 1)))
            
            if up_conv:
                self.upconv_layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size=2, padding=0)
                ))
            else:
                self.upconv_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            
            # Concatenated channels: out_ch (from upconv) + out_ch (from skip connection)
            self.decoder_blocks.append(DoubleConv(out_ch * 2, out_ch, 0, batch_norm, residual))
        
        # Final output layer
        final_in_ch = start_channels * 2 if residual else start_channels
        self.final_conv = nn.Conv2d(final_in_ch, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        current = x
        
        for i in range(self.depth + 1):
            current = self.encoder_blocks[i](current)
            if i < self.depth:
                encoder_outputs.append(current)
                current = self.pool_layers[i](current)
        
        # Decoder
        for i in range(self.depth):
            current = self.upconv_layers[i](current)
            skip = encoder_outputs[self.depth - 1 - i]
            
            # Handle size mismatch
            if current.size() != skip.size():
                current = F.interpolate(current, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            current = torch.cat([skip, current], dim=1)
            current = self.decoder_blocks[i](current)
        
        # Final output
        output = self.final_conv(current)
        return output


def get_model(model_name="srresnet", **kwargs):
    """Factory function to create models"""
    if model_name.lower() == "srresnet":
        return SRResNet(**kwargs)
    elif model_name.lower() == "unet":
        return UNet(**kwargs)
    else:
        raise ValueError(f"model_name should be 'srresnet' or 'unet', got {model_name}")


def main():
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test SRResNet
    print("Testing SRResNet...")
    model = get_model("srresnet")
    model.to(device)
    
    # Create dummy input
    x = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"SRResNet input shape: {x.shape}, output shape: {output.shape}")
    
    # Test UNet
    print("\nTesting UNet...")
    model = get_model("unet")
    model.to(device)
    
    with torch.no_grad():
        output = model(x)
    print(f"UNet input shape: {x.shape}, output shape: {output.shape}")
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    srresnet = get_model("srresnet")
    unet = get_model("unet")
    
    print(f"\nSRResNet parameters: {count_parameters(srresnet):,}")
    print(f"UNet parameters: {count_parameters(unet):,}")


if __name__ == '__main__':
    main()
