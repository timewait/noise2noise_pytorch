import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Plot training history",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--history_file", type=str, required=True,
                        help="path to history.npz file")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="directory to save plots")
    parser.add_argument("--show", action="store_true",
                        help="show plots instead of saving")
    args = parser.parse_args()
    return args


def plot_training_history(history_file, output_dir=None, show=False):
    """
    Plot training history from saved numpy file
    
    Args:
        history_file: Path to history.npz file
        output_dir: Directory to save plots (if not showing)
        show: Whether to show plots instead of saving
    """
    
    # Load history
    history_data = np.load(history_file)
    
    # Extract data
    train_loss = history_data.get('train_loss', [])
    val_loss = history_data.get('val_loss', [])
    val_psnr = history_data.get('val_psnr', [])
    lr = history_data.get('lr', [])
    
    epochs = range(1, len(train_loss) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation PSNR
    axes[0, 1].plot(epochs, val_psnr, 'g-', label='Validation PSNR', linewidth=2)
    axes[0, 1].set_title('Validation PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    axes[1, 0].plot(epochs, lr, 'm-', label='Learning Rate', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss Comparison (log scale)
    axes[1, 1].semilogy(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[1, 1].semilogy(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1, 1].set_title('Loss Comparison (Log Scale)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save combined plot
            plt.savefig(output_path / 'training_history.png', dpi=300, bbox_inches='tight')
            print(f"Saved combined plot: {output_path / 'training_history.png'}")
            
            # Save individual plots
            fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
            ax_loss.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
            ax_loss.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
            ax_loss.set_title('Training and Validation Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            fig_loss.savefig(output_path / 'loss_history.png', dpi=300, bbox_inches='tight')
            plt.close(fig_loss)
            
            fig_psnr, ax_psnr = plt.subplots(figsize=(10, 6))
            ax_psnr.plot(epochs, val_psnr, 'g-', label='Validation PSNR', linewidth=2)
            ax_psnr.set_title('Validation PSNR')
            ax_psnr.set_xlabel('Epoch')
            ax_psnr.set_ylabel('PSNR (dB)')
            ax_psnr.legend()
            ax_psnr.grid(True, alpha=0.3)
            fig_psnr.savefig(output_path / 'psnr_history.png', dpi=300, bbox_inches='tight')
            plt.close(fig_psnr)
            
            print(f"Saved individual plots to: {output_path}")
        
        plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total epochs: {len(epochs)}")
    print(f"Final training loss: {train_loss[-1]:.6f}")
    print(f"Final validation loss: {val_loss[-1]:.6f}")
    print(f"Final validation PSNR: {val_psnr[-1]:.6f} dB")
    print(f"Best validation PSNR: {max(val_psnr):.6f} dB (epoch {np.argmax(val_psnr) + 1})")
    print(f"Final learning rate: {lr[-1]:.8f}")
    
    # Calculate improvement
    if len(val_psnr) > 1:
        psnr_improvement = val_psnr[-1] - val_psnr[0]
        print(f"PSNR improvement: {psnr_improvement:+.6f} dB")
    
    print("="*50)


def main():
    args = get_args()
    
    history_file = Path(args.history_file)
    if not history_file.exists():
        print(f"History file not found: {history_file}")
        return
    
    plot_training_history(
        history_file=history_file,
        output_dir=args.output_dir if not args.show else None,
        show=args.show
    )


if __name__ == '__main__':
    main()
