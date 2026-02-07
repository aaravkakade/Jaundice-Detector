"""
Main training script.
"""

import torch
from src.config import DEVICE
from src.data.loader import get_dataloaders
from src.training.model import get_model
from src.training.trainer import Trainer


def main():
    """Main training function."""
    print("=" * 60)
    print("Jaundice Detection Model Training")
    print("=" * 60)
    
    # Check device
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Get data loaders
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nMake sure you have:")
        print("1. Placed images in data/raw/ organized by class (jaundice/ and normal/)")
        print("2. Run 'python scripts/split_data.py' to create train/val/test splits")
        return
    
    # Get model
    print("\nInitializing model...")
    model = get_model(pretrained=True)
    print(f"✓ Model initialized: {model.__class__.__name__}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device=device)
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
