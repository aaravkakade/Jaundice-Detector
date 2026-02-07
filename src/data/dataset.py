"""
PyTorch Dataset classes for loading images.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class JaundiceDataset(Dataset):
    """Dataset class for jaundice detection images."""
    
    def __init__(self, data_dir, transform=None, split="train"):
        """
        Args:
            data_dir: Path to the data directory (processed/train, processed/val, etc.)
            transform: Optional transform to be applied on a sample
            split: One of 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load images and labels
        self.samples = []
        self.class_to_idx = {"normal": 0, "jaundice": 1}
        self.idx_to_class = {0: "normal", 1: "jaundice"}
        
        # Find all class directories
        for class_name in self.class_to_idx.keys():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                # Get all image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_dir}. Make sure images are placed in class subdirectories.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split="train", input_size=224):
    """
    Get data transforms for training, validation, or testing.
    
    Args:
        split: One of 'train', 'val', or 'test'
        input_size: Target image size
    
    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        # Training: augmentation + normalization
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test: only normalization
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
