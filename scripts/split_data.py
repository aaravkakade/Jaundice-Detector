"""
Script to split manually placed images into train/val/test sets.
"""

import os
import shutil
from pathlib import Path
import random
from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)


def split_data():
    """Split images from raw/ into train/val/test sets in processed/."""
    
    raw_dir = Path(DATA_RAW_DIR)
    processed_dir = Path(DATA_PROCESSED_DIR)
    
    # Check if raw directory exists and has class folders
    if not raw_dir.exists():
        raise ValueError(f"Raw data directory not found: {raw_dir}")
    
    # Find class directories
    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {raw_dir}. Expected folders like 'jaundice/' and 'normal/'")
    
    print(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    
    # Create processed directory structure
    for split in ['train', 'val', 'test']:
        for class_dir in class_dirs:
            (processed_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            print(f"  Warning: No images found in {class_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Shuffle images
        random.seed(42)  # For reproducibility
        random.shuffle(image_files)
        
        # Calculate split sizes
        n_total = len(image_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # n_test = n_total - n_train - n_val (remaining)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"  Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Copy files to respective directories
        for file in train_files:
            shutil.copy2(file, processed_dir / "train" / class_name / file.name)
        
        for file in val_files:
            shutil.copy2(file, processed_dir / "val" / class_name / file.name)
        
        for file in test_files:
            shutil.copy2(file, processed_dir / "test" / class_name / file.name)
    
    print(f"\n✓ Data splitting complete! Processed data saved to: {processed_dir}")


if __name__ == "__main__":
    split_data()
