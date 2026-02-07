"""
DataLoader utilities for creating PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader
from src.data.dataset import JaundiceDataset, get_transforms
from src.config import BATCH_SIZE, NUM_WORKERS, INPUT_SIZE, DATA_PROCESSED_DIR


def get_dataloaders(data_dir=None, batch_size=None, num_workers=None, input_size=None):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Base directory for processed data (default: from config)
        batch_size: Batch size (default: from config)
        num_workers: Number of workers for DataLoader (default: from config)
        input_size: Image input size (default: from config)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if data_dir is None:
        data_dir = DATA_PROCESSED_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if input_size is None:
        input_size = INPUT_SIZE
    
    # Create datasets
    train_dataset = JaundiceDataset(
        data_dir / "train",
        transform=get_transforms("train", input_size),
        split="train"
    )
    
    val_dataset = JaundiceDataset(
        data_dir / "val",
        transform=get_transforms("val", input_size),
        split="val"
    )
    
    test_dataset = JaundiceDataset(
        data_dir / "test",
        transform=get_transforms("test", input_size),
        split="test"
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
