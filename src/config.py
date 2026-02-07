"""
Configuration file for paths and constants.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
NUM_WORKERS = 4  # For DataLoader

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model configuration
MODEL_NAME = "resnet18"  # Options: resnet18, resnet34, resnet50
NUM_CLASSES = 2  # jaundice and normal
INPUT_SIZE = 224  # Image input size

# Training settings
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
SAVE_BEST_MODEL = True
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs

# Class names
CLASS_NAMES = ["normal", "jaundice"]
