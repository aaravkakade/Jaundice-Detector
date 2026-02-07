"""
Model definitions for jaundice detection.
"""

import torch
import torch.nn as nn
from torchvision import models
from src.config import MODEL_NAME, NUM_CLASSES


def get_model(model_name=None, num_classes=None, pretrained=True):
    """
    Get a PyTorch model for jaundice detection.
    
    Args:
        model_name: Name of the model architecture (default: from config)
        num_classes: Number of output classes (default: from config)
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
    if model_name is None:
        model_name = MODEL_NAME
    if num_classes is None:
        num_classes = NUM_CLASSES
    
    # Load base model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: resnet18, resnet34, resnet50")
    
    return model
