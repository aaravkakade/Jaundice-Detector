"""
Model inference and prediction utilities.
"""

import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from src.config import MODELS_DIR, MODEL_NAME, NUM_CLASSES, INPUT_SIZE, DEVICE, CLASS_NAMES
from src.training.model import get_model


class JaundicePredictor:
    """Predictor class for jaundice detection."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint (default: models/best_model.pth)
            device: Device to run inference on (default: from config)
        """
        self.device = device if device else DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        # Default model path
        if model_path is None:
            model_path = MODELS_DIR / "best_model.pth"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        # Load model
        self.model = get_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (same as validation/test)
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = CLASS_NAMES
    
    def predict(self, image):
        """
        Predict jaundice from an image.
        
        Args:
            image: PIL Image or path to image file
        
        Returns:
            dict with 'class', 'confidence', and 'probabilities'
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or path to image file")
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        predicted_class_idx = predicted.item()
        predicted_class = self.class_names[predicted_class_idx]
        confidence_score = confidence.item()
        
        # Get all probabilities
        probs = probabilities[0].cpu().numpy()
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        return {
            'class': predicted_class,
            'confidence': confidence_score,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, images):
        """
        Predict jaundice for multiple images.
        
        Args:
            images: List of PIL Images or paths to image files
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in images]
