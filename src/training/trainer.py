"""
Training utilities and training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from src.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    DEVICE,
    MODELS_DIR,
    SAVE_BEST_MODEL,
    EARLY_STOPPING_PATIENCE
)


class Trainer:
    """Training class for jaundice detection model."""
    
    def __init__(self, model, train_loader, val_loader, device=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on (default: from config)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else DEVICE
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = MODELS_DIR / "checkpoint_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = MODELS_DIR / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_acc: {self.best_val_acc:.2f}%) to {best_path}")
    
    def train(self, num_epochs=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train (default: from config)
        """
        if num_epochs is None:
            num_epochs = NUM_EPOCHS
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of epochs: {num_epochs}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if SAVE_BEST_MODEL:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
            
            # Early stopping
            if EARLY_STOPPING_PATIENCE > 0 and self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break
        
        print(f"\n✓ Training complete! Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
