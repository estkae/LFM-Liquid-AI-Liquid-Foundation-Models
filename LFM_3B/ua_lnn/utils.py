"""
Training utilities for UA-LNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Callable, Tuple, List
import numpy as np
from tqdm import tqdm
import wandb
from .config import UALNNConfig
from .model import UpdateAdaptiveLNN
from .medical_adapter import MedicalUALNN


class UALNNTrainer:
    """
    Trainer class for UA-LNN models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: UALNNConfig,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_wandb: bool = False):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_reg
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss() if config.output_dim > 1 else nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'uncertainty': [],
            'adaptation_rate': []
        }
        
        if use_wandb:
            wandb.init(project="ua-lnn", config=config.to_dict())
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        uncertainty_scores = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            results = self.model(data, return_uncertainty=True)
            
            output = results['output']
            uncertainty = results.get('uncertainty', None)
            
            # Compute loss
            loss = self.criterion(output, targets)
            
            # Add uncertainty regularization
            if uncertainty is not None:
                uncertainty_penalty = (1 - uncertainty).mean() * 0.1
                loss = loss + uncertainty_penalty
                uncertainty_scores.append(uncertainty.mean().item())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if self.config.output_dim > 1:  # Classification
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total if total > 0 else 0
            })
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total if total > 0 else 0,
            'avg_uncertainty': np.mean(uncertainty_scores) if uncertainty_scores else 0
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        uncertainty_scores = []
        confidence_scores = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get predictions with uncertainty
                results = self.model.predict_with_uncertainty(data, n_samples=20)
                
                output = results['prediction']
                uncertainty = results.get('epistemic_uncertainty', None)
                confidence = results['confidence']
                
                # Compute loss
                loss = self.criterion(output, targets)
                total_loss += loss.item()
                
                if self.config.output_dim > 1:  # Classification
                    _, predicted = output.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                if uncertainty is not None:
                    uncertainty_scores.append(uncertainty.mean().item())
                confidence_scores.append(confidence.mean().item())
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total if total > 0 else 0,
            'avg_uncertainty': np.mean(uncertainty_scores) if uncertainty_scores else 0,
            'avg_confidence': np.mean(confidence_scores)
        }
        
        return metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: Optional[int] = None,
              save_path: Optional[str] = None,
              early_stopping_patience: int = 20):
        """
        Full training loop
        """
        
        if epochs is None:
            epochs = self.config.epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Uncertainty: {train_metrics['avg_uncertainty']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Confidence: {val_metrics['avg_confidence']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['uncertainty'].append(val_metrics['avg_uncertainty'])
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_acc': val_metrics['accuracy'],
                    'uncertainty': val_metrics['avg_uncertainty'],
                    'confidence': val_metrics['avg_confidence'],
                    'epoch': epoch
                })
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['metrics']


class OnlineAdaptationTrainer:
    """
    Trainer for online adaptation of UA-LNN
    """
    
    def __init__(self, model: UpdateAdaptiveLNN, config: UALNNConfig):
        self.model = model
        self.config = config
        self.adaptation_history = []
    
    def adapt_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Single online adaptation step"""
        
        # Get current prediction
        with torch.no_grad():
            results_before = self.model(x)
            pred_before = results_before['output']
            
        # Perform adaptation
        self.model.adapt_online(x, y)
        
        # Get updated prediction
        with torch.no_grad():
            results_after = self.model(x)
            pred_after = results_after['output']
        
        # Compute adaptation metrics
        if pred_before.size(-1) > 1:  # Classification
            loss_before = F.cross_entropy(pred_before, y).item()
            loss_after = F.cross_entropy(pred_after, y).item()
        else:  # Regression
            loss_before = F.mse_loss(pred_before.squeeze(), y).item()
            loss_after = F.mse_loss(pred_after.squeeze(), y).item()
        
        adaptation_info = {
            'loss_before': loss_before,
            'loss_after': loss_after,
            'improvement': loss_before - loss_after,
            'adaptation_rate': results_after['diagnostics']['adaptation_active']
        }
        
        self.adaptation_history.append(adaptation_info)
        
        return adaptation_info
    
    def evaluate_adaptation(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model adaptation performance"""
        
        initial_performance = []
        adapted_performance = []
        
        for x, y in test_loader:
            # Initial prediction
            with torch.no_grad():
                results = self.model(x)
                initial_pred = results['output']
            
            # Adapt
            self.model.adapt_online(x, y)
            
            # Adapted prediction
            with torch.no_grad():
                results = self.model(x)
                adapted_pred = results['output']
            
            # Compute metrics
            if initial_pred.size(-1) > 1:
                initial_acc = (initial_pred.argmax(dim=-1) == y).float().mean()
                adapted_acc = (adapted_pred.argmax(dim=-1) == y).float().mean()
                initial_performance.append(initial_acc.item())
                adapted_performance.append(adapted_acc.item())
        
        return {
            'initial_accuracy': np.mean(initial_performance),
            'adapted_accuracy': np.mean(adapted_performance),
            'improvement': np.mean(adapted_performance) - np.mean(initial_performance)
        }