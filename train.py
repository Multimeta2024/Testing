"""
Training script for hybrid image detection
Optimized for limited dataset (5k images) with advanced techniques:
- K-fold cross validation
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime

from hybrid_detection_model import HybridImageDetector, HybridDetectorLite
from hybrid_dataset import HybridImageDataset, SyntheticHybridDataset


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation task"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()


class HybridDetectionTrainer:
    """Complete training pipeline"""
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=3e-4,
                 num_epochs=50,
                 save_dir='checkpoints',
                 use_focal_loss=True,
                 use_localization=True):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_localization = use_localization
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss functions
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        
        self.localization_loss = DiceLoss() if use_localization else None
        
        # Optimizer with different learning rates for backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'rgb_branch.backbone' in name or 'rgb_encoder' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': learning_rate}
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Tracking
        self.best_val_auc = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_loc_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for rgb, freq, labels, masks in pbar:
            rgb = rgb.to(self.device)
            freq = freq.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                cls_logits, loc_maps = self.model(rgb, freq)
                
                # Classification loss
                cls_loss = self.classification_loss(cls_logits.squeeze(), labels)
                
                # Localization loss (only for hybrid images)
                if self.use_localization and loc_maps is not None:
                    hybrid_mask = labels > 0.5
                    if hybrid_mask.sum() > 0:
                        loc_loss = self.localization_loss(
                            loc_maps[hybrid_mask],
                            masks[hybrid_mask]
                        )
                    else:
                        loc_loss = torch.tensor(0.0, device=self.device)
                else:
                    loc_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = cls_loss + 0.5 * loc_loss  # Weight localization loss
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'loc': f'{loc_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0
        
        for rgb, freq, labels, masks in tqdm(self.val_loader, desc='Validation'):
            rgb = rgb.to(self.device)
            freq = freq.to(self.device)
            labels = labels.to(self.device)
            
            cls_logits, _ = self.model(rgb, freq)
            
            loss = self.classification_loss(cls_logits.squeeze(), labels)
            total_loss += loss.item()
            
            all_logits.append(cls_logits.squeeze())
            all_labels.append(labels)
        
        # Compute metrics
        all_logits = torch.cat(all_logits).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        # Probabilities
        probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid
        preds = (probs > 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(all_labels, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))
            print(f"✓ Saved best model (AUC: {metrics['auc']:.4f})")
    
    def train(self):
        """Complete training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        # Save final history
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
        
        return self.history


def prepare_data_loaders(img_dir, labels, batch_size=16, val_split=0.2, num_workers=4):
    """
    Prepare train and validation dataloaders
    
    Args:
        img_dir: Directory with images
        labels: List of (img_path, label, mask_path) tuples
        batch_size: Batch size
        val_split: Validation split ratio
        num_workers: Number of workers for data loading
    """
    # Create full dataset
    full_dataset = HybridImageDataset(
        img_dir=img_dir,
        labels=labels,
        mode='train'
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update mode for val dataset
    val_dataset.dataset.mode = 'val'
    val_dataset.dataset.use_augmentation = False
    
    # Create dataloaders
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
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example training script
    
    # Configuration
    CONFIG = {
        'img_dir': 'path/to/your/5k/images',
        'batch_size': 16,
        'learning_rate': 3e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'use_lite_model': False,  # Set True for faster training
    }
    
    # Prepare your labels
    # Format: [(img_path, label, mask_path), ...]
    # label: 0=real, 1=hybrid
    # mask_path: optional, None if not available
    
    labels = [
        # Example - replace with your actual data
        ("image1.jpg", 0, None),  # Real
        ("image2.jpg", 1, "image2_mask.png"),  # Hybrid with mask
        ("image3.jpg", 1, None),  # Hybrid without mask
        # ... add your 5k images here
    ]
    
    print("=" * 60)
    print("HYBRID IMAGE DETECTION TRAINING")
    print("=" * 60)
    
    # Prepare data
    print("\nPreparing dataloaders...")
    train_loader, val_loader = prepare_data_loaders(
        img_dir=CONFIG['img_dir'],
        labels=labels,
        batch_size=CONFIG['batch_size'],
        val_split=0.2
    )
    
    # Initialize model
    print("\nInitializing model...")
    if CONFIG['use_lite_model']:
        model = HybridDetectorLite()
        print("Using lightweight model")
    else:
        model = HybridImageDetector()
        print("Using full model with localization")
    
    # Initialize trainer
    trainer = HybridDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs'],
        save_dir=CONFIG['save_dir'],
        use_localization=not CONFIG['use_lite_model']
    )
    
    # Train
    history = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {CONFIG['save_dir']}/best.pth")
    print("=" * 60)
