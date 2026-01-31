"""
Complete Training Script for Folder-Based Dataset
Works directly with real/ and hybrid/ folders
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json
from datetime import datetime

from hybrid_detection_model import HybridImageDetector, HybridDetectorLite
from hybrid_dataset import HybridImageDataset
from prepare_data import prepare_dataset


def train_from_folders(
    dataset_root: str,
    real_folder: str = 'real',
    hybrid_folder: str = 'hybrid',
    mask_folder: str = 'mask',
    # Model settings
    use_lite_model: bool = False,
    use_localization: bool = True,
    # Training settings
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 3e-4,
    val_split: float = 0.2,
    # Hardware
    device: str = 'cuda',
    num_workers: int = 4,
    # Saving
    save_dir: str = '/kaggle/working/checkpoints',
    experiment_name: str = 'hybrid_detection'
):
    
    print("\n" + "=" * 80)
    print("HYBRID IMAGE DETECTION - TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Prepare dataset
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING DATASET")
    print("=" * 80)
    
    labels = prepare_dataset(
        dataset_root=dataset_root,
        real_folder=real_folder,
        hybrid_folder=hybrid_folder,
        mask_folder=mask_folder,
        verify_masks=True
    )
    
    if len(labels) == 0:
        raise ValueError("No images found! Check your folder paths.")
    
    # Step 2: Split data
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING DATA")
    print("=" * 80)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(labels))
    
    # Split
    val_size = int(len(labels) * val_split)
    train_size = len(labels) - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    
    print(f"\n✅ Data split complete:")
    print(f"   ├─ Training samples: {len(train_labels)}")
    print(f"   └─ Validation samples: {len(val_labels)}")
    
    # Check class balance
    train_real = sum(1 for l in train_labels if l[1] == 0)
    train_hybrid = sum(1 for l in train_labels if l[1] == 1)
    print(f"\n   Training set:")
    print(f"   ├─ Real: {train_real} ({train_real/len(train_labels)*100:.1f}%)")
    print(f"   └─ Hybrid: {train_hybrid} ({train_hybrid/len(train_labels)*100:.1f}%)")
    
    # Step 3: Create datasets and loaders
    print("\n" + "=" * 80)
    print("STEP 3: CREATING DATALOADERS")
    print("=" * 80)
    
    train_dataset = HybridImageDataset(
        img_dir=dataset_root,
        labels=train_labels,
        mode='train',
        use_augmentation=True
    )
    
    val_dataset = HybridImageDataset(
        img_dir=dataset_root,
        labels=val_labels,
        mode='val',
        use_augmentation=False
    )
    
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
    
    print(f"✅ Dataloaders created:")
    print(f"   ├─ Batch size: {batch_size}")
    print(f"   ├─ Training batches: {len(train_loader)}")
    print(f"   └─ Validation batches: {len(val_loader)}")
    
    # Step 4: Initialize model
    print("\n" + "=" * 80)
    print("STEP 4: INITIALIZING MODEL")
    print("=" * 80)
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    if use_lite_model:
        model = HybridDetectorLite().to(device)
        print("✅ Using HybridDetectorLite")
    else:
        model = HybridImageDetector().to(device)
        print("✅ Using HybridImageDetector (Full)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ├─ Total parameters: {total_params:,}")
    print(f"   └─ Trainable parameters: {trainable_params:,}")
    
    # Step 5: Setup training
    print("\n" + "=" * 80)
    print("STEP 5: SETTING UP TRAINING")
    print("=" * 80)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    print(f"✅ Optimizer: AdamW (lr={learning_rate}, wd=0.01)")
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    print(f"✅ Scheduler: CosineAnnealingWarmRestarts")
    
    # Loss functions
    from train import DiceLoss
    cls_loss_fn = nn.BCEWithLogitsLoss()
    loc_loss_fn = DiceLoss() if use_localization and not use_lite_model else None
    print(f"✅ Loss: BCEWithLogitsLoss")
    if loc_loss_fn:
        print(f"✅ Localization Loss: Dice Loss")
    
    # Mixed precision
    scaler = GradScaler()
    print(f"✅ Mixed Precision: Enabled")
    
    # Create save directory
    save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ Save directory: {save_dir}")
    
    # Step 6: Training loop
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING")
    print("=" * 80)
    print(f"\nStarting training for {num_epochs} epochs...")
    
    best_val_auc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for rgb, freq, labels, masks in pbar:
            rgb = rgb.to(device)
            freq = freq.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                cls_logits, loc_maps = model(rgb, freq)
                
                # Classification loss
                cls_loss = cls_loss_fn(cls_logits.squeeze(dim=1), labels)
                
                # Localization loss
                if loc_loss_fn and loc_maps is not None:
                    hybrid_mask = labels > 0.5
                    if hybrid_mask.sum() > 0:
                        loc_loss = loc_loss_fn(loc_maps[hybrid_mask], masks[hybrid_mask])
                    else:
                        loc_loss = torch.tensor(0.0, device=device)
                    total_loss = cls_loss + 0.5 * loc_loss
                else:
                    total_loss = cls_loss
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for rgb, freq, labels, masks in tqdm(val_loader, desc='Validation'):
                rgb = rgb.to(device)
                freq = freq.to(device)
                labels_tensor = labels.to(device)
                
                cls_logits, _ = model(rgb, freq)
                loss = cls_loss_fn(cls_logits.squeeze(dim=1), labels_tensor)
                val_loss += loss.item()
                
                probs = torch.sigmoid(cls_logits.squeeze(dim=1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)
        
        val_auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(f1)
        
        # Print metrics
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   ├─ Train Loss: {avg_train_loss:.4f}")
        print(f"   ├─ Val Loss: {avg_val_loss:.4f}")
        print(f"   ├─ Val AUC: {val_auc:.4f}")
        print(f"   ├─ Val F1: {f1:.4f}")
        print(f"   ├─ Val Precision: {precision:.4f}")
        print(f"   └─ Val Recall: {recall:.4f}")
        
        # Save checkpoint
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            print(f"\n   ✅ New best model! (AUC: {val_auc:.4f})")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_f1': f1
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best.pth'))
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, 'latest.pth'))
        
        # Update scheduler
        scheduler.step()
    
    # Save history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Best Validation AUC: {best_val_auc:.4f}")
    print(f"✅ Model saved to: {save_dir}/best.pth")
    print(f"✅ Training history saved to: {save_dir}/history.json")
    print("\n" + "=" * 80)
    
    return history, save_dir


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Ready-to-use training script for your folder structure
    
    Just set your dataset path and run!
    """
    
    # ========================================================================
    # CONFIGURATION - CHANGE THESE TO MATCH YOUR SETUP
    # ========================================================================
    
    CONFIG = {
        # Dataset paths
        'dataset_root': '/kaggle/input/hybrid-dataset/hybrid-dataset',  # ← CHANGE THIS!
        'real_folder': 'real',                    # Folder with real images
        'hybrid_folder': 'hybrid',                # Folder with hybrid images (or 'fake')
        'mask_folder': 'mask',                   # Folder with PNG masks
        
        # Model settings
        'use_lite_model': False,                  # True = faster, False = better performance
        'use_localization': True,                 # Use masks for localization
        
        # Training settings
        'batch_size': 16,                         # Reduce if out of memory
        'num_epochs': 50,                         # Number of training epochs
        'learning_rate': 3e-4,                    # Learning rate
        'val_split': 0.2,                         # 20% for validation
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,                         # Data loading workers
        
        # Saving
        'save_dir': '/kaggle/working/checkpoints',
        'experiment_name': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    
    # ========================================================================
    # RUN TRAINING
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    # Start training
    history, save_dir = train_from_folders(**CONFIG)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Evaluate your model:")
    print(f"   python evaluate.py --model_path {save_dir}/best.pth")
    print("\n2. Run inference on test images:")
    print(f"   python inference.py --model_path {save_dir}/best.pth")
    print("\n3. Check training curves:")
    print(f"   Check {save_dir}/history.json")
    print("\n" + "=" * 80)