"""
Training script for hybrid image detection.
Updated to match rewritten dataset and model:
  - Dataset now returns: rgb, freq, edge_map, label, mask  (5 items)
  - Model now takes:     rgb, freq, edge_map               (3 inputs)
  - Optimizer uses lower LR for pretrained backbone, higher for new heads
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

from hybrid_detection_model import HybridDetectorLite, HybridImageDetector
from hybrid_dataset import HybridImageDataset
from prepare_data import prepare_dataset




def train_from_folders(
    dataset_root: str,
    real_folder: str = 'real',
    hybrid_folder: str = 'hybrid',
    mask_folder: str = None,
    # Model settings
    use_lite_model: bool = True,
    use_localization: bool = False,
    # Training settings
    batch_size: int = 16,
    num_epochs: int = 20,
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
    
    # ========================================================================
    # Step 1: Prepare dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING DATASET")
    print("=" * 80)
    
    labels = prepare_dataset(
        dataset_root=dataset_root,
        real_folder=real_folder,
        hybrid_folder=hybrid_folder,
        mask_folder=mask_folder,
        verify_masks=(mask_folder is not None)
    )
    
    if len(labels) == 0:
        raise ValueError("No images found! Check your folder paths.")
    
    # ========================================================================
    # Step 2: Split data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING DATA")
    print("=" * 80)
    
    from pathlib import Path
    import random

    def get_base_id(path):
        """
        Extract base ID from filenames like:
        00000_00000_000003083_origin_0.png
        00000_00000_000003083_result_0.png
        """
        name = Path(path).stem

        if name.endswith("_origin_0"):
            return name.replace("_origin_0", "")
        if name.endswith("_result_0"):
            return name.replace("_result_0", "")

        # fallback safety
        return name


    # Group real + hybrid by base image
    groups = {}
    for item in labels:
        base = get_base_id(item[0])
        groups.setdefault(base, []).append(item)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    val_cut = int(len(group_keys) * val_split)
    val_keys = set(group_keys[:val_cut])

    train_labels, val_labels = [], []
    for k, items in groups.items():
        if k in val_keys:
            val_labels.extend(items)
        else:
            train_labels.extend(items)

    print(f"\n✅ Pair-aware split complete:")
    print(f"   ├─ Train pairs: {len(set(get_base_id(x[0]) for x in train_labels))}")
    print(f"   └─ Val pairs:   {len(set(get_base_id(x[0]) for x in val_labels))}")

    # -------- SMALL DEBUG RUN (300 images total) --------
    def limit_by_pairs(labels, max_pairs=150):
        from collections import defaultdict

        groups = defaultdict(list)

        # group by base id
        for item in labels:
            path = item[0]
            base = get_base_id(path)
            groups[base].append(item)

        selected = []
        pair_count = 0

        for base, items in groups.items():
            # expect exactly 2: one real (0) and one hybrid (1)
            real = [x for x in items if x[1] == 0]
            hybrid = [x for x in items if x[1] == 1]

            if len(real) == 1 and len(hybrid) == 1:
                selected.append(real[0])
                selected.append(hybrid[0])
                pair_count += 1

            if pair_count >= max_pairs:
                break

        return selected


    print(f"\n🧪 DEBUG MODE (PAIR-AWARE):")
    print(f"   ├─ Train samples: {len(train_labels)}")
    print(f"   └─ Val samples:   {len(val_labels)}")

    train_real = sum(1 for l in train_labels if l[1] == 0)
    train_hybrid = sum(1 for l in train_labels if l[1] == 1)
    val_real = sum(1 for l in val_labels if l[1] == 0)
    val_hybrid = sum(1 for l in val_labels if l[1] == 1)

    print(f"\n   Training set:")
    print(f"   ├─ Real:   {train_real}")
    print(f"   └─ Hybrid: {train_hybrid}")

    print(f"\n   Validation set:")
    print(f"   ├─ Real:   {val_real}")
    print(f"   └─ Hybrid: {val_hybrid}")

    
    # ========================================================================
    # Step 3: Create datasets and loaders
    # ========================================================================
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
    
    # ========================================================================
    # Step 4: Initialize model
    # ========================================================================
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
    
    # ========================================================================
    # Step 5: Optimizer with differential learning rates
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SETTING UP TRAINING")
    print("=" * 80)
    
    # Pretrained backbone gets 10x lower LR than the new heads.
    # The backbone already has good general features; the new freq/edge/classifier
    # heads need to learn from scratch so they need higher LR.
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'rgb_encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.01},
        {'params': head_params,     'lr': learning_rate,        'weight_decay': 0.01},
    ])
    print(f"✅ Optimizer: AdamW")
    print(f"   ├─ Backbone LR: {learning_rate * 0.1} ({len(backbone_params)} param groups)")
    print(f"   └─ Head LR:     {learning_rate} ({len(head_params)} param groups)")
    
    # Cosine schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)
    print(f"✅ Scheduler: CosineAnnealingLR (T_max={num_epochs})")
    
    

    # Mixed precision
    scaler = GradScaler()
    print(f"✅ Mixed Precision: Enabled")
    
    # Save directory
    save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ Save directory: {save_dir}")
    
    # ========================================================================
    # Step 6: Training loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING")
    print("=" * 80)
    print(f"\nStarting training for {num_epochs} epochs...")
    
    best_metric = -1e9
    history = {
    'train_loss': [],
    'real_mean': [],
    'hybrid_mean': [],
    'gap': []}

    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        # --------------------------------------------------------------------
        # Training
        # --------------------------------------------------------------------
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc="Training")

        for batch_idx, (rgb, freq, edge_map, labels, masks) in enumerate(pbar):

            rgb = rgb.to(device)
            freq = freq.to(device)
            edge_map = edge_map.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            with autocast():

                _, scores = model(rgb, freq, edge_map)

                risk = (
                    0.30 * torch.sigmoid(scores["patch"]) +
                    0.30 * torch.sigmoid(scores["texture"]) +
                    0.40 * torch.sigmoid(scores["energy"])
                )

                real_risk   = risk[labels == 0]
                hybrid_risk = risk[labels == 1]

                if len(real_risk) == 0 or len(hybrid_risk) == 0:
                    continue

                # 1️⃣ Tail ranking (only if confident)
                tail_threshold = 0.70
                hard_real   = real_risk[real_risk > tail_threshold]
                hard_hybrid = hybrid_risk[hybrid_risk > tail_threshold]

                if len(hard_real) > 0 and len(hard_hybrid) > 0:
                    tail_rank_loss = torch.relu(
                        0.15 - (hard_hybrid.mean() - hard_real.mean())
                    )
                else:
                    tail_rank_loss = torch.tensor(0.0, device=risk.device)

                # 2️⃣ Weak global ranking
                global_rank_loss = torch.relu(
                    0.05 - (hybrid_risk.mean() - real_risk.mean())
                )

                # 3️⃣ Spread (anti-collapse)
                spread_loss = torch.relu(0.10 - risk.std(unbiased=False))

                # 4️⃣ Mid-risk entropy nudge (break 0.50 spike)
                mid_mask = (risk > 0.45) & (risk < 0.55)
                if mid_mask.any():
                    entropy_nudge = -(
                        risk[mid_mask] * torch.log(risk[mid_mask] + 1e-6) +
                        (1 - risk[mid_mask]) * torch.log(1 - risk[mid_mask] + 1e-6)
                    ).mean()
                else:
                    entropy_nudge = torch.tensor(0.0, device=risk.device)

                loss = (
                    1.0 * tail_rank_loss +
                    0.3 * global_rank_loss +
                    0.3 * spread_loss +
                    0.1 * entropy_nudge)

                # Debug only once
                if batch_idx == 0:
                    print(
                        f"[RISK-DEBUG] "
                        f"real_mean={real_risk.mean().item():.3f}, "
                        f"hybrid_mean={hybrid_risk.mean().item():.3f}, "
                        f"gap={(hybrid_risk.mean() - real_risk.mean()).item():.3f}, "
                        f"std={risk.std(unbiased=False).item():.3f}"
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        
        # --------------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------------
        model.eval()

        all_risk = []
        all_labels = []

        with torch.no_grad():
            for rgb, freq, edge_map, labels, masks in tqdm(val_loader, desc="Validation"):

                rgb = rgb.to(device)
                freq = freq.to(device)
                edge_map = edge_map.to(device)
                labels = labels.to(device).float()

                _, scores = model(rgb, freq, edge_map)

                risk = (
                    0.30 * torch.sigmoid(scores["patch"]) +
                    0.30 * torch.sigmoid(scores["texture"]) +
                    0.40 * torch.sigmoid(scores["energy"])
                )

                all_risk.append(risk.cpu())
                all_labels.append(labels.cpu())

        all_risk = torch.cat(all_risk).numpy()
        all_labels = torch.cat(all_labels).numpy()

        real_mean   = all_risk[all_labels == 0].mean()
        hybrid_mean = all_risk[all_labels == 1].mean()
        gap = hybrid_mean - real_mean
        risk_std = all_risk.std()

        print(
            f"Validation Risk Stats → "
            f"Real mean: {real_mean:.3f} | "
            f"Hybrid mean: {hybrid_mean:.3f} | "
            f"Gap: {gap:.3f} | "
            f"STD: {risk_std:.3f}"
        )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        history['train_loss'].append(float(avg_train_loss))
        history['real_mean'].append(real_mean)
        history['hybrid_mean'].append(hybrid_mean)
        history['gap'].append(float(gap))

       

        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   ├─ Train Loss: {avg_train_loss:.4f}")
        print(f"   └─ Mean Gap: {gap:.4f}")

        # --------------------------------------------------
        # Checkpointing
        # --------------------------------------------------
        metric = gap  # or gap - 0.5 * std_penalty
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            print(f"\n   ✅ New best model! (mean_gap: {best_metric:.4f})")
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gap': gap
                },
                os.path.join(save_dir, 'best.pth')
            )

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            },
            os.path.join(save_dir, 'latest.pth')
        )

        scheduler.step(gap)

    # Save history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Best mean_gap: {best_metric:.4f}")
    print(f"✅ Model saved to: {save_dir}/best.pth")
    print(f"✅ Training history saved to: {save_dir}/history.json")
    print("\n" + "=" * 80)
    
    return history, save_dir


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    
    CONFIG = {
        'dataset_root': '/kaggle/input/hybrid-dataset/hybrid-dataset',
        'real_folder': 'real',
        'hybrid_folder': 'hybrid',
        'mask_folder': None,                  # Not needed for lite model
        
        'use_lite_model': True,
        'use_localization': False,
        
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 3e-4,
        'val_split': 0.2,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        
        'save_dir': '/kaggle/working/checkpoints',
        'experiment_name': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    history, save_dir = train_from_folders(**CONFIG)
