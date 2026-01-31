"""
Diagnostic Script for Hybrid Detection Training Issues
This will help identify why AUC is stuck at 0.50 (random guess)

Run this BEFORE training to verify:
1. Mask format (white vs black convention)
2. Data loading correctness
3. Model forward pass
4. Loss computation
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import your modules
from hybrid_dataset import HybridImageDataset
from hybrid_detection_model import HybridImageDetector, HybridDetectorLite
from prepare_data import prepare_dataset


def check_mask_convention(dataset_root, mask_folder='mask', num_samples=5):
    """
    Check if masks follow the correct convention
    CORRECT: 1 (white/255) = edited region, 0 (black) = real region
    WRONG: 0 (black) = edited region, 1 (white/255) = real region
    """
    print("\n" + "="*80)
    print("STEP 1: CHECKING MASK CONVENTION")
    print("="*80)
    
    mask_dir = Path(dataset_root) / mask_folder
    if not mask_dir.exists():
        print(f"❌ Mask folder not found: {mask_dir}")
        return None
    
    mask_files = list(mask_dir.glob('*.png'))[:num_samples]
    
    if len(mask_files) == 0:
        print(f"❌ No PNG masks found in {mask_dir}")
        return None
    
    print(f"\n📊 Analyzing {len(mask_files)} masks...\n")
    
    results = []
    for mask_path in mask_files:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        unique_vals = np.unique(mask_array)
        mean_val = mask_array.mean()
        
        # Determine likely convention
        white_pixels = (mask_array > 127).sum()
        black_pixels = (mask_array < 128).sum()
        total_pixels = mask_array.size
        
        white_ratio = white_pixels / total_pixels
        
        print(f"Mask: {mask_path.name}")
        print(f"  Shape: {mask_array.shape}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Mean value: {mean_val:.2f}")
        print(f"  White pixels (>127): {white_pixels} ({white_ratio*100:.1f}%)")
        print(f"  Black pixels (<128): {black_pixels} ({(1-white_ratio)*100:.1f}%)")
        
        # Interpretation
        if white_ratio > 0.5:
            print(f"  ⚠️  INTERPRETATION: Mask is mostly WHITE")
            print(f"      If white = edited region → This mask indicates LARGE edit")
            print(f"      If white = background → YOUR MASKS ARE INVERTED!")
        else:
            print(f"  ✅ INTERPRETATION: Mask is mostly BLACK (background)")
            print(f"      If black = background, white = edited → Correct!")
        
        results.append({
            'path': mask_path.name,
            'white_ratio': white_ratio,
            'unique_vals': unique_vals
        })
        print()
    
    # Summary
    avg_white = np.mean([r['white_ratio'] for r in results])
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"Average white pixel ratio: {avg_white*100:.1f}%")
    
    if avg_white > 0.5:
        print("\n❌ YOUR MASKS APPEAR TO BE INVERTED!")
        print("Expected: BLACK (0) = background, WHITE (255) = edited region")
        print("Your masks: WHITE (255) = background, BLACK (0) = edited region")
        print("\nFIX: You need to invert your masks!")
        return "INVERTED"
    else:
        print("\n✅ Masks appear to follow correct convention")
        print("BLACK (0) = background, WHITE (255) = edited region")
        return "CORRECT"
    
    print("="*80)


def check_dataset_loading(dataset_root, real_folder='real', hybrid_folder='hybrid', 
                          mask_folder='mask', num_samples=3):
    """
    Verify that dataset loads correctly and check label distribution
    """
    print("\n" + "="*80)
    print("STEP 2: CHECKING DATASET LOADING")
    print("="*80)
    
    # Prepare labels
    labels = prepare_dataset(
        dataset_root=dataset_root,
        real_folder=real_folder,
        hybrid_folder=hybrid_folder,
        mask_folder=mask_folder,
        verify_masks=False
    )
    
    # Create dataset
    dataset = HybridImageDataset(
        img_dir=dataset_root,
        labels=labels,
        mode='train',
        use_augmentation=False  # Disable for testing
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # Check label distribution
    label_counts = {0: 0, 1: 0}
    for _, label, _ in labels:
        label_counts[label] += 1
    
    print(f"  Real images (label=0): {label_counts[0]}")
    print(f"  Hybrid images (label=1): {label_counts[1]}")
    print(f"  Class balance: {label_counts[0]/label_counts[1]:.2f}:1")
    
    # Sample a few items
    print(f"\n🔍 Sampling {num_samples} items from dataset:")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            rgb, freq, label, mask = dataset[i]
            
            print(f"\nSample {i}:")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  Freq shape: {freq.shape}")
            print(f"  Label: {label.item():.0f} ({'HYBRID' if label.item() == 1 else 'REAL'})")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask range: [{mask.min().item():.3f}, {mask.max().item():.3f}]")
            print(f"  Mask mean: {mask.mean().item():.3f}")
            
            # Check if mask makes sense
            if label.item() == 1:  # Hybrid
                if mask.mean().item() < 0.01:
                    print(f"  ⚠️  WARNING: Hybrid image but mask is all zeros!")
                elif mask.mean().item() > 0.99:
                    print(f"  ⚠️  WARNING: Hybrid image but mask is all ones!")
                else:
                    print(f"  ✅ Mask has reasonable values")
            else:  # Real
                if mask.mean().item() > 0.01:
                    print(f"  ⚠️  WARNING: Real image but mask is not zero!")
                    
        except Exception as e:
            print(f"\n❌ Error loading sample {i}: {e}")
    
    print("\n" + "="*80)
    return dataset


def check_model_forward_pass(dataset, device='cuda'):
    """
    Test model forward pass and verify outputs
    """
    print("\n" + "="*80)
    print("STEP 3: CHECKING MODEL FORWARD PASS")
    print("="*80)
    
    if not torch.cuda.is_available():
        device = 'cpu'
        print(f"⚠️  CUDA not available, using CPU")
    
    # Create batch
    batch_size = 4
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    
    rgb = torch.stack([s[0] for s in samples]).to(device)
    freq = torch.stack([s[1] for s in samples]).to(device)
    labels = torch.stack([s[2] for s in samples]).to(device)
    masks = torch.stack([s[3] for s in samples]).to(device)
    
    print(f"\n📦 Batch shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Freq: {freq.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Masks: {masks.shape}")
    
    # Test both models
    models = {
        'Lite': HybridDetectorLite(),
        'Full': HybridImageDetector()
    }
    
    for model_name, model in models.items():
        print(f"\n🔬 Testing {model_name} Model:")
        model = model.to(device)
        model.eval()
        
        try:
            with torch.no_grad():
                cls_logits, loc_maps = model(rgb, freq)
            
            print(f"  ✅ Forward pass successful")
            print(f"  Classification logits shape: {cls_logits.shape}")
            print(f"  Classification logits range: [{cls_logits.min().item():.3f}, {cls_logits.max().item():.3f}]")
            
            # Convert to probabilities
            probs = torch.sigmoid(cls_logits.squeeze())
            print(f"  Probabilities: {probs.cpu().numpy()}")
            print(f"  Mean probability: {probs.mean().item():.3f}")
            
            # Check if model is just guessing
            if 0.45 < probs.mean().item() < 0.55:
                print(f"  ⚠️  WARNING: Probabilities centered around 0.5 (random guessing)")
            
            if loc_maps is not None:
                print(f"  Localization maps shape: {loc_maps.shape}")
                print(f"  Localization maps range: [{loc_maps.min().item():.3f}, {loc_maps.max().item():.3f}]")
            else:
                print(f"  Localization maps: None (expected for Lite model)")
                
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)


def check_loss_computation(dataset, device='cuda'):
    """
    Test loss computation to ensure gradients flow properly
    """
    print("\n" + "="*80)
    print("STEP 4: CHECKING LOSS COMPUTATION")
    print("="*80)
    
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # Create batch with mixed labels
    samples = []
    real_count = 0
    hybrid_count = 0
    
    for i in range(len(dataset)):
        _, label, _ = dataset.labels[i]
        if label == 0 and real_count < 2:
            samples.append(dataset[i])
            real_count += 1
        elif label == 1 and hybrid_count < 2:
            samples.append(dataset[i])
            hybrid_count += 1
        
        if len(samples) >= 4:
            break
    
    if len(samples) == 0:
        print("❌ Could not create test batch")
        return
    
    rgb = torch.stack([s[0] for s in samples]).to(device)
    freq = torch.stack([s[1] for s in samples]).to(device)
    labels = torch.stack([s[2] for s in samples]).to(device)
    masks = torch.stack([s[3] for s in samples]).to(device)
    
    print(f"\n📊 Test batch:")
    print(f"  Labels: {labels.cpu().numpy()}")
    print(f"  Real images: {(labels == 0).sum().item()}")
    print(f"  Hybrid images: {(labels == 1).sum().item()}")
    
    # Test with Full model
    model = HybridImageDetector().to(device)
    model.train()
    
    # Forward pass
    cls_logits, loc_maps = model(rgb, freq)
    
    # Classification loss
    cls_loss_fn = torch.nn.BCEWithLogitsLoss()
    cls_loss = cls_loss_fn(cls_logits.squeeze(dim=1), labels)
    
    print(f"\n📉 Classification Loss:")
    print(f"  Value: {cls_loss.item():.4f}")
    print(f"  Expected range: ~0.693 for random guess")
    
    if 0.65 < cls_loss.item() < 0.72:
        print(f"  ⚠️  Loss suggests random guessing!")
    
    # Test backward pass
    cls_loss.backward()
    
    # Check gradients
    has_grads = False
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    if has_grads:
        print(f"\n✅ Gradients computed successfully")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
        print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
        
        if np.mean(grad_norms) < 1e-6:
            print(f"  ⚠️  WARNING: Gradients are very small (vanishing gradient?)")
    else:
        print(f"\n❌ No gradients computed!")
    
    print("\n" + "="*80)


def check_data_augmentation_impact(dataset):
    """
    Check if augmentation is too aggressive and destroying signal
    """
    print("\n" + "="*80)
    print("STEP 5: CHECKING DATA AUGMENTATION")
    print("="*80)
    
    # Get same image with and without augmentation
    dataset_no_aug = HybridImageDataset(
        img_dir=dataset.img_dir,
        labels=dataset.labels[:10],
        mode='val',
        use_augmentation=False
    )
    
    dataset_with_aug = HybridImageDataset(
        img_dir=dataset.img_dir,
        labels=dataset.labels[:10],
        mode='train',
        use_augmentation=True
    )
    
    print("\n🔍 Comparing augmented vs non-augmented samples:")
    
    for i in range(min(3, len(dataset_no_aug))):
        rgb_no_aug, freq_no_aug, _, _ = dataset_no_aug[i]
        rgb_aug, freq_aug, _, _ = dataset_with_aug[i]
        
        print(f"\nSample {i}:")
        print(f"  No augmentation - RGB mean: {rgb_no_aug.mean().item():.3f}, std: {rgb_no_aug.std().item():.3f}")
        print(f"  With augmentation - RGB mean: {rgb_aug.mean().item():.3f}, std: {rgb_aug.std().item():.3f}")
        
        # Check if augmentation is too extreme
        if abs(rgb_no_aug.mean().item() - rgb_aug.mean().item()) > 1.0:
            print(f"  ⚠️  WARNING: Augmentation changed mean significantly!")
    
    print("\n" + "="*80)


def main_diagnostic(dataset_root, real_folder='real', hybrid_folder='hybrid', 
                    mask_folder='mask'):
    """
    Run complete diagnostic suite
    """
    print("\n" + "="*80)
    print("HYBRID DETECTION TRAINING DIAGNOSTICS")
    print("="*80)
    print(f"\nDataset root: {dataset_root}")
    print(f"Real folder: {real_folder}")
    print(f"Hybrid folder: {hybrid_folder}")
    print(f"Mask folder: {mask_folder}")
    
    # Check mask convention
    mask_status = check_mask_convention(dataset_root, mask_folder)
    
    # Check dataset loading
    dataset = check_dataset_loading(dataset_root, real_folder, hybrid_folder, mask_folder)
    
    if dataset is None:
        print("\n❌ Cannot proceed - dataset loading failed")
        return
    
    # Check model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_model_forward_pass(dataset, device)
    
    # Check loss
    check_loss_computation(dataset, device)
    
    # Check augmentation
    check_data_augmentation_impact(dataset)
    
    # Final recommendations
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE - RECOMMENDATIONS:")
    print("="*80)
    
    if mask_status == "INVERTED":
        print("\n❌ CRITICAL ISSUE: Your masks are inverted!")
        print("   ACTION: Run the mask inversion script (see below)")
    
    print("\n✅ Next steps:")
    print("   1. Review the diagnostic output above")
    print("   2. If masks are inverted, run mask_inversion.py")
    print("   3. Check that labels are correctly balanced")
    print("   4. Try training with fewer augmentations initially")
    print("   5. Start with lite model and fewer epochs to verify")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # CHANGE THIS TO YOUR DATASET PATH
    DATASET_ROOT = '/kaggle/input/hybrid-dataset/hybrid-dataset'
    
    main_diagnostic(
        dataset_root=DATASET_ROOT,
        real_folder='real',
        hybrid_folder='hybrid',
        mask_folder='mask'
    )
