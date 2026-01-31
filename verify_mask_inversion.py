"""
Simple Mask Inversion Verification Script
Shows EXACTLY what happens during training with correct vs inverted masks
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def simulate_training_with_masks(mask_is_inverted=False):
    """
    Simulate what happens during training to show the impact of inverted masks
    """
    print("\n" + "="*80)
    print(f"SIMULATION: Training with {'INVERTED' if mask_is_inverted else 'CORRECT'} masks")
    print("="*80 + "\n")
    
    # Create a simple example
    # Image: 8x8 pixels
    # Edit region: bottom-right 4x4 quadrant
    
    # Ground truth: bottom-right is edited
    true_edit_region = np.zeros((8, 8))
    true_edit_region[4:, 4:] = 1  # Bottom-right quadrant is edited
    
    print("Ground Truth Edit Region:")
    print("(1 = edited, 0 = real)")
    print(true_edit_region.astype(int))
    print()
    
    # What your mask file contains (assuming it's inverted)
    if mask_is_inverted:
        # INVERTED: white=background(255), black=edit(0)
        mask_file_values = np.zeros((8, 8))
        mask_file_values[4:, 4:] = 0    # Edited region = BLACK (0)
        mask_file_values[:4, :] = 255   # Real region = WHITE (255)
        mask_file_values[4:, :4] = 255  # Real region = WHITE (255)
        
        print("Mask File Content (INVERTED):")
        print("(255 = background/real, 0 = edited)")
        print(mask_file_values.astype(int))
        print()
    else:
        # CORRECT: black=background(0), white=edit(255)
        mask_file_values = np.zeros((8, 8))
        mask_file_values[4:, 4:] = 255  # Edited region = WHITE (255)
        
        print("Mask File Content (CORRECT):")
        print("(0 = background/real, 255 = edited)")
        print(mask_file_values.astype(int))
        print()
    
    # What the training code receives after normalization
    # The dataset code converts 0-255 to 0-1 range using ToTensor()
    mask_tensor = torch.from_numpy(mask_file_values / 255.0).float()
    
    print("What Training Code Receives (after ToTensor):")
    print("(1.0 = signal for edited, 0.0 = signal for real)")
    print(mask_tensor.numpy())
    print()
    
    # The model predicts something (random at start)
    model_prediction = torch.rand((8, 8))  # Random values between 0 and 1
    
    print("Model Prediction (example):")
    print(model_prediction.numpy().round(2))
    print()
    
    # Loss computation (Dice Loss)
    # The model is trying to predict the MASK TENSOR values
    # If mask is inverted, it learns the WRONG thing!
    
    intersection = (model_prediction * mask_tensor).sum()
    dice = (2. * intersection) / (model_prediction.sum() + mask_tensor.sum())
    loss = 1 - dice
    
    print(f"Dice Loss: {loss.item():.4f}")
    print()
    
    # What the model learns
    if mask_is_inverted:
        print("❌ PROBLEM WITH INVERTED MASKS:")
        print("   Model learns to predict:")
        print("   - 1.0 for background/real regions (wrong!)")
        print("   - 0.0 for edited regions (wrong!)")
        print("   This is OPPOSITE of what we want!")
    else:
        print("✅ CORRECT MASKS:")
        print("   Model learns to predict:")
        print("   - 1.0 for edited regions (correct!)")
        print("   - 0.0 for background/real regions (correct!)")
    
    print("\n" + "="*80 + "\n")
    
    return mask_tensor


def show_inversion_fix():
    """
    Show how to fix inverted masks
    """
    print("\n" + "="*80)
    print("HOW TO FIX INVERTED MASKS")
    print("="*80 + "\n")
    
    # Your inverted mask
    inverted_mask = np.array([
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255,   0,   0,   0,   0],
        [255, 255, 255, 255,   0,   0,   0,   0],
        [255, 255, 255, 255,   0,   0,   0,   0],
        [255, 255, 255, 255,   0,   0,   0,   0],
    ], dtype=np.uint8)
    
    print("BEFORE (Inverted):")
    print("(255 = background, 0 = edited)")
    print(inverted_mask)
    print()
    
    # Fix: 255 - value
    corrected_mask = 255 - inverted_mask
    
    print("AFTER (Corrected):")
    print("(0 = background, 255 = edited)")
    print(corrected_mask)
    print()
    
    print("Python code to invert:")
    print("```python")
    print("from PIL import Image")
    print("import numpy as np")
    print("")
    print("# Load inverted mask")
    print("mask = Image.open('inverted_mask.png').convert('L')")
    print("mask_array = np.array(mask)")
    print("")
    print("# Invert")
    print("corrected = 255 - mask_array")
    print("")
    print("# Save")
    print("Image.fromarray(corrected.astype(np.uint8)).save('corrected_mask.png')")
    print("```")
    
    print("\n" + "="*80 + "\n")


def visualize_mask_impact():
    """
    Create visual comparison of correct vs inverted masks
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Create sample image (8x8)
    sample_image = np.random.rand(8, 8, 3) * 255
    
    # Ground truth edit region
    true_edit = np.zeros((8, 8))
    true_edit[4:, 4:] = 1
    
    # Inverted mask (what you have)
    inverted_mask = np.zeros((8, 8))
    inverted_mask[:4, :] = 255
    inverted_mask[4:, :4] = 255
    
    # Corrected mask
    corrected_mask = 255 - inverted_mask
    
    # Row 1: Inverted masks
    axes[0, 0].imshow(sample_image.astype(np.uint8))
    axes[0, 0].set_title('Sample Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(inverted_mask, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Your Mask (INVERTED)\nWhite=BG, Black=Edit')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inverted_mask / 255.0, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Training Receives\n(Model learns WRONG)')
    axes[0, 2].axis('off')
    
    # Row 2: Corrected masks
    axes[1, 0].imshow(sample_image.astype(np.uint8))
    axes[1, 0].set_title('Sample Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(corrected_mask, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Corrected Mask\nBlack=BG, White=Edit')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(corrected_mask / 255.0, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Training Receives\n(Model learns CORRECT)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/mask_impact_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to mask_impact_visualization.png\n")
    plt.close()


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("MASK INVERSION - COMPLETE EXPLANATION")
    print("="*80)
    
    # Simulate with inverted masks
    print("\n1️⃣ SIMULATION WITH INVERTED MASKS (YOUR CURRENT SITUATION)")
    mask_inverted = simulate_training_with_masks(mask_is_inverted=True)
    
    # Simulate with correct masks
    print("\n2️⃣ SIMULATION WITH CORRECT MASKS (WHAT YOU NEED)")
    mask_correct = simulate_training_with_masks(mask_is_inverted=False)
    
    # Show how to fix
    print("\n3️⃣ HOW TO FIX")
    show_inversion_fix()
    
    # Visual comparison
    print("\n4️⃣ CREATING VISUAL COMPARISON")
    visualize_mask_impact()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n📌 KEY POINTS:")
    print()
    print("1. Your masks have: WHITE (255) = background, BLACK (0) = edited")
    print("   Training expects: BLACK (0) = background, WHITE (255) = edited")
    print()
    print("2. When masks are inverted:")
    print("   - Model learns that background = edited (WRONG)")
    print("   - Model learns that edits = background (WRONG)")
    print("   - Localization loss becomes meaningless")
    print("   - AUC stuck at 0.50 (random guessing)")
    print()
    print("3. To fix:")
    print("   inverted_mask = 255 - corrected_mask")
    print()
    print("4. When to use masks:")
    print("   - use_localization=True  → Masks ARE used")
    print("   - use_localization=False → Masks NOT used")
    print()
    print("5. Next steps:")
    print("   a. Run: python invert_masks.py (dry run first)")
    print("   b. Verify inverted masks look correct")
    print("   c. Train with: mask_folder='masks_inverted'")
    print("   d. Check AUC improves above 0.60")
    print()
    print("="*80 + "\n")
