"""
Mask Inversion Script

Your masks have: WHITE (255) = background, BLACK (0) = edited region
Correct format: BLACK (0) = background, WHITE (255) = edited region

This script will:
1. Verify masks are inverted
2. Create inverted copies
3. Provide visual comparison
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def verify_and_invert_masks(
    mask_dir: str,
    output_dir: str = None,
    visualize: bool = True,
    dry_run: bool = False
):
    """
    Invert masks: BLACK ↔ WHITE
    
    Args:
        mask_dir: Directory containing masks to invert
        output_dir: Where to save inverted masks (if None, overwrites original)
        visualize: Show before/after comparison
        dry_run: Don't actually save files, just show what would happen
    """
    
    mask_dir = Path(mask_dir)
    
    if output_dir is None:
        output_dir = mask_dir / "inverted"
    else:
        output_dir = Path(output_dir)
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG masks
    mask_files = list(mask_dir.glob('*.png'))
    
    if len(mask_files) == 0:
        print(f"❌ No PNG files found in {mask_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"MASK INVERSION")
    print(f"{'='*80}")
    print(f"\nSource directory: {mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of masks: {len(mask_files)}")
    print(f"Dry run: {'Yes (no files will be modified)' if dry_run else 'No'}")
    print(f"{'='*80}\n")
    
    # Analyze first few to confirm inversion needed
    print("📊 Analyzing sample masks to verify inversion is needed...\n")
    
    sample_masks = mask_files[:5]
    needs_inversion = []
    
    for mask_path in sample_masks:
        mask = np.array(Image.open(mask_path).convert('L'))
        white_ratio = (mask > 127).sum() / mask.size
        
        print(f"{mask_path.name}:")
        print(f"  White pixels: {white_ratio*100:.1f}%")
        print(f"  Black pixels: {(1-white_ratio)*100:.1f}%")
        
        # If mostly white, probably needs inversion
        if white_ratio > 0.5:
            print(f"  → Appears to need inversion (background is white)")
            needs_inversion.append(True)
        else:
            print(f"  → Appears correct (background is black)")
            needs_inversion.append(False)
        print()
    
    # Decision
    inversion_votes = sum(needs_inversion)
    
    if inversion_votes == 0:
        print("✅ Masks appear to already be in correct format!")
        print("   BLACK = background, WHITE = edited region")
        response = input("\nDo you still want to invert them? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    elif inversion_votes == len(needs_inversion):
        print("❌ Masks appear to be INVERTED!")
        print("   Current: WHITE = background, BLACK = edited")
        print("   After inversion: BLACK = background, WHITE = edited")
        print("\n✅ Proceeding with inversion...")
    else:
        print("⚠️  WARNING: Masks have mixed formats!")
        print(f"   {inversion_votes}/{len(needs_inversion)} appear to need inversion")
        response = input("\nDo you want to proceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Process all masks
    print(f"\n{'='*80}")
    print("INVERTING MASKS")
    print(f"{'='*80}\n")
    
    processed = 0
    errors = 0
    
    for mask_path in tqdm(mask_files, desc="Processing masks"):
        try:
            # Load mask
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            
            # Invert: 255 - pixel_value
            inverted_array = 255 - mask_array
            
            # Convert back to image
            inverted_mask = Image.fromarray(inverted_array.astype(np.uint8))
            
            # Save
            if not dry_run:
                output_path = output_dir / mask_path.name
                inverted_mask.save(output_path)
            
            processed += 1
            
            # Visualize first few
            if visualize and processed <= 3:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(mask_array, cmap='gray')
                plt.title(f'Original\n{mask_path.name}')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(inverted_array, cmap='gray')
                plt.title('Inverted')
                plt.axis('off')
                
                # Difference
                plt.subplot(1, 3, 3)
                plt.imshow(inverted_array - mask_array, cmap='RdBu', vmin=-255, vmax=255)
                plt.title('Difference')
                plt.axis('off')
                
                plt.tight_layout()
                
                if dry_run:
                    plt.savefig(f'/home/claude/mask_comparison_{processed}.png', dpi=150, bbox_inches='tight')
                    print(f"  Saved comparison to mask_comparison_{processed}.png")
                
                plt.show()
                plt.close()
        
        except Exception as e:
            print(f"❌ Error processing {mask_path.name}: {e}")
            errors += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n✅ Processed: {processed}/{len(mask_files)}")
    
    if errors > 0:
        print(f"❌ Errors: {errors}")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN - No files were actually saved")
        print(f"   Remove dry_run=True to save inverted masks")
    else:
        print(f"\n✅ Inverted masks saved to: {output_dir}")
        print(f"\n📝 Next steps:")
        print(f"   1. Verify the inverted masks look correct")
        print(f"   2. Update your training script to use: mask_folder='{output_dir.name}'")
        print(f"   3. Re-run training")
    
    print(f"{'='*80}\n")


def compare_mask_with_image(image_path: str, mask_path: str):
    """
    Overlay mask on image to verify correctness
    """
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # Resize mask to match image
    mask = mask.resize(img.size, Image.NEAREST)
    
    # Convert to arrays
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    # Create overlay (red for edited region)
    overlay = img_array.copy()
    overlay[mask_array > 127, 0] = 255  # Red channel
    
    # Blend
    blended = (0.6 * img_array + 0.4 * overlay).astype(np.uint8)
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_array)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_array, cmap='gray')
    plt.title('Mask\n(white = edited region)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title('Overlay\n(red = edited region)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/mask_overlay_verification.png', dpi=150, bbox_inches='tight')
    print("✅ Saved overlay to mask_overlay_verification.png")
    plt.show()


# ============================================================================
# QUICK USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # EXAMPLE 1: Dry run (safe - just shows what would happen)
    print("="*80)
    print("EXAMPLE 1: DRY RUN")
    print("="*80)
    
    verify_and_invert_masks(
        mask_dir='/kaggle/input/hybrid-dataset/hybrid-dataset/mask',
        output_dir='/kaggle/working/masks_inverted',
        visualize=True,
        dry_run=True  # Safe mode - doesn't save files
    )
    
    # EXAMPLE 2: Actually invert and save
    # Uncomment below when ready:
    """
    print("="*80)
    print("EXAMPLE 2: ACTUAL INVERSION")
    print("="*80)
    
    verify_and_invert_masks(
        mask_dir='/kaggle/input/hybrid-dataset/hybrid-dataset/mask',
        output_dir='/kaggle/working/masks_inverted',
        visualize=True,
        dry_run=False  # Actually save inverted masks
    )
    """
    
    # EXAMPLE 3: Verify mask with corresponding image
    # Uncomment and update paths:
    """
    compare_mask_with_image(
        image_path='/path/to/hybrid/image.jpg',
        mask_path='/path/to/mask/image_mask.png'
    )
    """
