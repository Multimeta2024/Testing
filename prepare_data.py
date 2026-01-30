"""
Data Preparation Script for Hybrid Image Detection
Handles folder structure: real/ and hybrid/ with PNG masks
"""

import os
from pathlib import Path
import json
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np


class DataPreparer:
    """
    Prepare data from folder structure:
    
    dataset/
    ├── real/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── hybrid/  (or fake/)
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── masks/  (optional)
        ├── img1.png
        ├── img2.png
        └── ...
    """
    
    def __init__(self, 
                 dataset_root: str,
                 real_folder: str = 'real',
                 hybrid_folder: str = 'hybrid',
                 mask_folder: Optional[str] = 'masks'):
        """
        Args:
            dataset_root: Root directory containing real/, hybrid/, masks/
            real_folder: Name of folder with real images (default: 'real')
            hybrid_folder: Name of folder with hybrid/fake images (default: 'hybrid')
            mask_folder: Name of folder with masks (default: 'masks', None if no masks)
        """
        self.dataset_root = Path(dataset_root)
        self.real_folder = self.dataset_root / real_folder
        self.hybrid_folder = self.dataset_root / hybrid_folder
        self.mask_folder = self.dataset_root / mask_folder if mask_folder else None
        
        # Validate folders exist
        if not self.real_folder.exists():
            raise ValueError(f"Real folder not found: {self.real_folder}")
        if not self.hybrid_folder.exists():
            raise ValueError(f"Hybrid folder not found: {self.hybrid_folder}")
        
        if self.mask_folder and not self.mask_folder.exists():
            print(f"⚠️  Warning: Mask folder not found: {self.mask_folder}")
            self.mask_folder = None
    
    def get_image_files(self, folder: Path) -> List[Path]:
        """Get all image files from a folder"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        files = []
        for ext in extensions:
            files.extend(folder.glob(f'*{ext}'))
            files.extend(folder.glob(f'*{ext.upper()}'))
        return sorted(files)
    
    def find_mask_for_image(self, image_path: Path) -> Optional[Path]:
        """Find corresponding mask for an image"""
        if not self.mask_folder:
            return None

        base_stem = image_path.stem.replace("_result_0", "").replace("_result", "")

        mask_name = f"{base_stem}_mask_0.png"
        mask_path = self.mask_folder / mask_name

        if mask_path.exists():
            return mask_path

        return None

    
    def create_labels_list(self) -> List[Tuple[str, int, Optional[str]]]:
        """
        Create labels list in the format:
        [(relative_img_path, label, relative_mask_path), ...]
        
        Returns:
            List of tuples: (image_path, label, mask_path)
            - label: 0 = real, 1 = hybrid
            - mask_path: relative path to mask or None
        """
        labels = []
        
        # Process real images (label = 0)
        print("\n📁 Processing real images...")
        real_images = self.get_image_files(self.real_folder)
        print(f"   Found {len(real_images)} real images")
        
        for img_path in real_images:
            relative_path = str(img_path.relative_to(self.dataset_root))
            labels.append((relative_path, 0, None))  # Real images don't have masks
        
        # Process hybrid images (label = 1)
        print("\n📁 Processing hybrid images...")
        hybrid_images = self.get_image_files(self.hybrid_folder)
        print(f"   Found {len(hybrid_images)} hybrid images")
        
        masks_found = 0
        for img_path in hybrid_images:
            relative_img_path = str(img_path.relative_to(self.dataset_root))
            
            # Find corresponding mask
            mask_path = self.find_mask_for_image(img_path)
            if mask_path:
                relative_mask_path = str(mask_path.relative_to(self.dataset_root))
                masks_found += 1
            else:
                relative_mask_path = None
            
            labels.append((relative_img_path, 1, relative_mask_path))
        
        if self.mask_folder:
            print(f"   Found masks for {masks_found}/{len(hybrid_images)} hybrid images")
        
        return labels
    
    def validate_dataset(self, labels: List[Tuple]) -> dict:
        """Validate dataset and return statistics"""
        stats = {
            'total_images': len(labels),
            'real_images': sum(1 for l in labels if l[1] == 0),
            'hybrid_images': sum(1 for l in labels if l[1] == 1),
            'images_with_masks': sum(1 for l in labels if l[2] is not None),
            'images_without_masks': sum(1 for l in labels if l[1] == 1 and l[2] is None),
        }
        
        # Check if images exist
        print("\n🔍 Validating dataset...")
        missing_images = []
        missing_masks = []
        
        for img_path, label, mask_path in labels:
            full_img_path = self.dataset_root / img_path
            if not full_img_path.exists():
                missing_images.append(img_path)
            
            if mask_path:
                full_mask_path = self.dataset_root / mask_path
                if not full_mask_path.exists():
                    missing_masks.append(mask_path)
        
        stats['missing_images'] = len(missing_images)
        stats['missing_masks'] = len(missing_masks)
        
        if missing_images:
            print(f"   ⚠️  Warning: {len(missing_images)} images not found!")
            print(f"      First few: {missing_images[:3]}")
        
        if missing_masks:
            print(f"   ⚠️  Warning: {len(missing_masks)} masks not found!")
            print(f"      First few: {missing_masks[:3]}")
        
        return stats
    
    def verify_masks(self, num_samples: int = 5):
        """Verify that masks are valid binary images"""
        print("\n🖼️  Verifying mask format...")
        
        hybrid_images = self.get_image_files(self.hybrid_folder)
        checked = 0
        
        for img_path in hybrid_images[:num_samples]:
            mask_path = self.find_mask_for_image(img_path)
            if mask_path:
                try:
                    mask = Image.open(mask_path)
                    mask_array = np.array(mask)
                    
                    print(f"\n   Mask: {mask_path.name}")
                    print(f"   - Shape: {mask_array.shape}")
                    print(f"   - Dtype: {mask_array.dtype}")
                    print(f"   - Unique values: {np.unique(mask_array)[:10]}")  # First 10 unique values
                    print(f"   - Min: {mask_array.min()}, Max: {mask_array.max()}")
                    
                    # Check if binary
                    unique_values = np.unique(mask_array)
                    if len(unique_values) <= 3:  # Binary or grayscale with few values
                        print(f"   ✅ Looks like a valid mask")
                    else:
                        print(f"   ⚠️  Mask has {len(unique_values)} unique values (expected 2-3)")
                    
                    checked += 1
                    
                except Exception as e:
                    print(f"   ❌ Error loading mask: {e}")
        
        if checked == 0:
            print("   ⚠️  No masks found to verify")
    
    
    def print_statistics(self, stats: dict):
        """Print dataset statistics"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        print(f"\n📊 Total Images: {stats['total_images']}")
        print(f"   ├─ Real images: {stats['real_images']} ({stats['real_images']/stats['total_images']*100:.1f}%)")
        print(f"   └─ Hybrid images: {stats['hybrid_images']} ({stats['hybrid_images']/stats['total_images']*100:.1f}%)")
        
        if stats['images_with_masks'] > 0:
            print(f"\n🎭 Masks:")
            print(f"   ├─ Images with masks: {stats['images_with_masks']}")
            print(f"   └─ Images without masks: {stats['images_without_masks']}")
        
        if stats['missing_images'] > 0 or stats['missing_masks'] > 0:
            print(f"\n⚠️  Issues:")
            if stats['missing_images'] > 0:
                print(f"   ├─ Missing images: {stats['missing_images']}")
            if stats['missing_masks'] > 0:
                print(f"   └─ Missing masks: {stats['missing_masks']}")
        else:
            print(f"\n✅ All files validated successfully!")
        
        # Class balance
        balance_ratio = stats['real_images'] / stats['hybrid_images'] if stats['hybrid_images'] > 0 else 0
        print(f"\n⚖️  Class Balance Ratio (Real:Hybrid): {balance_ratio:.2f}:1")
        if 0.8 <= balance_ratio <= 1.2:
            print(f"   ✅ Good balance!")
        elif 0.5 <= balance_ratio <= 2.0:
            print(f"   ⚠️  Slight imbalance (acceptable)")
        else:
            print(f"   ⚠️  Significant imbalance - consider balancing")
        
        print("=" * 70 + "\n")


def prepare_dataset(dataset_root: str,
                   real_folder: str = 'real',
                   hybrid_folder: str = 'hybrid',
                   mask_folder: str = 'masks',
                   verify_masks: bool = True) -> List[Tuple]:
    """
    Main function to prepare dataset
    
    Args:
        dataset_root: Root directory with your data
        real_folder: Name of folder with real images
        hybrid_folder: Name of folder with hybrid/fake images  
        mask_folder: Name of folder with masks
        verify_masks: Whether to verify mask format
    
    Returns:
        List of tuples: [(image_path, label, mask_path), ...]
    """
    print("=" * 70)
    print("HYBRID IMAGE DATASET PREPARATION")
    print("=" * 70)
    print(f"\n📂 Dataset root: {dataset_root}")
    print(f"   ├─ Real images folder: {real_folder}/")
    print(f"   ├─ Hybrid images folder: {hybrid_folder}/")
    print(f"   └─ Masks folder: {mask_folder}/")
    
    # Initialize preparer
    preparer = DataPreparer(
        dataset_root=dataset_root,
        real_folder=real_folder,
        hybrid_folder=hybrid_folder,
        mask_folder=mask_folder
    )
    
    # Create labels
    labels = preparer.create_labels_list()
    
    # Validate dataset
    stats = preparer.validate_dataset(labels)
    
    # Print statistics
    preparer.print_statistics(stats)
    
    # Verify masks
    if verify_masks and preparer.mask_folder:
        preparer.verify_masks(num_samples=5)
    
    
    
    print("\n✅ Dataset preparation complete!")
    print(f"✅ Ready to use with {len(labels)} images")
    
    return labels


if __name__ == "__main__":
    
    # CHANGE THIS to your actual dataset path
    DATASET_ROOT = '/kaggle/input/hybrid-dataset/hybrid-dataset'
    
    # If your folders have different names, change these:
    REAL_FOLDER = 'real'        # or 'authentic', 'original', etc.
    HYBRID_FOLDER = 'hybrid'    # or 'fake', 'edited', 'manipulated', etc.
    MASK_FOLDER = 'mask'       # or 'ground_truth', 'annotations', etc.
    
    # Prepare the dataset
    labels = prepare_dataset(
        dataset_root=DATASET_ROOT,
        real_folder=REAL_FOLDER,
        hybrid_folder=HYBRID_FOLDER,
        mask_folder=MASK_FOLDER,
        verify_masks=True  # Set False to skip mask verification
    )
    
