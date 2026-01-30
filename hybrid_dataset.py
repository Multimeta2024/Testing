"""
Dataset handler for hybrid images with smart augmentation
Designed to maximize learning from limited data (5k images)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import PIL
import torch.nn.functional as F
import random


class HybridImageDataset(Dataset):
    """
    Dataset for hybrid image detection
    Supports both classification and localization tasks
    """
    def __init__(self, 
                 img_dir, 
                 labels,  # List of tuples: (image_path, label, mask_path)
                 img_size=224,
                 mode='train',
                 use_augmentation=True):
        """
        Args:
            img_dir: Directory containing images
            labels: List of (img_path, label, mask_path) where:
                    - label: 0=real, 1=hybrid
                    - mask_path: Optional, path to manipulation mask (if available)
            img_size: Input image size
            mode: 'train', 'val', or 'test'
            use_augmentation: Whether to apply data augmentation
        """
        self.img_dir = img_dir
        self.labels = labels
        self.img_size = img_size
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == 'train')
        
        # RGB transformations
        if self.use_augmentation:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),  # Larger for random crop
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),  # Sometimes useful
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                # Random JPEG compression (AI edits might have compression artifacts)
                transforms.RandomApply([
                    transforms.Lambda(lambda x: self._jpeg_compression(x))
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Mask transform (if available)
        self.mask_transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Match localization output size
            transforms.ToTensor()
        ])
    
    def _jpeg_compression(self, img, quality=None):
        """Simulate JPEG compression artifacts"""
        if quality is None:
            quality = random.randint(60, 95)
        
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    def fft_transform(self, img):
        """
        Compute FFT magnitude spectrum
        AI edits often introduce frequency domain artifacts
        """
        # Convert to grayscale
        img_gray = np.array(img.convert("L"), dtype=np.float32)
        
        # Apply 2D FFT
        fft = np.fft.fft2(img_gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Magnitude spectrum in log scale
        magnitude = np.log(np.abs(fft_shift) + 1e-8)
        
        # Normalize to [0, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Convert to tensor
        freq_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        
        # Resize to match input size
        freq_tensor = F.interpolate(
            freq_tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return freq_tensor
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Parse label info
        if len(self.labels[idx]) == 3:
            img_path, label, mask_path = self.labels[idx]
        else:
            img_path, label = self.labels[idx]
            mask_path = None
        
        # Load image with error handling
        full_path = os.path.join(self.img_dir, img_path)
        
        try:
            img = Image.open(full_path).convert("RGB")
        except (PIL.UnidentifiedImageError, OSError, IOError) as e:
            # If image is corrupted, try next one or return a black image
            print(f"⚠️  Warning: Could not load {img_path}, using black placeholder")
            img = Image.new('RGB', (224, 224), color='black')
        
        # Store original for FFT before augmentation
        img_original = img.copy()
        
        # Apply RGB transformations
        rgb = self.rgb_transform(img)
        
        # Compute FFT from original (before augmentation)
        freq = self.fft_transform(img_original)
        
        # Load mask if available
        if mask_path is not None:
            mask_full_path = os.path.join(self.img_dir, mask_path)
            if os.path.exists(mask_full_path):
                mask = Image.open(mask_full_path).convert("L")
                mask = self.mask_transform(mask)
            else:
                # Create dummy mask if file doesn't exist
                mask = torch.zeros(1, 28, 28)
        else:
            # Create dummy mask (all zeros for real, all ones for hybrid if no mask provided)
            if label == 1:
                mask = torch.ones(1, 28, 28)  # Assume entire image is hybrid
            else:
                mask = torch.zeros(1, 28, 28)
        
        return rgb, freq, torch.tensor(label, dtype=torch.float32), mask


class SyntheticHybridDataset(Dataset):
    """
    Generate synthetic hybrid images on-the-fly for data augmentation
    This helps when you have limited real hybrid data
    """
    def __init__(self, 
                 real_img_dir,
                 real_image_list,
                 img_size=224,
                 hybrid_ratio=0.5):
        """
        Args:
            real_img_dir: Directory with real images
            real_image_list: List of real image paths
            img_size: Image size
            hybrid_ratio: Probability of creating a hybrid (vs returning real)
        """
        self.img_dir = real_img_dir
        self.img_list = real_image_list
        self.img_size = img_size
        self.hybrid_ratio = hybrid_ratio
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def create_synthetic_hybrid(self, img):
        """
        Create synthetic hybrid by simple copy-paste or local modifications
        This is NOT perfect but helps regularization
        """
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Random rectangular region
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(x1 + w // 4, w)
        y2 = random.randint(y1 + h // 4, h)
        
        # Apply simple modification (blur, color shift, etc.)
        manipulation_type = random.choice(['blur', 'color_shift', 'noise'])
        
        if manipulation_type == 'blur':
            region = Image.fromarray(img_array[y1:y2, x1:x2])
            blurred = region.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))
            img_array[y1:y2, x1:x2] = np.array(blurred)
        
        elif manipulation_type == 'color_shift':
            img_array[y1:y2, x1:x2] = np.clip(
                img_array[y1:y2, x1:x2] * random.uniform(0.7, 1.3),
                0, 255
            ).astype(np.uint8)
        
        elif manipulation_type == 'noise':
            noise = np.random.normal(0, 15, img_array[y1:y2, x1:x2].shape)
            img_array[y1:y2, x1:x2] = np.clip(
                img_array[y1:y2, x1:x2] + noise,
                0, 255
            ).astype(np.uint8)
        
        # Update mask
        mask[y1:y2, x1:x2] = 1.0
        
        return Image.fromarray(img_array), mask
    
    def fft_transform(self, img):
        """Same as HybridImageDataset"""
        img_gray = np.array(img.convert("L"), dtype=np.float32)
        fft = np.fft.fft2(img_gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1e-8)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        freq_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        freq_tensor = F.interpolate(
            freq_tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return freq_tensor
    
    def __len__(self):
        return len(self.img_list) * 2  # Can generate many synthetic variants
    
    def __getitem__(self, idx):
        # Get base image
        img_idx = idx % len(self.img_list)
        img_path = os.path.join(self.img_dir, self.img_list[img_idx])
        img = Image.open(img_path).convert("RGB")
        
        # Decide if creating hybrid or using real
        if random.random() < self.hybrid_ratio:
            # Create synthetic hybrid
            img, mask_np = self.create_synthetic_hybrid(img)
            label = 1.0
            
            # Resize mask
            mask = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask = transforms.Resize((28, 28))(mask)
            mask = transforms.ToTensor()(mask)
        else:
            # Use real image
            label = 0.0
            mask = torch.zeros(1, 28, 28)
        
        # Transform
        rgb = self.transform(img)
        freq = self.fft_transform(img)
        
        return rgb, freq, torch.tensor(label, dtype=torch.float32), mask


if __name__ == "__main__":
    # Example usage
    print("Testing HybridImageDataset...")
    
    # Example label format
    train_labels = [
        ("real_img1.jpg", 0, None),  # Real image
        ("hybrid_img1.jpg", 1, "hybrid_img1_mask.png"),  # Hybrid with mask
        ("hybrid_img2.jpg", 1, None),  # Hybrid without mask
    ]
    
    # dataset = HybridImageDataset(
    #     img_dir="path/to/images",
    #     labels=train_labels,
    #     mode='train'
    # )
    
    # rgb, freq, label, mask = dataset[0]
    # print(f"RGB shape: {rgb.shape}")
    # print(f"Freq shape: {freq.shape}")
    # print(f"Label: {label}")
    # print(f"Mask shape: {mask.shape}")
    
    print("\nDataset classes ready!")
    print("Use HybridImageDataset for your 5k real hybrid images")
    print("Use SyntheticHybridDataset to augment with synthetic data")
