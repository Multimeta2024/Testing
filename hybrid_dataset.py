"""
Dataset handler for hybrid images with smart augmentation
Designed for PAIRED IMAGE detection: same base image, small local edit.

KEY FIXES vs previous version:
1. RGB and FFT are now computed from the SAME transformed image (were misaligned before)
2. Augmentations are gentle - no RandomCrop/Rotation/ColorJitter that destroy edit artifacts
3. FFT is computed per-patch (7x7 grid) so local edits aren't diluted by global averaging
4. Edge map uses fixed Sobel/Laplacian filters to highlight boundary artifacts
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import PIL
import torch.nn.functional as F
import random


class HybridImageDataset(Dataset):
    """
    Dataset for hybrid image detection where real and hybrid are the SAME base image
    with a small local edit (e.g. object added/removed in one region).
    """
    def __init__(self, 
                 img_dir, 
                 labels,
                 img_size=224,
                 mode='train',
                 use_augmentation=True):
        self.img_dir = img_dir
        self.labels = labels
        self.img_size = img_size
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == 'train')
        
        # Resize only - augmentation is handled manually below so we can
        # keep RGB, FFT, and edge map in sync
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # -> [0, 1] float
        ])
        
        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        # Fixed Sobel/Laplacian kernels (not learned - stable signal)
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        laplacian = torch.tensor([[ 0,  1,  0],
                                  [ 1, -4,  1],
                                  [ 0,  1,  0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_x = sobel_x
        self.sobel_y = sobel_y
        self.laplacian = laplacian

    def compute_edge_map(self, img_tensor):
        """
        Compute edge map using fixed Sobel + Laplacian filters.
        AI-generated regions have characteristic boundary artifacts.
        
        Args:
            img_tensor: [3, H, W] float tensor in [0, 1]
        Returns:
            edge_map: [3, H, W] - sobel magnitude, laplacian, combined
        """
        gray = (0.299 * img_tensor[0:1] + 0.587 * img_tensor[1:2] + 0.114 * img_tensor[2:3]).unsqueeze(0)
        
        sx = F.conv2d(gray, self.sobel_x, padding=1)
        sy = F.conv2d(gray, self.sobel_y, padding=1)
        lap = F.conv2d(gray, self.laplacian, padding=1)
        
        sobel_mag = torch.sqrt(sx**2 + sy**2 + 1e-8)
        sobel_mag = sobel_mag / (sobel_mag.amax() + 1e-8)
        
        lap = lap / (lap.abs().amax() + 1e-8)
        lap = (lap + 1) / 2
        
        combined = (sobel_mag + lap) / 2
        
        edge_map = torch.cat([sobel_mag, lap, combined], dim=1).squeeze(0)  # [3, H, W]
        return edge_map

    def compute_patch_fft(self, img_tensor):
        """
        Compute FFT on a 7x7 grid of local patches.
        Whole-image FFT dilutes small edits; patch FFT preserves local frequency anomalies.
        
        Args:
            img_tensor: [3, H, W] float tensor in [0, 1]
        Returns:
            patch_fft: [1, H, W]
        """
        gray = (0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]).numpy()
        H, W = gray.shape
        
        grid = 7
        ph = H // grid
        pw = W // grid
        fft_map = np.zeros((H, W), dtype=np.float32)
        
        for i in range(grid):
            for j in range(grid):
                y1 = i * ph
                y2 = (i + 1) * ph if i < grid - 1 else H
                x1 = j * pw
                x2 = (j + 1) * pw if j < grid - 1 else W
                
                patch = gray[y1:y2, x1:x2]
                fft = np.fft.fft2(patch)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.log(np.abs(fft_shift) + 1e-8)
                
                mag_min, mag_max = magnitude.min(), magnitude.max()
                if mag_max - mag_min > 1e-8:
                    magnitude = (magnitude - mag_min) / (mag_max - mag_min)
                else:
                    magnitude = np.zeros_like(magnitude)
                
                fft_map[y1:y2, x1:x2] = magnitude
        
        return torch.tensor(fft_map, dtype=torch.float32).unsqueeze(0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if len(self.labels[idx]) == 3:
            img_path, label, mask_path = self.labels[idx]
        else:
            img_path, label = self.labels[idx]
            mask_path = None
        
        full_path = os.path.join(self.img_dir, img_path)
        try:
            img = Image.open(full_path).convert("RGB")
        except (PIL.UnidentifiedImageError, OSError, IOError):
            print(f"⚠️  Warning: Could not load {img_path}, using black placeholder")
            img = Image.new('RGB', (224, 224), color='black')
        
        # Step 1: Resize to standard size (only non-destructive transform)
        raw_tensor = self.base_transform(img)  # [3, 224, 224] in [0, 1]
        
        # Step 2: Compute edge map and patch FFT from the SAME raw tensor
        edge_map = self.compute_edge_map(raw_tensor)   # [3, 224, 224]
        freq = self.compute_patch_fft(raw_tensor)      # [1, 224, 224]
        
        # Step 3: Augmentation - apply the SAME flip to ALL three inputs
        if self.use_augmentation and random.random() < 0.5:
            raw_tensor = raw_tensor.flip(-1)
            edge_map = edge_map.flip(-1)
            freq = freq.flip(-1)
        
        # Step 4: Normalize RGB for the backbone
        rgb = raw_tensor.clone()
        rgb[0] = (rgb[0] - 0.485) / 0.229
        rgb[1] = (rgb[1] - 0.456) / 0.224
        rgb[2] = (rgb[2] - 0.406) / 0.225
        
        # Step 5: Load mask
        if mask_path is not None:
            mask_full_path = os.path.join(self.img_dir, mask_path)
            if os.path.exists(mask_full_path):
                mask = Image.open(mask_full_path).convert("L")
                mask_array = np.array(mask)
                mask_array = 255 - mask_array
                mask = Image.fromarray(mask_array)
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros(1, 28, 28)
        else:
            if label == 1:
                mask = torch.ones(1, 28, 28)
            else:
                mask = torch.zeros(1, 28, 28)
        
        return rgb, freq, edge_map, torch.tensor(label, dtype=torch.float32), mask


class SyntheticHybridDataset(Dataset):
    """Kept for compatibility but not used in current training"""
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        pass
