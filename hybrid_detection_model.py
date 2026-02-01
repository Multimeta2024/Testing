"""
Hybrid Image Detection Model - REWRITTEN for paired-image detection.

The key insight: real and hybrid images are the SAME base image with a small
local edit. Global features (whole-image classification) can't detect this.
The model needs to find LOCAL anomalies — boundary artifacts, frequency
inconsistencies, and texture mismatches in the edited region.

Architecture (HybridDetectorLite):
    Input:  RGB [B,3,224,224] | Patch-FFT [B,1,224,224] | Edge map [B,3,224,224]
    
    1. RGB backbone (EfficientNet-B0) → spatial feature map [B,1280,7,7]
         - We KEEP the spatial map. Previous version global-pooled it to [B,1280],
           which destroyed all spatial information.
    
    2. Frequency branch: Conv stack on patch-FFT → [B,64,7,7] spatial map
    
    3. Edge branch: Conv stack on edge map → [B,64,7,7] spatial map
    
    4. Fusion: Concatenate all three spatial maps → [B,1408,7,7]
    
    5. Classifier: Small conv layers on the fused spatial map, then adaptive
       average pool → single logit. This lets the model learn "is there ANY
       anomalous region in this image?" rather than trying to classify the
       whole image at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HybridDetectorLite(nn.Module):
    """
    Lightweight detector for paired-image hybrid detection.
    Preserves spatial information throughout to detect local edits.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        
        # RGB backbone - extract spatial features, DON'T global pool
        self.rgb_encoder = timm.create_model(
            'efficientnet_b0', 
            pretrained=True, 
            features_only=True,   # Return feature maps at each stage
            out_indices=(4,)      # Only the last stage: [B, 1280, 7, 7]
        )
        rgb_channels = 1280
        
        # Frequency branch: processes [B, 1, 224, 224] patch-FFT map
        # Output must be [B, 64, 7, 7] to match RGB spatial size
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(28),       # 224 -> 28

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(7),        # 28 -> 7
        )
        freq_channels = 64
        
        # Edge branch: processes [B, 3, 224, 224] Sobel/Laplacian edge map
        # Output: [B, 64, 7, 7]
        self.edge_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(28),       # 224 -> 28

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(7),        # 28 -> 7
        )
        edge_channels = 64
        
        # Spatial classifier: works on the fused [B, 1408, 7, 7] feature map.
        # Uses 1x1 convs to mix channels, then pools spatially.
        # This lets the model learn "does ANY 7x7 region look anomalous?"
        total_channels = rgb_channels + freq_channels + edge_channels  # 1408
        
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1),   # 1x1 conv: mix channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),                   # spatial dropout
            
            nn.Conv2d(256, 64, 1),               # compress further
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),             # [B, 64, 1, 1] — NOW pool
            nn.Flatten(),                        # [B, 64]
            nn.Linear(64, num_classes)           # final logit
        )
    
    def forward(self, rgb, freq_map, edge_map):
        """
        Args:
            rgb:       [B, 3, 224, 224] - normalized RGB image
            freq_map:  [B, 1, 224, 224] - patch-wise FFT magnitude
            edge_map:  [B, 3, 224, 224] - Sobel/Laplacian edge channels
        
        Returns:
            classification: [B, 1] logit
            None:           (placeholder for API compatibility)
        """
        # RGB spatial features
        rgb_feat = self.rgb_encoder(rgb)[0]      # [B, 1280, 7, 7]
        
        # Frequency spatial features
        freq_feat = self.freq_branch(freq_map)   # [B, 64, 7, 7]
        
        # Edge spatial features
        edge_feat = self.edge_branch(edge_map)   # [B, 64, 7, 7]
        
        # Fuse all three spatial maps
        fused = torch.cat([rgb_feat, freq_feat, edge_feat], dim=1)  # [B, 1408, 7, 7]
        
        # Classify from fused spatial map
        classification = self.classifier(fused)  # [B, 1]
        
        return classification, None


class HybridImageDetector(nn.Module):
    """
    Full model — kept for API compatibility.
    For now just wraps the lite model. Can be expanded later.
    """
    def __init__(self, num_classes=1, backbone='efficientnet_b4'):
        super().__init__()
        self.model = HybridDetectorLite(num_classes=num_classes)
    
    def forward(self, rgb, freq_map, edge_map):
        return self.model(rgb, freq_map, edge_map)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = HybridDetectorLite().to(device)
    
    rgb = torch.randn(2, 3, 224, 224).to(device)
    freq = torch.randn(2, 1, 224, 224).to(device)
    edge = torch.randn(2, 3, 224, 224).to(device)
    
    cls, _ = model(rgb, freq, edge)
    print(f"Output shape: {cls.shape}")  # [2, 1]
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
