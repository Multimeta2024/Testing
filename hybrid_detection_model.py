"""
Hybrid Image Detection Model
Detects AI-edited regions within real images (e.g., Gemini Nano edits)
Optimized for limited dataset (5k images) with strong generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class SpatialAttention(nn.Module):
    """Focuses on regions with potential manipulation"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class EdgeDetectionBranch(nn.Module):
    """Detects boundary inconsistencies from AI blending"""
    def __init__(self):
        super().__init__()
        
        # Sobel-inspired learnable edge filters
        self.edge_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        
        self.edge_processor = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        edges = self.edge_conv(x)
        features = self.edge_processor(edges)
        return features.flatten(1)


class FrequencyBranch(nn.Module):
    """Analyzes frequency domain for AI artifacts"""
    def __init__(self):
        super().__init__()
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, freq_map):
        return self.freq_encoder(freq_map).flatten(1)


class MultiScaleRGBBranch(nn.Module):
    """Extracts features at multiple scales for better localization"""
    def __init__(self, backbone='efficientnet_b4'):
        super().__init__()
        
        # Use EfficientNet-B4 as backbone
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        # Returns features at different scales: [24, 32, 56, 160, 448] channels
        
        # Spatial attention at each scale
        self.attentions = nn.ModuleList([
            SpatialAttention(24),
            SpatialAttention(32),
            SpatialAttention(56),
            SpatialAttention(160),
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(24 + 32 + 56 + 160, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Apply attention to intermediate features
        attended = []
        for i in range(4):
            att_feat = self.attentions[i](features[i])
            # Resize to common spatial size
            att_feat = F.interpolate(att_feat, size=(28, 28), mode='bilinear', align_corners=False)
            attended.append(att_feat)
        
        # Concatenate multi-scale features
        fused = torch.cat(attended, dim=1)
        fused = self.fusion(fused)
        
        return fused, features[-1]  # Return both fused features and final layer


class HybridImageDetector(nn.Module):
    """
    Complete hybrid image detection model with:
    1. Binary classification (real vs hybrid)
    2. Localization map (WHERE is the AI edit?)
    3. Multi-stream fusion (RGB + Frequency + Edge)
    """
    def __init__(self, num_classes=1, backbone='efficientnet_b4'):
        super().__init__()
        
        # Multi-stream feature extractors
        self.rgb_branch = MultiScaleRGBBranch(backbone)
        self.freq_branch = FrequencyBranch()
        self.edge_branch = EdgeDetectionBranch()
        
        # Feature dimensions
        rgb_multiscale_dim = 256  # from fusion
        # Query the backbone for actual final feature channel count
        # (features_only=True returns feature maps, NOT the 1792-dim classifier input)
        rgb_final_dim = self.rgb_branch.backbone.feature_info.channels()[-1]  # 448 for efficientnet_b4
        freq_dim = 128
        edge_dim = 64
        
        # Global pooling for final RGB features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Combined feature classifier
        total_dim = rgb_final_dim + freq_dim + edge_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Localization head (segmentation)
        self.localization = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),  # Single channel output
            nn.Sigmoid()
        )
    
    def forward(self, rgb, freq_map):
        """
        Args:
            rgb: RGB image tensor [B, 3, 224, 224]
            freq_map: FFT magnitude [B, 1, 224, 224]
        
        Returns:
            classification_logits: [B, 1] - real vs hybrid
            localization_map: [B, 1, 28, 28] - manipulation mask
        """
        # Extract features from all branches
        rgb_multiscale, rgb_final = self.rgb_branch(rgb)
        freq_features = self.freq_branch(freq_map)
        edge_features = self.edge_branch(rgb)
        
        # Pool final RGB features
        rgb_pooled = self.global_pool(rgb_final).flatten(1)
        
        # Concatenate all features for classification
        combined = torch.cat([rgb_pooled, freq_features, edge_features], dim=1)
        classification = self.classifier(combined)
        
        # Generate localization map from multi-scale features
        localization = self.localization(rgb_multiscale)
        
        return classification, localization


class HybridDetectorLite(nn.Module):
    """
    Lightweight detector for paired-image hybrid detection.
    
    Key design decisions:
    - features_only=True on EfficientNet so we keep the [B, C, 7, 7] spatial map
      instead of collapsing to a single vector. Local edits need spatial info.
    - rgb_channels is queried from feature_info at runtime — never hardcoded,
      because it differs between timm versions (448 vs 1280 etc).
    - Freq and edge branches output [B, 64, 7, 7] to match RGB spatial size.
    - Classifier is 1x1 convs on the fused spatial map, pools only at the end.
      This lets the model learn "is there ANY anomalous region?" rather than
      trying to classify a single global vector.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        
        # RGB backbone — keep spatial feature map, query actual channel count
        self.rgb_encoder = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            features_only=True,
            out_indices=(4,)          # last stage only
        )
        # This is the only reliable way to get the channel count —
        # it reads it from the model itself, not from a hardcoded number
        rgb_channels = self.rgb_encoder.feature_info.channels()[-1]
        
        # Frequency branch: [B, 1, 224, 224] -> [B, 64, 7, 7]
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(28),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(7),
        )
        freq_channels = 64
        
        # Edge branch: [B, 3, 224, 224] -> [B, 64, 7, 7]
        self.edge_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(28),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(7),
        )
        edge_channels = 64
        
        # Spatial classifier on fused [B, total, 7, 7] map
        total_channels = rgb_channels + freq_channels + edge_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveMaxPool2d(1),   # pool only here, at the very end
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, rgb, freq_map, edge_map):
        """
        Args:
            rgb:       [B, 3, 224, 224]
            freq_map:  [B, 1, 224, 224]  patch-wise FFT
            edge_map:  [B, 3, 224, 224]  Sobel/Laplacian channels
        Returns:
            logits: [B, 1]
            None
        """
        rgb_feat  = self.rgb_encoder(rgb)[0]        # [B, C, 7, 7]
        freq_feat = self.freq_branch(freq_map)      # [B, 64, 7, 7]
        edge_feat = self.edge_branch(edge_map)      # [B, 64, 7, 7]
        
        fused = torch.cat([rgb_feat, freq_feat, edge_feat], dim=1)
        # 1️⃣ Spatial map BEFORE pooling (keep 7x7)
        spatial_logits = self.classifier[:-3](fused)   # stops BEFORE AdaptiveMaxPool2d
        # 2️⃣ Final decision (any-patch logic)
        classification = self.classifier[-3:](spatial_logits)

        patch_score, texture_score, energy_score = self.spatial_scores(spatial_logits)

        return classification, {"patch": patch_score,"texture": texture_score,"energy": energy_score}
    
    def spatial_scores(self, spatial_map):
        """
        spatial_map: [B, C, H, W]
        returns 3 scalar scores per image
        """

        # 1️⃣ Patch anomaly score (max activation)
        patch_score = spatial_map.view(spatial_map.size(0), -1).max(dim=1)[0]

        # 2️⃣ Texture inconsistency score (std of activations)
        texture_score = spatial_map.view(spatial_map.size(0), -1).std(dim=1)

        # 3️⃣ Energy score (mean absolute activation)
        energy_score = spatial_map.abs().mean(dim=[1, 2, 3])

        return patch_score, texture_score, energy_score




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("--- Lite Model ---")
    lite_model = HybridDetectorLite().to(device)
    
    rgb  = torch.randn(2, 3, 224, 224).to(device)
    freq = torch.randn(2, 1, 224, 224).to(device)
    edge = torch.randn(2, 3, 224, 224).to(device)
    
    cls, _ = lite_model(rgb, freq, edge)
    print(f"Output shape: {cls.shape}")
    
    total = sum(p.numel() for p in lite_model.parameters())
    print(f"Parameters: {total:,}")
