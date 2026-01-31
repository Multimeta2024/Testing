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
    Lightweight version for faster training/inference
    Recommended for initial experiments with 5k dataset
    """
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Use smaller backbone
        self.rgb_encoder = timm.create_model('efficientnet_b0', pretrained=True)
        self.rgb_encoder.classifier = nn.Identity()
        rgb_dim = 1280
        
        # Frequency branch
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        freq_dim = 64
        
        # Edge detection
        self.edge_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        edge_dim = 32
        
        # Classifier
        total_dim = rgb_dim + freq_dim + edge_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, rgb, freq_map):
        rgb_feat = self.rgb_encoder(rgb)
        freq_feat = self.freq_branch(freq_map).flatten(1)
        edge_feat = self.edge_branch(rgb).flatten(1)
        
        combined = torch.cat([rgb_feat, freq_feat, edge_feat], dim=1)
        classification = self.classifier(combined)
        
        return classification, None  # No localization in lite version


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Full model
    model = HybridImageDetector().to(device)
    rgb = torch.randn(2, 3, 224, 224).to(device)
    freq = torch.randn(2, 1, 224, 224).to(device)
    
    cls, loc = model(rgb, freq)
    print(f"Classification output: {cls.shape}")  # [2, 1]
    print(f"Localization output: {loc.shape}")    # [2, 1, 28, 28]
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Lite model
    print("\n--- Lite Model ---")
    lite_model = HybridDetectorLite().to(device)
    cls_lite, _ = lite_model(rgb, freq)
    print(f"Classification output: {cls_lite.shape}")
    
    lite_params = sum(p.numel() for p in lite_model.parameters())
    print(f"Lite model parameters: {lite_params:,}")
