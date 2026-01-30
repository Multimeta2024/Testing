"""
Configuration file for hybrid image detection training
Modify these settings based on your dataset and hardware
"""

import torch

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    # Path to your image directory
    'img_dir': 'data/images',
    
    # Labels format: [(img_path, label, mask_path), ...]
    # label: 0=real, 1=hybrid
    # mask_path: optional, None if not available
    'labels_file': 'data/labels.txt',  # Or load from CSV/JSON
    
    # Data split
    'val_split': 0.2,      # 20% for validation
    'test_split': 0.1,     # 10% for testing (optional)
    
    # Image preprocessing
    'img_size': 224,       # Input image size (224 or 256)
    'use_augmentation': True,
    
    # Synthetic data (optional)
    'use_synthetic': False,
    'synthetic_ratio': 0.3,  # 30% synthetic in training
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Model architecture
    'model_type': 'full',  # 'full' or 'lite'
    'backbone': 'efficientnet_b4',  # or 'efficientnet_b0' for lite
    
    # Multi-task learning
    'use_localization': True,  # Predict WHERE edits are
    'localization_weight': 0.5,  # Loss weight for localization
    
    # Pretrained weights
    'use_pretrained': True,
    'freeze_backbone_epochs': 0,  # Freeze backbone for first N epochs
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Basic settings
    'batch_size': 16,       # Reduce if out of memory
    'num_epochs': 50,
    'num_workers': 4,       # Data loading workers
    
    # Optimization
    'learning_rate': 3e-4,
    'backbone_lr_ratio': 0.1,  # Backbone LR = lr * 0.1
    'weight_decay': 0.01,
    'optimizer': 'adamw',   # 'adamw' or 'adam'
    
    # Learning rate scheduling
    'scheduler': 'cosine',  # 'cosine', 'step', or 'plateau'
    'lr_warmup_epochs': 5,
    'min_lr': 1e-6,
    
    # Loss function
    'loss_type': 'focal',   # 'focal' or 'bce'
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'pos_weight': 2.0,      # For class imbalance
    
    # Regularization
    'dropout': 0.4,
    'use_mixup': False,     # Mixup augmentation
    'mixup_alpha': 0.2,
    
    # Mixed precision
    'use_amp': True,        # Automatic mixed precision
    
    # Early stopping
    'patience': 10,         # Stop if no improvement for N epochs
    'min_delta': 0.001,     # Minimum improvement threshold
    
    # Checkpointing
    'save_dir': 'checkpoints',
    'save_frequency': 5,    # Save every N epochs
}

# ============================================================================
# VALIDATION & TESTING
# ============================================================================

EVAL_CONFIG = {
    # Metrics
    'metrics': ['auc', 'f1', 'precision', 'recall', 'accuracy'],
    'threshold': 0.5,       # Decision threshold
    
    # Evaluation frequency
    'eval_every_n_epochs': 1,
    
    # Test-time augmentation
    'use_tta': False,       # Average predictions over augmentations
    'tta_transforms': 5,    # Number of augmented versions
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    # Model checkpoint
    'checkpoint_path': 'checkpoints/best.pth',
    
    # Prediction settings
    'threshold': 0.5,
    'return_localization': True,
    'return_confidence': True,
    
    # Batch processing
    'batch_size': 32,
    
    # Visualization
    'save_visualizations': True,
    'output_dir': 'results',
}

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

HARDWARE_CONFIG = {
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gpu_ids': [0],         # Multi-GPU: [0, 1, 2, 3]
    
    # Memory optimization
    'pin_memory': True,
    'persistent_workers': False,
}

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

LOGGING_CONFIG = {
    'use_tensorboard': False,
    'use_wandb': False,     # Weights & Biases
    'log_dir': 'logs',
    'experiment_name': 'hybrid_detection',
    
    # What to log
    'log_images': True,
    'log_frequency': 100,   # Log every N batches
}

# ============================================================================
# ADVANCED FEATURES
# ============================================================================

ADVANCED_CONFIG = {
    # K-Fold cross validation
    'use_kfold': False,
    'n_folds': 5,
    
    # Ensemble
    'use_ensemble': False,
    'ensemble_size': 3,
    
    # Active learning
    'active_learning': False,
    'uncertainty_threshold': 0.3,
    
    # Data augmentation library
    'use_albumentations': False,  # More advanced augmentation
}

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Quick start - Fast training on small dataset
QUICK_START = {
    **DATASET_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'model_type': 'lite',
    'batch_size': 32,
    'num_epochs': 20,
    'use_localization': False,
}

# Production - Best performance
PRODUCTION = {
    **DATASET_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'model_type': 'full',
    'batch_size': 16,
    'num_epochs': 100,
    'use_localization': True,
    'use_amp': True,
}

# Debug - Fast iteration for development
DEBUG = {
    **DATASET_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'model_type': 'lite',
    'batch_size': 4,
    'num_epochs': 2,
    'num_workers': 0,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(preset='default'):
    """
    Get configuration based on preset
    
    Args:
        preset: 'default', 'quick_start', 'production', or 'debug'
    
    Returns:
        dict: Configuration dictionary
    """
    if preset == 'quick_start':
        return QUICK_START
    elif preset == 'production':
        return PRODUCTION
    elif preset == 'debug':
        return DEBUG
    else:
        return {
            **DATASET_CONFIG,
            **MODEL_CONFIG,
            **TRAINING_CONFIG,
            **EVAL_CONFIG,
            **INFERENCE_CONFIG,
            **HARDWARE_CONFIG,
            **LOGGING_CONFIG,
            **ADVANCED_CONFIG,
        }


def print_config(config):
    """Pretty print configuration"""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    for section, settings in [
        ("Dataset", DATASET_CONFIG),
        ("Model", MODEL_CONFIG),
        ("Training", TRAINING_CONFIG),
        ("Hardware", HARDWARE_CONFIG),
    ]:
        print(f"\n{section}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    # Example usage
    config = get_config('production')
    print_config(config)
    
    # Access specific settings
    print(f"\nBatch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Model type: {config['model_type']}")
