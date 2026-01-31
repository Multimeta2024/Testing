# ============================================================================
# STEP 1: PREPARE YOUR DATA
# ============================================================================

from prepare_data import prepare_dataset

# CHANGE THIS TO YOUR DATASET PATH
DATASET_PATH = '/kaggle/input/hybrid-dataset/hybrid-dataset'

# ============================================================================
# STEP 1: TRAIN THE MODEL
# ============================================================================

from train_simple import train_from_folders
import torch

print("\nStep 2: Training model...")

# Quick training (lite model, fewer epochs)
history, model_path = train_from_folders(
    dataset_root=DATASET_PATH,
    real_folder='real',
    hybrid_folder='hybrid',
    mask_folder='mask',
    
    # Quick settings (change for production)
    use_lite_model=True,      # Fast training
    use_localization=False,
    num_epochs=20,            # Quick test
    batch_size=16,
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"\n✅ Model trained and saved to: {model_path}/best.pth")


# ============================================================================
# DONE!
# ============================================================================

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)
print("\nWhat you can do next:")
print("1. Check the training results in the checkpoints folder")
print("2. Test on more images using inference.py")
print("3. Evaluate on a test set using evaluate.py")
print("4. For better performance, train the full model with more epochs")
print("="*60)


# ============================================================================
# FOR PRODUCTION USE (Better Performance)
# ============================================================================

"""
Once you're happy with the quick test, train the full model:

history, model_path = train_from_folders(
    dataset_root=DATASET_PATH,
    real_folder='real',
    hybrid_folder='hybrid',
    mask_folder='masks',
    
    # Production settings
    use_lite_model=False,     # Full model
    use_localization=True,    # Use masks
    num_epochs=50,            # More epochs
    batch_size=16,
    learning_rate=3e-4,
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

Expected performance:
- Lite model (20 epochs): 78-83% AUC
- Full model (50 epochs): 85-92% AUC
"""
