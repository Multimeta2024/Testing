# ============================================================================
# Run this file to train. All config is here.
# ============================================================================

from train_simple import train_from_folders
import torch

print("\nStarting hybrid image detection training...")

history, model_path = train_from_folders(
    dataset_root='/kaggle/input/hybrid-dataset/hybrid-dataset',
    real_folder='real',
    hybrid_folder='hybrid',
    mask_folder=None,

    use_lite_model=True,
    use_localization=False,

    num_epochs=20,
    batch_size=16,
    learning_rate=3e-4,

    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"\n✅ Done. Model saved to: {model_path}/best.pth")
