"""
Inference script for hybrid image detection
Supports single image, batch processing, and visualization
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple, Optional

from hybrid_detection_model import HybridImageDetector, HybridDetectorLite


class HybridImagePredictor:
    """Inference class for hybrid image detection"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda',
                 img_size: int = 224,
                 use_lite_model: bool = False):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            img_size: Input image size
            use_lite_model: Whether to use lite model
        """
        self.device = device
        self.img_size = img_size
        
        # Load model
        if use_lite_model:
            self.model = HybridDetectorLite().to(device)
        else:
            self.model = HybridImageDetector().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Best validation AUC: {checkpoint['metrics']['auc']:.4f}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def compute_fft(self, img: Image.Image) -> torch.Tensor:
        """Compute FFT magnitude spectrum"""
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
    
    @torch.no_grad()
    def predict(self, 
                img_path: str,
                return_localization: bool = True) -> dict:
        """
        Predict if an image is hybrid
        
        Args:
            img_path: Path to image
            return_localization: Whether to return manipulation localization map
        
        Returns:
            dict with:
                - 'probability': float, probability of being hybrid (0-1)
                - 'prediction': str, 'Real' or 'Hybrid'
                - 'confidence': float, confidence score
                - 'localization_map': np.ndarray, manipulation heatmap (if available)
        """
        # Load image
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        
        # Preprocess
        rgb = self.transform(img).unsqueeze(0).to(self.device)
        freq = self.compute_fft(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        cls_logits, loc_map = self.model(rgb, freq)
        
        # Get probability
        probability = torch.sigmoid(cls_logits).item()
        prediction = 'Hybrid' if probability > 0.5 else 'Real'
        confidence = probability if probability > 0.5 else (1 - probability)
        
        result = {
            'probability': probability,
            'prediction': prediction,
            'confidence': confidence,
            'image_path': img_path
        }
        
        # Add localization if available
        if return_localization and loc_map is not None:
            loc_map_np = loc_map.squeeze().cpu().numpy()
            # Resize to original image size
            loc_map_pil = Image.fromarray((loc_map_np * 255).astype(np.uint8))
            loc_map_resized = loc_map_pil.resize(original_size, Image.BILINEAR)
            result['localization_map'] = np.array(loc_map_resized) / 255.0
        
        return result
    
    def predict_batch(self, img_paths: list) -> list:
        """Predict for multiple images"""
        results = []
        for img_path in img_paths:
            result = self.predict(img_path)
            results.append(result)
        return results
    
    def visualize_prediction(self, 
                           img_path: str,
                           save_path: Optional[str] = None,
                           show: bool = True):
        """
        Visualize prediction with localization heatmap
        
        Args:
            img_path: Path to image
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        # Get prediction
        result = self.predict(img_path, return_localization=True)
        
        # Load original image
        img = Image.open(img_path).convert('RGB')
        
        # Create visualization
        if 'localization_map' in result:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = [axes[0], axes[1], None]
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction info
        pred_text = f"Prediction: {result['prediction']}\n"
        pred_text += f"Confidence: {result['confidence']:.2%}\n"
        pred_text += f"Hybrid Probability: {result['probability']:.2%}"
        
        color = 'red' if result['prediction'] == 'Hybrid' else 'green'
        axes[1].text(0.5, 0.5, pred_text, 
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        
        # Localization map
        if 'localization_map' in result and axes[2] is not None:
            loc_map = result['localization_map']
            
            # Overlay heatmap on original image
            overlay = np.array(img).astype(np.float32) / 255.0
            heatmap = plt.cm.jet(loc_map)[:, :, :3]
            
            # Blend
            alpha = 0.5
            blended = overlay * (1 - alpha) + heatmap * alpha
            
            axes[2].imshow(blended)
            axes[2].set_title('Manipulation Localization', fontsize=12, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return result


def batch_process_directory(predictor: HybridImagePredictor,
                            input_dir: str,
                            output_dir: str,
                            visualize: bool = True):
    """
    Process all images in a directory
    
    Args:
        predictor: HybridImagePredictor instance
        input_dir: Directory with input images
        output_dir: Directory to save results
        visualize: Whether to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in img_extensions
    ]
    
    print(f"Found {len(img_paths)} images in {input_dir}")
    
    # Process each image
    results = []
    for img_path in img_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        result = predictor.predict(img_path, return_localization=True)
        results.append(result)
        
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        # Save visualization
        if visualize:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{img_name}_result.png")
            predictor.visualize_prediction(img_path, save_path, show=False)
    
    # Save summary
    import json
    summary = {
        'total_images': len(results),
        'num_hybrid': sum(1 for r in results if r['prediction'] == 'Hybrid'),
        'num_real': sum(1 for r in results if r['prediction'] == 'Real'),
        'results': results
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Processed {len(results)} images")
    print(f"✓ Results saved to {output_dir}")
    print(f"  - Hybrid: {summary['num_hybrid']}")
    print(f"  - Real: {summary['num_real']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    
    # Initialize predictor
    predictor = HybridImagePredictor(
        model_path='checkpoints/best.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_lite_model=False  # Set to True if you used lite model
    )
    
    # Single image prediction
    print("\n" + "=" * 60)
    print("SINGLE IMAGE PREDICTION")
    print("=" * 60)
    
    result = predictor.predict('path/to/test/image.jpg')
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Hybrid Probability: {result['probability']:.2%}")
    
    # Visualize
    predictor.visualize_prediction(
        'path/to/test/image.jpg',
        save_path='prediction_result.png'
    )
    
    # Batch processing
    print("\n" + "=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)
    
    # batch_process_directory(
    #     predictor=predictor,
    #     input_dir='path/to/test/images',
    #     output_dir='results',
    #     visualize=True
    # )
