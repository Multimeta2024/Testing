"""
Comprehensive evaluation script for hybrid image detection
Provides detailed metrics, confusion matrix, and error analysis
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score
)
import json

from hybrid_detection_model import HybridImageDetector, HybridDetectorLite
from hybrid_dataset import HybridImageDataset


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict_all(self):
        """Get predictions for entire test set"""
        all_probs = []
        all_labels = []
        all_localization = []
        
        for rgb, freq, labels, masks in tqdm(self.test_loader, desc='Predicting'):
            rgb = rgb.to(self.device)
            freq = freq.to(self.device)
            
            cls_logits, loc_maps = self.model(rgb, freq)
            probs = torch.sigmoid(cls_logits.squeeze()).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
            if loc_maps is not None:
                all_localization.append(loc_maps.cpu().numpy())
        
        return {
            'probabilities': np.array(all_probs),
            'labels': np.array(all_labels),
            'localization': all_localization if all_localization else None
        }
    
    def compute_metrics(self, predictions, threshold=0.5):
        """Compute comprehensive metrics"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        preds = (probs > threshold).astype(int)
        
        # Basic metrics
        metrics = {}
        
        # AUC-ROC
        metrics['auc_roc'] = roc_auc_score(labels, probs)
        
        # Average Precision (AP)
        metrics['average_precision'] = average_precision_score(labels, probs)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
        
        # False positive rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False negative rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def find_optimal_threshold(self, predictions):
        """Find optimal threshold using Youden's J statistic"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'threshold': optimal_threshold,
            'tpr': tpr[optimal_idx],
            'fpr': fpr[optimal_idx],
            'j_score': j_scores[optimal_idx]
        }
    
    def evaluate_by_confidence(self, predictions, bins=10):
        """Analyze performance by confidence level"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        
        # Convert to confidence (distance from 0.5)
        confidence = np.abs(probs - 0.5) * 2
        
        # Bin by confidence
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(confidence, bin_edges) - 1
        
        results = []
        for i in range(bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_probs = probs[mask]
            bin_labels = labels[mask]
            bin_preds = (bin_probs > 0.5).astype(int)
            
            accuracy = (bin_preds == bin_labels).mean()
            
            results.append({
                'confidence_bin': f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                'num_samples': int(mask.sum()),
                'accuracy': accuracy,
                'avg_confidence': confidence[mask].mean()
            })
        
        return results
    
    def analyze_errors(self, predictions, threshold=0.5):
        """Detailed error analysis"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        preds = (probs > threshold).astype(int)
        
        # Find errors
        errors = preds != labels
        
        # False positives
        fp_mask = (labels == 0) & (preds == 1)
        fp_probs = probs[fp_mask]
        
        # False negatives
        fn_mask = (labels == 1) & (preds == 0)
        fn_probs = probs[fn_mask]
        
        analysis = {
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.mean()),
            'false_positives': {
                'count': int(fp_mask.sum()),
                'avg_confidence': float(fp_probs.mean()) if len(fp_probs) > 0 else 0,
                'max_confidence': float(fp_probs.max()) if len(fp_probs) > 0 else 0,
            },
            'false_negatives': {
                'count': int(fn_mask.sum()),
                'avg_confidence': float(1 - fn_probs.mean()) if len(fn_probs) > 0 else 0,
                'min_confidence': float(fn_probs.min()) if len(fn_probs) > 0 else 0,
            }
        }
        
        return analysis
    
    def plot_roc_curve(self, predictions, save_path=None):
        """Plot ROC curve"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Hybrid Image Detection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, predictions, save_path=None):
        """Plot Precision-Recall curve"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, predictions, threshold=0.5, save_path=None):
        """Plot confusion matrix"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        preds = (probs > threshold).astype(int)
        
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Real', 'Hybrid'],
                   yticklabels=['Real', 'Hybrid'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_probability_distribution(self, predictions, save_path=None):
        """Plot probability distribution for real vs hybrid"""
        probs = predictions['probabilities']
        labels = predictions['labels']
        
        real_probs = probs[labels == 0]
        hybrid_probs = probs[labels == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(real_probs, bins=50, alpha=0.5, label='Real', color='blue')
        plt.hist(hybrid_probs, bins=50, alpha=0.5, label='Hybrid', color='red')
        plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel('Predicted Probability (Hybrid)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Probability Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, predictions, output_dir='evaluation_results'):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute metrics at default threshold
        metrics_50 = self.compute_metrics(predictions, threshold=0.5)
        
        # Find optimal threshold
        optimal = self.find_optimal_threshold(predictions)
        metrics_optimal = self.compute_metrics(predictions, threshold=optimal['threshold'])
        
        # Confidence analysis
        confidence_analysis = self.evaluate_by_confidence(predictions)
        
        # Error analysis
        error_analysis = self.analyze_errors(predictions, threshold=0.5)
        
        # Create report
        report = {
            'summary': {
                'total_samples': len(predictions['labels']),
                'num_real': int((predictions['labels'] == 0).sum()),
                'num_hybrid': int((predictions['labels'] == 1).sum()),
            },
            'metrics_at_threshold_0.5': metrics_50,
            'optimal_threshold': optimal,
            'metrics_at_optimal_threshold': metrics_optimal,
            'confidence_analysis': confidence_analysis,
            'error_analysis': error_analysis,
        }
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self.plot_roc_curve(predictions, 
                          save_path=os.path.join(output_dir, 'roc_curve.png'))
        self.plot_precision_recall_curve(predictions,
                                        save_path=os.path.join(output_dir, 'pr_curve.png'))
        self.plot_confusion_matrix(predictions,
                                  save_path=os.path.join(output_dir, 'confusion_matrix.png'))
        self.plot_probability_distribution(predictions,
                                          save_path=os.path.join(output_dir, 'prob_distribution.png'))
        
        # Print summary
        self._print_report(report)
        
        return report
    
    def _print_report(self, report):
        """Pretty print evaluation report"""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        
        print("\n📊 Dataset Summary:")
        print(f"  Total samples: {report['summary']['total_samples']}")
        print(f"  Real images: {report['summary']['num_real']}")
        print(f"  Hybrid images: {report['summary']['num_hybrid']}")
        
        print("\n📈 Metrics at Threshold 0.5:")
        m = report['metrics_at_threshold_0.5']
        print(f"  AUC-ROC: {m['auc_roc']:.4f}")
        print(f"  Average Precision: {m['average_precision']:.4f}")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall: {m['recall']:.4f}")
        print(f"  F1 Score: {m['f1_score']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")
        
        print("\n🎯 Optimal Threshold:")
        opt = report['optimal_threshold']
        print(f"  Threshold: {opt['threshold']:.4f}")
        print(f"  TPR: {opt['tpr']:.4f}")
        print(f"  FPR: {opt['fpr']:.4f}")
        
        print("\n⚠️  Error Analysis:")
        err = report['error_analysis']
        print(f"  Total errors: {err['total_errors']} ({err['error_rate']:.2%})")
        print(f"  False Positives: {err['false_positives']['count']}")
        print(f"  False Negatives: {err['false_negatives']['count']}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = HybridImageDetector()
    checkpoint = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare test data
    test_labels = [
        # Your test data here
        ("test_img1.jpg", 0, None),
        ("test_img2.jpg", 1, None),
        # ...
    ]
    
    test_dataset = HybridImageDataset(
        img_dir='test_images',
        labels=test_labels,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, device=device)
    
    print("\nRunning evaluation...")
    predictions = evaluator.predict_all()
    
    print("\nGenerating report...")
    report = evaluator.generate_report(predictions, output_dir='evaluation_results')
    
    print("\n✓ Evaluation complete!")
    print("Results saved to evaluation_results/")
