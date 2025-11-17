#!/usr/bin/env python3
"""
Evaluation metrics for image restoration tasks.
Supports: PSNR, SSIM, LPIPS, and task-specific metrics.
"""

import numpy as np
from pathlib import Path
import sys

try:
    import cv2
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Install with: pip install scikit-image lpips", file=sys.stderr)
    sys.exit(1)

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips", file=sys.stderr)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array [0-255]."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_for_lpips(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to torch tensor for LPIPS."""
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    # Normalize to [-1, 1] for LPIPS
    img = img * 2.0 - 1.0
    return img


class MetricsCalculator:
    """Calculate evaluation metrics for image restoration."""
    
    def __init__(self, use_lpips: bool = True, device: str = "cpu"):
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        self.device = device
        if self.use_lpips:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
    
    def calculate_psnr(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate PSNR (Peak Signal-to-Noise Ratio). Higher is better."""
        # Ensure same shape
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        return psnr(gt, pred, data_range=255.0)
    
    def calculate_ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate SSIM (Structural Similarity Index). Higher is better [0-1]."""
        # Ensure same shape
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        # SSIM expects grayscale or multichannel
        return ssim(gt, pred, data_range=255.0, channel_axis=2 if len(gt.shape) == 3 else None)
    
    def calculate_lpips(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity). Lower is better [0-1]."""
        if not self.use_lpips:
            return None
        
        # Ensure same shape
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        pred_tensor = preprocess_for_lpips(pred).to(self.device)
        gt_tensor = preprocess_for_lpips(gt).to(self.device)
        
        with torch.no_grad():
            dist = self.lpips_model(pred_tensor, gt_tensor)
        return dist.item()
    
    def calculate_all(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Calculate all metrics."""
        results = {
            'psnr': self.calculate_psnr(pred, gt),
            'ssim': self.calculate_ssim(pred, gt),
        }
        
        if self.use_lpips:
            results['lpips'] = self.calculate_lpips(pred, gt)
        
        return results


def evaluate_task(
    pred_dir: Path,
    gt_dir: Path,
    task_name: str = "denoise",
    use_lpips: bool = True,
    device: str = "cpu"
) -> dict:
    """
    Evaluate a task by comparing predictions to ground truth.
    
    Args:
        pred_dir: Directory containing predicted images
        gt_dir: Directory containing ground truth images
        task_name: Name of the task (for reporting)
        use_lpips: Whether to calculate LPIPS (requires GPU for speed)
        device: Device for LPIPS calculation
    
    Returns:
        Dictionary with metrics and statistics
    """
    calc = MetricsCalculator(use_lpips=use_lpips, device=device)
    
    pred_files = sorted([f for f in pred_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    gt_files = sorted([f for f in gt_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    
    if len(pred_files) != len(gt_files):
        print(f"Warning: Mismatch - {len(pred_files)} predictions vs {len(gt_files)} ground truth")
    
    # Match files by name (handle extensions)
    matched_pairs = []
    for pred_file in pred_files:
        # Try exact match first
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            # Try with different extension
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_gt = gt_dir / (pred_file.stem + ext)
                if alt_gt.exists():
                    gt_file = alt_gt
                    break
        
        if gt_file.exists():
            matched_pairs.append((pred_file, gt_file))
    
    if not matched_pairs:
        raise ValueError(f"No matching files found between {pred_dir} and {gt_dir}")
    
    all_metrics = {'psnr': [], 'ssim': []}
    if use_lpips:
        all_metrics['lpips'] = []
    
    print(f"Evaluating {task_name}: {len(matched_pairs)} image pairs...")
    
    for i, (pred_path, gt_path) in enumerate(matched_pairs):
        try:
            pred_img = load_image(pred_path)
            gt_img = load_image(gt_path)
            
            metrics = calc.calculate_all(pred_img, gt_img)
            
            for key, value in metrics.items():
                if value is not None:
                    all_metrics[key].append(value)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(matched_pairs)}")
        
        except Exception as e:
            print(f"Error processing {pred_path.name}: {e}")
            continue
    
    # Calculate statistics
    results = {
        'task': task_name,
        'num_samples': len(matched_pairs),
        'metrics': {}
    }
    
    for metric_name, values in all_metrics.items():
        if values:
            results['metrics'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {results['task']}")
    print(f"{'='*60}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"\nMetrics:")
    
    for metric_name, stats in results['metrics'].items():
        print(f"\n  {metric_name.upper()}:")
        print(f"    Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print(f"\n{'='*60}\n")

