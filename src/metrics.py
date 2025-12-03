#!/usr/bin/env python3
"""
Evaluation metrics for image restoration tasks.
Supports: PSNR, SSIM, LPIPS, and FID metrics.
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

try:
    from scipy import linalg
    from torchvision.models import inception_v3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: FID dependencies not available. Install with: pip install scipy", file=sys.stderr)


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
    
    def __init__(self, use_lpips: bool = True, use_fid: bool = True, device: str = "cpu"):
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        self.use_fid = use_fid and FID_AVAILABLE
        self.device = device
        
        if self.use_lpips:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        
        if self.use_fid:
            # Load Inception v3 for FID
            self.inception_model = inception_v3(weights='DEFAULT', transform_input=False)
            self.inception_model.fc = torch.nn.Identity()  # Remove final layer
            self.inception_model = self.inception_model.to(device)
            self.inception_model.eval()
            # Preprocessing for Inception
            self.fid_transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
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
    
    def _get_inception_features(self, img: np.ndarray) -> torch.Tensor:
        """Extract Inception v3 features for FID calculation."""
        if not self.use_fid:
            return None
        
        # Convert numpy to PIL to tensor
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)
        img_tensor = self.fid_transform(img_tensor)
        
        with torch.no_grad():
            features = self.inception_model(img_tensor)
        return features.cpu()
    
    def calculate_fid(self, pred_images, gt_images):
        """
        Calculate FID (Fréchet Inception Distance) between two sets of images.
        Lower is better. This is a dataset-level metric.
        
        Args:
            pred_images: List of predicted images as numpy arrays [0-255]
            gt_images: List of ground truth images as numpy arrays [0-255]
        
        Returns:
            FID score (lower is better)
        """
        if not self.use_fid:
            return None
        
        if len(pred_images) != len(gt_images):
            print(f"Warning: FID requires equal number of images. Got {len(pred_images)} pred vs {len(gt_images)} gt")
            return None
        
        # Extract features for all images
        pred_features = []
        gt_features = []
        
        print("  Extracting Inception features for FID...")
        for i, (pred_img, gt_img) in enumerate(zip(pred_images, gt_images)):
            # Resize to match if needed
            if pred_img.shape[:2] != gt_img.shape[:2]:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            pred_feat = self._get_inception_features(pred_img)
            gt_feat = self._get_inception_features(gt_img)
            
            pred_features.append(pred_feat)
            gt_features.append(gt_feat)
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(pred_images)} images")
        
        # Stack features
        pred_features = torch.cat(pred_features, dim=0).numpy()
        gt_features = torch.cat(gt_features, dim=0).numpy()
        
        # Calculate FID
        mu1, sigma1 = pred_features.mean(axis=0), np.cov(pred_features, rowvar=False)
        mu2, sigma2 = gt_features.mean(axis=0), np.cov(gt_features, rowvar=False)
        
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        # Calculate sqrt of product between cov
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
        return float(fid)
    
    def calculate_all(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Calculate all per-image metrics."""
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
    use_fid: bool = True,
    device: str = "cpu"
) -> dict:
    """
    Evaluate a task by comparing predictions to ground truth.
    
    Args:
        pred_dir: Directory containing predicted images
        gt_dir: Directory containing ground truth images
        task_name: Name of the task (for reporting)
        use_lpips: Whether to calculate LPIPS (requires GPU for speed)
        use_fid: Whether to calculate FID (dataset-level metric)
        device: Device for LPIPS/FID calculation
    
    Returns:
        Dictionary with metrics and statistics
    """
    calc = MetricsCalculator(use_lpips=use_lpips, use_fid=use_fid, device=device)
    
    pred_files = sorted([f for f in pred_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    gt_file_set = {f.name for f in gt_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    
    if len(pred_files) != len(gt_file_set):
        print(f"Warning: Mismatch - {len(pred_files)} predictions vs {len(gt_file_set)} ground truth")
    
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
    
    # For FID, we need to collect all images first
    pred_images_list = []
    gt_images_list = []
    
    print(f"Evaluating {task_name}: {len(matched_pairs)} image pairs...")
    
    for i, (pred_path, gt_path) in enumerate(matched_pairs):
        try:
            pred_img = load_image(pred_path)
            gt_img = load_image(gt_path)
            
            # Calculate per-image metrics
            metrics = calc.calculate_all(pred_img, gt_img)
            
            for key, value in metrics.items():
                if value is not None:
                    all_metrics[key].append(value)
            
            # Collect images for FID (dataset-level metric)
            if use_fid:
                pred_images_list.append(pred_img)
                gt_images_list.append(gt_img)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(matched_pairs)}")
        
        except Exception as e:
            print(f"Error processing {pred_path.name}: {e}")
            continue
    
    # Calculate FID if requested (dataset-level metric)
    if use_fid and pred_images_list:
        print(f"\n  Calculating FID (dataset-level metric)...")
        try:
            fid_score = calc.calculate_fid(pred_images_list, gt_images_list)
            if fid_score is not None:
                all_metrics['fid'] = [fid_score]  # Store as list for consistency
        except Exception as e:
            print(f"  Warning: FID calculation failed: {e}")
    
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

