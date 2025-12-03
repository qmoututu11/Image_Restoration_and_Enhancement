#!/usr/bin/env python3
"""
Evaluate model performance on test datasets for all 4 tasks.
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import evaluate_task, print_results
import torch


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test datasets")
    parser.add_argument("--pred_root", type=str, required=True, 
                       help="Root directory containing predictions (should have denoise/, sr_x4/, colorize/, inpaint/ subdirs)")
    parser.add_argument("--gt_root", type=str, default="data/pairs", 
                       help="Root directory containing ground truth (default: data/pairs)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Which split to evaluate (default: test)")
    parser.add_argument("--tasks", nargs="+", default=["denoise", "sr_x4", "colorize", "inpaint"],
                       choices=["denoise", "sr_x4", "colorize", "inpaint"],
                       help="Tasks to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS calculation (faster)")
    parser.add_argument("--no-fid", action="store_true", help="Skip FID calculation (faster, but FID is important for generative models)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device for LPIPS/FID (auto/cpu/cuda, default: auto)")
    args = parser.parse_args()
    
    pred_root = Path(args.pred_root).resolve()
    gt_root = Path(args.gt_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"LPIPS: {'enabled' if not args.no_lpips else 'disabled'}")
    print(f"FID: {'enabled' if not args.no_fid else 'disabled'}")
    print(f"Evaluating split: {args.split}\n")
    
    all_results = {}
    
    # Task name mapping (prediction dir name -> task name)
    task_mapping = {
        "denoise": "denoise",
        "sr_x4": "super-resolution",
        "colorize": "colorization",
        "inpaint": "inpainting"
    }
    
    for task_dir in args.tasks:
        pred_dir = pred_root / task_dir / args.split
        gt_dir = gt_root / task_dir / args.split / "gt"
        
        if not pred_dir.exists():
            print(f"Warning: Prediction directory not found: {pred_dir}")
            continue
        
        if not gt_dir.exists():
            print(f"Warning: Ground truth directory not found: {gt_dir}")
            continue
        
        task_name = task_mapping.get(task_dir, task_dir)
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Predictions: {pred_dir}")
        print(f"Ground Truth: {gt_dir}")
        print(f"{'='*60}")
        
        try:
            results = evaluate_task(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                task_name=task_name,
                use_lpips=not args.no_lpips,
                use_fid=not args.no_fid,
                device=device
            )
            print_results(results)
            all_results[task_dir] = results
        
        except Exception as e:
            print(f"Error evaluating {task_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_dir, results in all_results.items():
        task_name = task_mapping.get(task_dir, task_dir)
        metrics = results.get('metrics', {})
        if 'psnr' in metrics:
            metric_str = f"{task_name:20s} PSNR: {metrics['psnr']['mean']:6.2f} dB, SSIM: {metrics['ssim']['mean']:.4f}"
            if 'lpips' in metrics:
                metric_str += f", LPIPS: {metrics['lpips']['mean']:.4f}"
            if 'fid' in metrics:
                metric_str += f", FID: {metrics['fid']['mean']:.2f}"
            print(metric_str)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

