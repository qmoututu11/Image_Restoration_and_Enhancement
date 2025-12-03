#!/usr/bin/env python3
"""
Generate predictions on test set for evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import RestorationPipeline
from PIL import Image
from tqdm import tqdm


def generate_predictions(test_root: Path, output_root: Path, split: str = "test"):
    """Generate predictions for all tasks on test set."""
    
    pipeline = RestorationPipeline()
    
    # Task configuration
    tasks = {
        "denoise": {
            "task_list": ["denoise"],
            "kwargs": {}
        },
        "sr_x4": {
            "task_list": ["sr"],
            "kwargs": {"sr_scale": 4}
        },
        "colorize": {
            "task_list": ["colorize"],
            "kwargs": {}
        },
        "inpaint": {
            "task_list": ["inpaint"],
            "kwargs": {}
        }
    }
    
    for task_name, config in tasks.items():
        input_dir = test_root / task_name / split / "input"
        output_dir = output_root / task_name / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"Skipping {task_name}: input directory not found: {input_dir}")
            continue
        
        # Get mask directory for inpainting
        mask_dir = None
        if task_name == "inpaint":
            mask_dir = test_root / task_name / split / "mask"
        
        print(f"\n{'='*60}")
        print(f"Processing {task_name}...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            continue
        
        for img_path in tqdm(image_files, desc=f"  {task_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                
                # Load mask for inpainting
                if task_name == "inpaint" and mask_dir:
                    mask_path = mask_dir / img_path.name
                    if mask_path.exists():
                        mask = Image.open(mask_path).convert("L")
                        config["kwargs"]["mask"] = mask
                    else:
                        config["kwargs"]["mask"] = None
                
                # Process image
                result = pipeline.process(img, config["task_list"], **config["kwargs"])
                
                # Save final result
                output_path = output_dir / img_path.name
                result["final"].save(output_path)
                
            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                continue
        
        print(f"{task_name}: {len(image_files)} images processed")
    
    print(f"\n{'='*60}")
    print("All predictions generated!")
    print(f"{'='*60}")
    print(f"\nPredictions saved to: {output_root}")
    print(f"\nNext step: Run evaluation:")
    print(f"  python3 scripts/evaluate_model.py \\")
    print(f"    --pred_root {output_root} \\")
    print(f"    --gt_root {test_root} \\")
    print(f"    --split {split}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions on test set")
    parser.add_argument("--test_root", type=str, default="data/pairs",
                       help="Root directory with test data (default: data/pairs)")
    parser.add_argument("--output_root", type=str, default="outputs/predictions",
                       help="Output directory for predictions (default: outputs/predictions)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Which split to process (default: test)")
    args = parser.parse_args()
    
    test_root = Path(args.test_root).resolve()
    output_root = Path(args.output_root).resolve()
    
    if not test_root.exists():
        print(f"Error: Test root not found: {test_root}")
        sys.exit(1)
    
    generate_predictions(test_root, output_root, args.split)


