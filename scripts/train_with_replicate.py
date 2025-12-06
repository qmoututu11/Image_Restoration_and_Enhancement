#!/usr/bin/env python3
"""
Fine-tune models on Replicate AI platform.

Replicate supports fine-tuning image models (like Stable Diffusion) using their API.
This script helps you fine-tune restoration models on Replicate instead of locally.

Benefits:
- No local GPU needed
- Faster training (Replicate's infrastructure)
- Easy model deployment
- Automatic versioning

Requirements:
- Replicate account with API token
- Training data prepared in the expected format
- Destination model created on Replicate
"""

import os
import sys
import argparse
from pathlib import Path
import zipfile
import tempfile
import logging

try:
    import replicate
except ImportError:
    print("Error: replicate package not installed. Install with: pip install replicate", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_zip(input_dir: Path, gt_dir: Path, output_zip: Path):
    """Create a zip file with training pairs for Replicate."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        input_images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
        gt_images = sorted(gt_dir.glob("*.jpg")) + sorted(gt_dir.glob("*.png"))
        
        # Create pairs
        pairs = []
        for inp in input_images:
            # Find matching GT (same name)
            gt = gt_dir / inp.name
            if not gt.exists():
                # Try with different extension
                gt = gt_dir / (inp.stem + ".jpg")
                if not gt.exists():
                    gt = gt_dir / (inp.stem + ".png")
            
            if gt.exists():
                pairs.append((inp, gt))
        
        logger.info(f"Found {len(pairs)} image pairs")
        
        # Add to zip
        for inp, gt in pairs:
            zf.write(inp, f"input/{inp.name}")
            zf.write(gt, f"gt/{gt.name}")
        
        logger.info(f"Created training zip: {output_zip} with {len(pairs)} pairs")
        return len(pairs)


def fine_tune_on_replicate(
    model_owner: str,
    model_name: str,
    training_data_zip: Path,
    destination_model: str,
    trigger_word: str = None,
    api_token: str = None
):
    """
    Fine-tune a model on Replicate.
    
    Args:
        model_owner: Owner of the base model (e.g., "stability-ai")
        model_name: Name of the base model (e.g., "stable-diffusion-v1-5")
        training_data_zip: Path to zip file with training data
        destination_model: Destination model in format "owner/name" (must exist on Replicate)
        trigger_word: Optional trigger word for the fine-tuned model
        api_token: Replicate API token
    
    Returns:
        Training job information
    """
    api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN not set")
    
    replicate.api_token = api_token
    client = replicate.Client(api_token=api_token)
    
    # Check if destination model exists
    try:
        dest_owner, dest_name = destination_model.split("/")
        dest_model = client.models.get(owner=dest_owner, name=dest_name)
        logger.info(f"✓ Destination model found: {destination_model}")
    except Exception as e:
        logger.error(f"✗ Destination model not found: {destination_model}")
        logger.error(f"  Error: {e}")
        logger.info("\nTo create a destination model:")
        logger.info("  1. Go to https://replicate.com/create")
        logger.info("  2. Create a new model")
        logger.info("  3. Use the format: your-username/model-name")
        raise
    
    # Upload training data
    logger.info(f"Uploading training data: {training_data_zip}")
    with open(training_data_zip, 'rb') as f:
        training_data = client.files.create(file=f)
    
    logger.info(f"✓ Training data uploaded: {training_data.url}")
    
    # Start fine-tuning
    logger.info(f"Starting fine-tuning...")
    logger.info(f"  Base model: {model_owner}/{model_name}")
    logger.info(f"  Destination: {destination_model}")
    
    try:
        # Replicate fine-tuning API
        # Note: API format may vary - check https://replicate.com/docs for latest
        # 
        # For image models (like Stable Diffusion), Replicate uses:
        # - input_images: list of image URLs or file objects
        # - trigger_word: optional trigger word to activate the concept
        # - destination: destination model (must exist)
        #
        # Common format:
        training = client.trainings.create(
            version=f"{model_owner}/{model_name}",  # Base model version
            destination=destination_model,  # Your destination model
            input={
                "input_images": training_data.url,  # URL to training data
                "trigger_word": trigger_word or f"{model_name}_style",  # Trigger word
            }
        )
        
        # Alternative format (if above doesn't work, try):
        # training = replicate.trainings.create(
        #     model=f"{model_owner}/{model_name}",
        #     destination=destination_model,
        #     input_images=[training_data.url],
        #     trigger_word=trigger_word,
        # )
        
        logger.info(f"✓ Training started!")
        logger.info(f"  Training ID: {training.id}")
        logger.info(f"  Status: {training.status}")
        logger.info(f"  Monitor at: https://replicate.com/{destination_model}")
        
        return training
        
    except Exception as e:
        logger.error(f"✗ Failed to start training: {e}")
        logger.info("\nNote: Fine-tuning API may vary by model type.")
        logger.info("Check Replicate docs: https://replicate.com/docs/guides/fine-tune-an-image-model")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune models on Replicate AI platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune denoising model
  python scripts/train_with_replicate.py \\
    --task denoise \\
    --base_model stability-ai/stable-diffusion-v1-5 \\
    --destination your-username/denoise-model \\
    --train_input data/pairs/denoise/train/input \\
    --train_gt data/pairs/denoise/train/gt

  # Fine-tune with trigger word
  python scripts/train_with_replicate.py \\
    --task colorize \\
    --base_model stability-ai/stable-diffusion-v1-5 \\
    --destination your-username/colorize-model \\
    --trigger_word "colorized photo" \\
    --train_input data/pairs/colorize/train/input \\
    --train_gt data/pairs/colorize/train/gt

Note:
  - You must create the destination model on Replicate first
  - Go to https://replicate.com/create to create a new model
  - Training costs credits on Replicate
  - Check your account balance at https://replicate.com/account/billing
        """
    )
    
    parser.add_argument("--task", type=str, required=True,
                       choices=["denoise", "sr", "colorize", "inpaint"],
                       help="Task to fine-tune for")
    parser.add_argument("--base_model", type=str,
                       default="stability-ai/stable-diffusion-v1-5",
                       help="Base model to fine-tune (e.g., stability-ai/stable-diffusion-v1-5)")
    parser.add_argument("--destination", type=str, required=True,
                       help="Destination model in format 'owner/name' (must exist on Replicate)")
    parser.add_argument("--train_input", type=str, required=True,
                       help="Directory with input images")
    parser.add_argument("--train_gt", type=str, required=True,
                       help="Directory with ground truth images")
    parser.add_argument("--trigger_word", type=str, default=None,
                       help="Optional trigger word for the fine-tuned model")
    parser.add_argument("--api_token", type=str, default=None,
                       help="Replicate API token (or set REPLICATE_API_TOKEN env var)")
    parser.add_argument("--output_zip", type=str, default=None,
                       help="Output path for training data zip (default: temp file)")
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.train_input)
    gt_dir = Path(args.train_gt)
    
    if not input_dir.exists():
        parser.error(f"Input directory not found: {input_dir}")
    if not gt_dir.exists():
        parser.error(f"GT directory not found: {gt_dir}")
    
    # Create training zip
    if args.output_zip:
        zip_path = Path(args.output_zip)
    else:
        zip_path = Path(f"training_data_{args.task}.zip")
    
    logger.info(f"Creating training data zip...")
    num_pairs = create_training_zip(input_dir, gt_dir, zip_path)
    
    if num_pairs == 0:
        logger.error("No training pairs found!")
        sys.exit(1)
    
    # Start fine-tuning
    try:
        model_owner, model_name = args.base_model.split("/")
        training = fine_tune_on_replicate(
            model_owner=model_owner,
            model_name=model_name,
            training_data_zip=zip_path,
            destination_model=args.destination,
            trigger_word=args.trigger_word,
            api_token=args.api_token
        )
        
        logger.info("\n" + "="*60)
        logger.info("Training started successfully!")
        logger.info("="*60)
        logger.info(f"Training ID: {training.id}")
        logger.info(f"Monitor progress at: https://replicate.com/{args.destination}")
        logger.info(f"\nNote: Training costs credits. Check balance at:")
        logger.info(f"  https://replicate.com/account/billing")
        
    except Exception as e:
        logger.error(f"\nFailed to start training: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Ensure destination model exists on Replicate")
        logger.info("  2. Check API token is valid")
        logger.info("  3. Verify you have sufficient credits")
        logger.info("  4. Check Replicate docs for latest API: https://replicate.com/docs")
        sys.exit(1)


if __name__ == "__main__":
    main()

