# Image_Restoration_and_Enhancement
RestoraGen: A Generative AI System for Image Restoration and Enhancement

## Overview

A complete pipeline for image restoration and enhancement using fine-tuned Stable Diffusion models. Supports four tasks: denoising, super-resolution, colorization, and inpainting.

## Quick Start

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download COCO images and generate synthetic training pairs:

```
# Download 2,300 images (2000 train, 200 val, 100 test)
python3 scripts/download_coco_subset.py --split val2017 --num_images 2300 --out_dir data

# Generate synthetic pairs for all tasks
python3 scripts/make_synthetic_pairs.py --clean_root data/clean --out_root data/pairs --sr_scale 4
```

This creates training pairs in:
- `data/pairs/denoise/{train,val,test}/{input,gt}`
- `data/pairs/sr_x4/{train,val,test}/{input,gt}`
- `data/pairs/colorize/{train,val,test}/{input,gt}` (inputs are grayscale PNG)
- `data/pairs/inpaint/{train,val,test}/{input,mask,gt}`

**Dataset Split**: 2000 training, 200 validation, 100 test images per task.

### 3. Launch Gradio UI

Run the interactive web interface:

```
python3 app.py
```

The app will open in your browser, allowing you to:
- Upload images for restoration
- Select tasks (denoise, super-resolution, colorization, inpainting)
- Choose between fine-tuned or pretrained models
- Adjust task-specific parameters (strength, scale, prompts)
- Upload masks for inpainting

### 4. Quick Test (Command Line)

Test inpainting on a single sample:

```
python3 scripts/test_inference_sd_inpaint.py \
  --pairs_root data/pairs/inpaint \
  --output outputs/sd_inpaint_test.png
```

## Project Structure

```
Image_Restoration_and_Enhancement/
├── app.py                    # Gradio web interface (main entry point)
├── src/
│   ├── inference.py          # Core inference pipeline (RestorationPipeline)
│   └── metrics.py            # Evaluation metrics (PSNR, SSIM, LPIPS)
├── scripts/
│   ├── download_coco_subset.py      # Download COCO dataset
│   ├── make_synthetic_pairs.py      # Generate training pairs
│   ├── train_denoising.py           # Train denoising model
│   ├── train_super_resolution.py    # Train super-resolution model
│   ├── train_colorization.py        # Train colorization model
│   ├── train_inpainting.py          # Train inpainting model
│   ├── evaluate_model.py            # Evaluate model performance
│   └── generate_predictions.py      # Generate predictions on test set
├── data/
│   ├── clean/                # Original COCO images (train/val/test splits)
│   ├── pairs/                # Synthetic training pairs per task
│   └── demo/                 # Demo images and masks
└── outputs/
    └── models/               # Fine-tuned models (one per task)
        ├── denoising/best/
        ├── super_resolution/best/
        ├── colorization/best/
        └── inpainting/best/
```

## Important Files

- **`app.py`**: Gradio web interface for interactive image restoration
- **`src/inference.py`**: Core `RestorationPipeline` class handling model loading and inference
- **`scripts/train_*.py`**: Training scripts for each task (fine-tune Stable Diffusion)
- **`scripts/make_synthetic_pairs.py`**: Generate synthetic degraded/clean image pairs
- **`outputs/models/{task}/best/`**: Fine-tuned model checkpoints (auto-loaded by pipeline)

## Training Results

Fine-tuned models were trained on 2000 training images with 200 validation and 100 test images per task. Performance metrics:

- **Denoising**: PSNR ≈ 13.2, SSIM ≈ 0.17, LPIPS ≈ 0.72 — moderate noise removal with limited structural recovery
- **Super-resolution**: PSNR ≈ 9.7, SSIM ≈ 0.10, LPIPS ≈ 0.88 — challenging detail synthesis at current training scale
- **Colorization**: PSNR ≈ 8.2, SSIM ≈ 0.07, LPIPS ≈ 0.87 — plausible global tones but limited perceptual sharpness
- **Inpainting**: PSNR ≈ 9.7, SSIM ≈ 0.08, LPIPS ≈ 0.80 — reasonable region filling with limited texture consistency

Fine-tuned models are stored in `outputs/models/{task}/best/` and automatically loaded when available. The pipeline falls back to pretrained models if fine-tuned models are not found (when using pretrained mode).

