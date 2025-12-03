#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion for super-resolution on your synthetic pairs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from datetime import datetime
from src.metrics import MetricsCalculator


class SuperResolutionDataset(Dataset):
    """Dataset for super-resolution: low-resolution input -> high-resolution output."""
    
    def __init__(self, input_dir: Path, gt_dir: Path, image_size=512, max_samples=None):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        
        # Get all image files
        input_files = sorted(list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png")))
        self.files = []
        for f in input_files:
            # Try to find matching GT file (could be .jpg or .png)
            gt_path_jpg = self.gt_dir / f"{f.stem}.jpg"
            gt_path_png = self.gt_dir / f"{f.stem}.png"
            if gt_path_jpg.exists() or gt_path_png.exists():
                self.files.append(f)
        
        # Limit samples for quick testing
        if max_samples is not None and max_samples > 0:
            self.files = self.files[:max_samples]
        
        print(f"Found {len(self.files)} super-resolution pairs" + (f" (limited to {max_samples} for quick test)" if max_samples else ""))
        
        # Transform for input (low-res): resize to smaller size, convert to tensor, normalize to [-1, 1]
        # Transform for GT (high-res): resize to image_size, convert to tensor, normalize to [-1, 1]
        self.input_transform = transforms.Compose([
            transforms.Resize((image_size // 4, image_size // 4), Image.LANCZOS),  # Low-res input
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB to [-1, 1]
        ])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.LANCZOS),  # High-res GT
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        input_path = self.files[idx]
        # GT file might be .jpg or .png (match by stem)
        gt_path_jpg = self.gt_dir / f"{input_path.stem}.jpg"
        gt_path_png = self.gt_dir / f"{input_path.stem}.png"
        gt_path = gt_path_jpg if gt_path_jpg.exists() else gt_path_png
        
        # Load images
        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        
        # Apply transforms
        input_tensor = self.input_transform(input_img)
        gt_tensor = self.gt_transform(gt_img)
        
        return {
            "input": input_tensor,
            "gt": gt_tensor
        }


def train_super_resolution(
    train_input_dir: Path,
    train_gt_dir: Path,
    val_input_dir: Path,
    val_gt_dir: Path,
    output_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    resume_from: str = None,
    image_size: int = 256,  # Reduced from 512 to save memory
    max_train_samples: int = None,  # Limit training samples for quick test
    max_val_samples: int = None,  # Limit validation samples for quick test
    base_model: str = "runwayml/stable-diffusion-v1-5"  # Base model to fine-tune from
):
    """Fine-tune Stable Diffusion for super-resolution."""
    
    # Setup logging to file
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    
    # Configure logging to file only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a')  # Append mode
        ],
        force=True  # Override any existing configuration
    )
    logger = logging.getLogger(__name__)
    
    # Ensure file handler flushes immediately
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO)
    
    # Log start
    logger.info("="*60)
    logger.info("Fine-tuning Stable Diffusion for Super-Resolution")
    logger.info("="*60)
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("="*60)
    print("Fine-tuning Stable Diffusion for Super-Resolution")
    print("="*60)
    print(f"Log file: {log_file}")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Available: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"GPU Available: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Device: {torch.cuda.current_device()}")
    else:
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        print("WARNING: CUDA not available! Training will be very slow on CPU.")
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )
    
    # Load pre-trained model
    print("\nLoading pre-trained Stable Diffusion for super-resolution...")
    # Use Img2Img pipeline instead of Upscaler for training
    # The Upscaler UNet expects 7 input channels which complicates training
    # For super-resolution: Use specialized upscaler model if available
    # Note: StableDiffusionUpscalePipeline has different UNet architecture (7 channels)
    # So we use Img2Img which is compatible with our training setup
    from diffusers import StableDiffusionImg2ImgPipeline
    # Load pre-trained model
    print(f"\nLoading pre-trained Stable Diffusion Img2Img...")
    print(f"Using base model: {base_model}")
    
    # Get Hugging Face token if available
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print("Hugging Face token detected")
        logger.info("Hugging Face token detected")
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        logger.info(f"Resuming from checkpoint: {resume_from}")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(resume_from)
    else:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=False,
            token=hf_token,
        )
        # Disable safety checker immediately after loading
        if hasattr(pipeline, 'disable_safety_checker'):
            pipeline.disable_safety_checker()
        else:
            # Manual disable if method doesn't exist
            pipeline.safety_checker = None
            pipeline.feature_extractor = None
    
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    # Verify UNet has correct input channels (should be 4 for Img2Img, not 7 for Upscaler)
    # Check the first conv layer's input channels
    if hasattr(unet, 'conv_in'):
        in_channels = unet.conv_in.in_channels
        if in_channels != 4:
            raise ValueError(
                f"UNet has {in_channels} input channels, but expected 4. "
                f"This suggests the model is from StableDiffusionUpscalePipeline. "
                f"Please restart your Colab runtime to clear cached models, or use a checkpoint from StableDiffusionImg2ImgPipeline."
            )
        print(f"Verified UNet has {in_channels} input channels (correct for Img2Img)")
    elif hasattr(unet.config, 'in_channels'):
        in_channels = unet.config.in_channels
        if in_channels != 4:
            raise ValueError(
                f"UNet has {in_channels} input channels, but expected 4. "
                f"This suggests the model is from StableDiffusionUpscalePipeline. "
                f"Please restart your Colab runtime to clear cached models, or use a checkpoint from StableDiffusionImg2ImgPipeline."
            )
        print(f"Verified UNet has {in_channels} input channels (correct for Img2Img)")
    
    # Freeze VAE and text encoder (only train UNet)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    
    # Enable gradient checkpointing to save memory
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing to save memory")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Setup scheduler
    # Calculate steps based on actual dataset size
    train_dataset_temp = SuperResolutionDataset(train_input_dir, train_gt_dir, image_size=image_size)
    num_train_steps = max(
        1,
        len(train_dataset_temp) * num_epochs // (batch_size * gradient_accumulation_steps)
    )
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    # Setup datasets
    train_dataset = SuperResolutionDataset(train_input_dir, train_gt_dir, image_size=image_size, max_samples=max_train_samples)
    val_dataset = SuperResolutionDataset(val_input_dir, val_gt_dir, image_size=image_size, max_samples=max_val_samples) if val_input_dir.exists() else None
    
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    vae.eval()
    text_encoder.eval()
    
    # Prepare with accelerator
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    
    # Validation function
    def run_validation(epoch: int, val_dataset, pipeline, unet_model, output_dir: Path, num_samples: int = 4):
        """Run validation: sample images, compute metrics, save comparisons."""
        if val_dataset is None or len(val_dataset) == 0:
            return
        
        val_output_dir = output_dir / "val_samples"
        val_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample a few validation images
        num_samples = min(num_samples, len(val_dataset))
        sample_indices = np.linspace(0, len(val_dataset) - 1, num_samples, dtype=int)
        
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        # Initialize metrics calculator (use GPU if available for LPIPS)
        device_for_metrics = accelerator.device if torch.cuda.is_available() else "cpu"
        metrics_calc = MetricsCalculator(use_lpips=True, use_fid=False, device=device_for_metrics)
        
        # Temporarily update pipeline with current UNet
        pipeline.unet = accelerator.unwrap_model(unet_model)
        
        device = accelerator.device
        pipeline = pipeline.to(device)
        
        # Disable safety checker
        if hasattr(pipeline, 'safety_checker'):
            pipeline.safety_checker = None
        if hasattr(pipeline, 'feature_extractor'):
            pipeline.feature_extractor = None
        if hasattr(pipeline, 'requires_safety_checker'):
            pipeline.requires_safety_checker = False
        
        # Set components to eval mode
        pipeline.unet.eval()
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        
        try:
            with torch.no_grad():
                for i, idx in enumerate(sample_indices):
                    sample = val_dataset[idx]
                    input_img = sample["input"]
                    gt_img = sample["gt"]
                    
                    # Denormalize input for visualization
                    input_vis = (input_img + 1.0) / 2.0
                    input_vis = torch.clamp(input_vis, 0, 1)
                    input_pil = transforms.ToPILImage()(input_vis)
                    
                    # Run inference
                    prompt = "high quality, detailed, sharp"
                    result = pipeline(
                        prompt=prompt,
                        image=input_pil,
                        num_inference_steps=20,
                        guidance_scale=7.0
                    ).images[0]
                    
                    # Convert to numpy for metrics
                    result_np = np.array(result)
                    gt_vis = (gt_img + 1.0) / 2.0
                    gt_vis = torch.clamp(gt_vis, 0, 1)
                    gt_np = (gt_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # Resize result to match GT if needed
                    if result_np.shape[:2] != gt_np.shape[:2]:
                        result_np = cv2.resize(result_np, (gt_np.shape[1], gt_np.shape[0]))
                    
                    # Compute metrics using MetricsCalculator (includes PSNR, SSIM, LPIPS)
                    try:
                        metrics = metrics_calc.calculate_all(result_np, gt_np)
                        psnr_values.append(metrics['psnr'])
                        ssim_values.append(metrics['ssim'])
                        if metrics.get('lpips') is not None:
                            lpips_values.append(metrics['lpips'])
                    except Exception as e:
                        # Fallback to basic metrics if LPIPS fails
                        try:
                            from skimage.metrics import peak_signal_noise_ratio as psnr
                            from skimage.metrics import structural_similarity as ssim
                            psnr_val = psnr(gt_np, result_np, data_range=255.0)
                            ssim_val = ssim(gt_np, result_np, data_range=255.0, channel_axis=2)
                            psnr_values.append(psnr_val)
                            ssim_values.append(ssim_val)
                        except ImportError:
                            pass
                    
                    # Create side-by-side comparison
                    input_np = (input_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # Resize input to match if needed
                    if input_np.shape[:2] != gt_np.shape[:2]:
                        input_np = cv2.resize(input_np, (gt_np.shape[1], gt_np.shape[0]))
                    comparison = np.hstack([input_np, result_np, gt_np])
                    comparison_pil = Image.fromarray(comparison)
                    
                    # Save comparison (overwrite if exists)
                    save_path = val_output_dir / f"epoch_{epoch+1}_sample_{i+1}_idx{idx}.png"
                    comparison_pil.save(save_path)
        finally:
            pipeline.unet.train()
            torch.cuda.empty_cache()
        
            # Log metrics
            if psnr_values:
                avg_psnr = np.mean(psnr_values)
                avg_ssim = np.mean(ssim_values)
                metric_str = f"Validation (epoch {epoch+1}): PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}"
                
                if lpips_values:
                    avg_lpips = np.mean(lpips_values)
                    metric_str += f", LPIPS={avg_lpips:.4f}"
                
                logger.info(metric_str)
                print(metric_str)
    
    # Training loop
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Total steps: {num_train_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Image size: {image_size}")
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Total steps: {num_train_steps}")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        unet.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            with accelerator.accumulate(unet):
                # Get images
                input_images = batch["input"].to(accelerator.device)  # Low-res
                gt_images = batch["gt"].to(accelerator.device)  # High-res
                
                # Upsample input to match GT size for VAE encoding
                input_images_upsampled = torch.nn.functional.interpolate(
                    input_images,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False
                )
                
                # Encode images with VAE
                with torch.no_grad():
                    # Ensure input images match VAE dtype
                    input_images_encoded = input_images_upsampled.to(vae.dtype)
                    gt_images_encoded = gt_images.to(vae.dtype)
                    input_latents = vae.encode(input_images_encoded).latent_dist.sample()
                    gt_latents = vae.encode(gt_images_encoded).latent_dist.sample()
                    input_latents = input_latents * vae.config.scaling_factor
                    gt_latents = gt_latents * vae.config.scaling_factor
                
                # For super-resolution: use standard DDPM approach
                # Add noise to the CLEAN ground truth (high-res image), predict that noise
                # Condition on the low-res upsampled input to learn super-resolution
                
                # Sample timesteps - use full range for better learning
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (gt_latents.shape[0],),
                    device=gt_latents.device
                ).long()
                
                # Add noise to the CLEAN ground truth (standard diffusion training)
                # This simulates the super-resolution process: we'll learn to remove this noise
                noise = torch.randn_like(gt_latents)
                noisy_gt_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)
                
                # Blend low-res input with noisy GT for conditioning
                alpha = timesteps.float() / noise_scheduler.config.num_train_timesteps
                alpha = alpha.view(-1, 1, 1, 1)
                model_input = (1 - alpha) * input_latents + alpha * noisy_gt_latents
                
                # Prepare text embeddings (use a simple prompt for super-resolution)
                prompt = "high quality, detailed, sharp, high resolution photo"
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    text_embeddings = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                batch_size = gt_latents.shape[0]
                if text_embeddings.shape[0] == 1 and batch_size > 1:
                    text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
                
                model_input = model_input.to(unet.dtype)
                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings.to(unet.dtype)
                ).sample
                
                noise = noise.to(noise_pred.dtype)
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            train_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint (if save_steps > 0, save every N steps; if 0, save at epoch end; if -1, skip)
            if save_steps > 0 and global_step % save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save UNet
                    unet_to_save = accelerator.unwrap_model(unet)
                    unet_to_save.save_pretrained(checkpoint_dir / "unet")
                    
                    logger.info(f"Saved checkpoint at step {global_step}")
                    for handler in logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            handler.flush()
                    print(f"\nSaved checkpoint at step {global_step}")
        
        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")
        # Force flush to ensure log is written immediately
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        if save_steps == 0:
            if accelerator.is_main_process:
                checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save UNet
                unet_to_save = accelerator.unwrap_model(unet)
                unet_to_save.save_pretrained(checkpoint_dir / "unet")
                
                logger.info(f"Saved checkpoint at end of epoch {epoch+1}")
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
                print(f"\nSaved checkpoint at end of epoch {epoch+1}")
        
        # Run validation every epoch for consistent monitoring across all tasks
        if val_dataset is not None and accelerator.is_main_process:
            run_validation(epoch, val_dataset, pipeline, unet, output_dir, num_samples=2)
    
    # Save final model
    print("\nSaving final model...")
    logger.info("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Saving UNet...")
    unet_to_save = accelerator.unwrap_model(unet)
    unet_to_save.save_pretrained(final_dir / "unet")
    print("  UNet saved")
    
    # Save full pipeline
    print("  Saving full pipeline (this may take a few minutes)...")
    pipeline.unet = unet_to_save
    pipeline.save_pretrained(final_dir)
    print("  Pipeline saved")
    
    logger.info(f"Training complete! Model saved to: {final_dir}")
    logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Final flush
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    
    print(f"\nTraining complete! Model saved to: {final_dir}")
    print(f"To use the fine-tuned model, update inference.py to load from: {final_dir}")
    print(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion for super-resolution")
    parser.add_argument("--train_input", type=str, default="data/pairs/sr_x4/train/input",
                       help="Training input directory (low-resolution images)")
    parser.add_argument("--train_gt", type=str, default="data/pairs/sr_x4/train/gt",
                       help="Training ground truth directory (high-resolution images)")
    parser.add_argument("--val_input", type=str, default="data/pairs/sr_x4/val/input",
                       help="Validation input directory")
    parser.add_argument("--val_gt", type=str, default="data/pairs/sr_x4/val/gt",
                       help="Validation ground truth directory")
    parser.add_argument("--output_dir", type=str, default="outputs/models/super_resolution",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (use 1 if GPU memory is limited)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps (increase to 8+ for low memory)")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps (0 = only save at end of each epoch, -1 = only save final model)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size for training (256 for low memory, 512 for better quality)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit number of training samples for quick test (default: None, use all)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                       help="Limit number of validation samples for quick test (default: None, use all)")
    
    args = parser.parse_args()
    
    train_super_resolution(
        Path(args.train_input),
        Path(args.train_gt),
        Path(args.val_input),
        Path(args.val_gt),
        Path(args.output_dir),
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.gradient_accumulation_steps,
        args.save_steps,
        args.resume_from,
        args.image_size,
        args.max_train_samples,
        args.max_val_samples,
        args.base_model
    )

