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
        
        input_files = sorted(list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png")))
        self.files = []
        for f in input_files:
            gt_path_jpg = self.gt_dir / f"{f.stem}.jpg"
            gt_path_png = self.gt_dir / f"{f.stem}.png"
            if gt_path_jpg.exists() or gt_path_png.exists():
                self.files.append(f)
        
        if max_samples is not None and max_samples > 0:
            self.files = self.files[:max_samples]
        
        print(f"Found {len(self.files)} super-resolution pairs" + (f" (limited to {max_samples} for quick test)" if max_samples else ""))
        
        self.input_transform = transforms.Compose([
            transforms.Resize((image_size // 4, image_size // 4), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        input_path = self.files[idx]
        gt_path_jpg = self.gt_dir / f"{input_path.stem}.jpg"
        gt_path_png = self.gt_dir / f"{input_path.stem}.png"
        gt_path = gt_path_jpg if gt_path_jpg.exists() else gt_path_png
        
        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        
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
    base_model: str = "sd-legacy/stable-diffusion-v1-5",  # Base model to fine-tune from
    lambda_img: float = 0.05  # Weight for image-space L1 loss (0.0 to disable)
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
        force=True
    )
    logger = logging.getLogger(__name__)
    
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO)
    
    training_start_time = datetime.now()
    logger.info("="*60)
    logger.info("Fine-tuning Stable Diffusion for Super-Resolution")
    logger.info("="*60)
    logger.info(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Max train samples: {max_train_samples if max_train_samples else 'all'}")
    logger.info(f"Max val samples: {max_val_samples if max_val_samples else 'all'}")
    
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
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps * 0.05),  # 5% warmup
        num_training_steps=num_train_steps
    )
    
    # Track best model
    best_val_metric = -float("inf")  # PSNR (higher is better)
    best_checkpoint_dir = output_dir / "best"
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup metrics CSV file
    metrics_file = output_dir / "metrics.csv"
    if accelerator.is_main_process and not metrics_file.exists():
        with open(metrics_file, "w") as f:
            f.write("epoch,psnr,ssim,lpips,psnr_y,ssim_y,train_loss\n")
    
    train_dataset = SuperResolutionDataset(train_input_dir, train_gt_dir, image_size=image_size, max_samples=max_train_samples)
    val_dataset = SuperResolutionDataset(val_input_dir, val_gt_dir, image_size=image_size, max_samples=max_val_samples) if val_input_dir.exists() else None
    
    logger.info(f"Training dataset size: {len(train_dataset)} samples")
    logger.info(f"Validation dataset size: {len(val_dataset) if val_dataset else 0} samples")
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset) if val_dataset else 0} samples")
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    vae.eval()
    text_encoder.eval()
    
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    
    logger.info(f"DataLoader batches per epoch: {len(train_loader)}")
    logger.info(f"UNet parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,} trainable")
    logger.info(f"UNet input channels: {unet.config.in_channels}")
    logger.info(f"VAE scaling factor: {vae.config.scaling_factor}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Trainable UNet parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    def run_validation(epoch: int, val_dataset, pipeline, unet_model, output_dir: Path, num_samples: int = 4, image_size: int = None):
        """Run validation: sample images, compute metrics, save comparisons.
        
        Args:
            image_size: Target image size for upscaling (defaults to dataset's image_size)
        
        Returns:
            dict with keys 'psnr', 'ssim', 'lpips', 'psnr_y', 'ssim_y' (if available), or None if validation skipped
        """
        if val_dataset is None or len(val_dataset) == 0:
            return None
        
        if image_size is None:
            image_size = val_dataset.image_size if hasattr(val_dataset, 'image_size') else 256
        
        val_output_dir = output_dir / "val_samples"
        val_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample a few validation images
        num_samples = min(num_samples, len(val_dataset))
        sample_indices = np.linspace(0, len(val_dataset) - 1, num_samples, dtype=int)
        
        psnr_values = []
        ssim_values = []
        lpips_values = []
        psnr_y_values = []  # Y-channel PSNR
        ssim_y_values = []  # Y-channel SSIM
        
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
        
        # Helper function to compute Y-channel metrics (YCbCr Y channel)
        def compute_y_channel_metrics(result_np, gt_np):
            """Compute PSNR/SSIM on Y channel of YCbCr color space."""
            # Convert to YCbCr
            result_ycbcr = cv2.cvtColor(result_np, cv2.COLOR_RGB2YCrCb)
            gt_ycbcr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2YCrCb)
            
            # Extract Y channel (luminance)
            result_y = result_ycbcr[:, :, 0]
            gt_y = gt_ycbcr[:, :, 0]
            
            # Compute metrics on Y channel
            from skimage.metrics import peak_signal_noise_ratio as psnr
            from skimage.metrics import structural_similarity as ssim
            psnr_y = psnr(gt_y, result_y, data_range=255.0)
            ssim_y = ssim(gt_y, result_y, data_range=255.0)
            return psnr_y, ssim_y
        
        try:
            with torch.no_grad():
                for i, idx in enumerate(sample_indices):
                    sample = val_dataset[idx]
                    input_img = sample["input"]  # Low-res: (H/4, W/4) after transform
                    gt_img = sample["gt"]  # High-res: (H, W) after transform
                    
                    # Denormalize input for visualization
                    input_vis = (input_img + 1.0) / 2.0
                    input_vis = torch.clamp(input_vis, 0, 1)
                    input_pil = transforms.ToPILImage()(input_vis)
                    
                    # Ensure scale alignment: Low-res is (H/4, W/4), upscale to (H, W) before pipeline
                    # The input_pil is already at (H/4, W/4) from the dataset transform
                    # Pipeline expects input at target resolution, so we upscale with bicubic
                    lr_h, lr_w = input_pil.size
                    target_h, target_w = image_size, image_size
                    # Upscale low-res input to target resolution using bicubic interpolation
                    input_pil_upscaled = input_pil.resize((target_w, target_h), Image.BICUBIC)
                    
                    # Run inference with lower guidance for SR (reduces hallucination)
                    prompt = "high quality, detailed, sharp"
                    result = pipeline(
                        prompt=prompt,
                        image=input_pil_upscaled,
                        num_inference_steps=25,  # More steps for better detail
                        guidance_scale=3.5  # Lower guidance to reduce hallucinated textures
                    ).images[0]
                    
                    # Convert to numpy for metrics
                    # Pipeline output and GT should both be at (H, W) resolution
                    result_np = np.array(result)
                    gt_vis = (gt_img + 1.0) / 2.0
                    gt_vis = torch.clamp(gt_vis, 0, 1)
                    gt_np = (gt_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # Ensure result matches GT resolution (should already match, but double-check)
                    if result_np.shape[:2] != gt_np.shape[:2]:
                        result_np = cv2.resize(result_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_CUBIC)
                    
                    # Compute RGB metrics using MetricsCalculator (includes PSNR, SSIM, LPIPS)
                    try:
                        metrics = metrics_calc.calculate_all(result_np, gt_np)
                        psnr_rgb = metrics['psnr']
                        ssim_rgb = metrics['ssim']
                        psnr_values.append(psnr_rgb)
                        ssim_values.append(ssim_rgb)
                        if metrics.get('lpips') is not None:
                            lpips_values.append(metrics['lpips'])
                    except Exception as e:
                        try:
                            from skimage.metrics import peak_signal_noise_ratio as psnr
                            from skimage.metrics import structural_similarity as ssim
                            psnr_rgb = psnr(gt_np, result_np, data_range=255.0)
                            ssim_rgb = ssim(gt_np, result_np, data_range=255.0, channel_axis=2)
                            psnr_values.append(psnr_rgb)
                            ssim_values.append(ssim_rgb)
                        except ImportError:
                            psnr_rgb = None
                            ssim_rgb = None
                    
                    if psnr_rgb is not None:
                        try:
                            psnr_y, ssim_y = compute_y_channel_metrics(result_np, gt_np)
                            psnr_y_values.append(psnr_y)
                            ssim_y_values.append(ssim_y)
                        except Exception as e:
                            logger.warning(f"Failed to compute Y-channel metrics for sample {idx}: {e}")
                    
                    input_np = (input_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if input_np.shape[:2] != gt_np.shape[:2]:
                        input_np = cv2.resize(input_np, (gt_np.shape[1], gt_np.shape[0]))
                    comparison = np.hstack([input_np, result_np, gt_np])
                    comparison_pil = Image.fromarray(comparison)
                    
                    save_path = val_output_dir / f"epoch_{epoch+1}_sample_{i+1}_idx{idx}.png"
                    comparison_pil.save(save_path)
        finally:
            pipeline.unet.train()
            torch.cuda.empty_cache()
        
        # Log metrics and return
        if psnr_values:
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            metric_str = f"Validation (epoch {epoch+1}): PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}"
            
            result = {"psnr": avg_psnr, "ssim": avg_ssim}
            
            if lpips_values:
                avg_lpips = np.mean(lpips_values)
                metric_str += f", LPIPS={avg_lpips:.4f}"
                result["lpips"] = avg_lpips
            
            if psnr_y_values:
                avg_psnr_y = np.mean(psnr_y_values)
                avg_ssim_y = np.mean(ssim_y_values)
                metric_str += f"\n  Y-channel: PSNR={avg_psnr_y:.2f} dB, SSIM={avg_ssim_y:.4f}"
                result["psnr_y"] = avg_psnr_y
                result["ssim_y"] = avg_ssim_y
            
            logger.info(metric_str)
            print(metric_str)
            return result
        return None
    
    # Training loop
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Total steps: {num_train_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Image size: {image_size}")
    
    # Pre-compute text embeddings once (reused for all batches)
    prompt = "high quality, detailed, sharp, high resolution photo"
    logger.info("Pre-computing text embeddings...")
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        encoder_output = text_encoder(text_inputs.input_ids.to(accelerator.device))
        if hasattr(encoder_output, 'last_hidden_state'):
            cached_text_embeddings = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
            cached_text_embeddings = encoder_output[0]
        else:
            cached_text_embeddings = encoder_output
    
    global_step = 0
    
    for epoch in range(num_epochs):
        unet.train()
        train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            with accelerator.accumulate(unet):
                input_images = batch["input"].to(accelerator.device)
                gt_images = batch["gt"].to(accelerator.device)
                
                input_images_upsampled = torch.nn.functional.interpolate(
                    input_images,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False
                )
                
                with torch.no_grad():
                    input_images_encoded = input_images_upsampled.to(vae.dtype)
                    gt_images_encoded = gt_images.to(vae.dtype)
                    input_latents = vae.encode(input_images_encoded).latent_dist.sample()
                    gt_latents = vae.encode(gt_images_encoded).latent_dist.sample()
                    input_latents = input_latents * vae.config.scaling_factor
                    gt_latents = gt_latents * vae.config.scaling_factor
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (gt_latents.shape[0],),
                    device=gt_latents.device
                ).long()
                
                noise = torch.randn_like(gt_latents)
                noisy_gt_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)
                
                # Uses "soft conditioning" via latent blending
                alpha = timesteps.float() / noise_scheduler.config.num_train_timesteps
                alpha = alpha.view(-1, 1, 1, 1)
                model_input = (1 - alpha) * input_latents + alpha * noisy_gt_latents
                
                # Reuse pre-computed text embeddings (expand for batch size if needed)
                batch_size = model_input.shape[0]
                text_embeddings = cached_text_embeddings
                if text_embeddings.shape[0] == 1 and batch_size > 1:
                    text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
                
                model_input = model_input.to(unet.dtype)
                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings.to(unet.dtype)
                ).sample
                
                noise = noise.to(noise_pred.dtype)
                noise_pred_loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                img_loss = 0.0
                if lambda_img > 0.0:
                    alpha_cumprod_t = noise_scheduler.alphas_cumprod[timesteps]
                    alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1).to(noisy_gt_latents.device)
                    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
                    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod_t)
                    
                    pred_clean_latents = (noisy_gt_latents - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
                    
                    with torch.no_grad():
                        pred_clean_latents_scaled = pred_clean_latents / vae.config.scaling_factor
                        pred_img = vae.decode(pred_clean_latents_scaled.to(vae.dtype)).sample
                        pred_img = (pred_img + 1.0) / 2.0
                    
                    gt_images_normalized = (gt_images + 1.0) / 2.0
                    img_loss = torch.mean(torch.abs(pred_img - gt_images_normalized))
                
                loss = noise_pred_loss + lambda_img * img_loss
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            train_loss += loss.detach().item()
            global_step += 1
            num_batches += 1
            
            # Show running average loss in progress bar
            avg_loss_so_far = train_loss / max(1, num_batches)
            progress_bar.set_postfix({"loss": f"{avg_loss_so_far:.4f}", "batch_loss": f"{loss.detach().item():.4f}"})
            
            # Save checkpoint (if save_steps > 0, save every N steps; if 0, save at epoch end; if -1, skip)
            if save_steps > 0 and global_step % save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    unet_to_save = accelerator.unwrap_model(unet)
                    unet_to_save.save_pretrained(checkpoint_dir / "unet")
                    
                    logger.info(f"Saved checkpoint at step {global_step}")
                    for handler in logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            handler.flush()
                    print(f"\nSaved checkpoint at step {global_step}")
        
        avg_loss = train_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}, Processed {len(train_loader)} batches")
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        
        if save_steps == 0:
            if accelerator.is_main_process:
                checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                unet_to_save = accelerator.unwrap_model(unet)
                unet_to_save.eval()
                unet_to_save.save_pretrained(
                    checkpoint_dir / "unet",
                    safe_serialization=True,
                    is_main_process=True
                )
                
                logger.info(f"Saved checkpoint at end of epoch {epoch+1} to {checkpoint_dir}")
                logger.info(f"Checkpoint contains: UNet weights and config")
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
                print(f"\nSaved checkpoint at end of epoch {epoch+1}")
        
        # Run validation every epoch for consistent monitoring across all tasks
        if val_dataset is not None and accelerator.is_main_process:
            val_stats = run_validation(epoch, val_dataset, pipeline, unet, output_dir, num_samples=2, image_size=image_size)
            if val_stats is not None:
                psnr = val_stats["psnr"]
                # Check if this is the best model so far
                if psnr > best_val_metric:
                    best_val_metric = psnr
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unet_to_save = accelerator.unwrap_model(unet)
                        pipeline.unet = unet_to_save
                        save_dir = best_checkpoint_dir
                        pipeline.save_pretrained(save_dir)
                        logger.info(f"New best model (PSNR={psnr:.2f} dB) saved to: {save_dir}")
                        print(f"New best model (PSNR={psnr:.2f} dB) saved to: {save_dir}")
                
                # Log metrics to CSV
                with open(metrics_file, "a") as f:
                    lpips_val = val_stats.get('lpips', float('nan'))
                    psnr_y_val = val_stats.get('psnr_y', float('nan'))
                    ssim_y_val = val_stats.get('ssim_y', float('nan'))
                    f.write(f"{epoch+1},{val_stats['psnr']:.4f},{val_stats['ssim']:.4f},"
                            f"{lpips_val:.4f},{psnr_y_val:.4f},{ssim_y_val:.4f},{avg_loss:.6f}\n")
    
    if accelerator.is_main_process:
        print("\nSaving final model...")
        logger.info("Saving final model...")
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        print("  Saving UNet...")
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.eval()
        
        was_checkpointing = False
        if hasattr(unet_to_save, "gradient_checkpointing") and unet_to_save.gradient_checkpointing:
            was_checkpointing = True
            unet_to_save.disable_gradient_checkpointing()
        
        unet_dir = final_dir / "unet"
        unet_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            unet_to_save.save_pretrained(
                str(unet_dir),
                safe_serialization=True,
                is_main_process=True
            )
            logger.info(f"UNet config saved to {unet_dir}")
        except Exception as e:
            logger.warning(f"Failed to save UNet config via save_pretrained: {e}")
        
        print("  Saving UNet weights directly...")
        try:
            from safetensors.torch import save_file
            import json
            
            state_dict = unet_to_save.state_dict()
            weight_path = unet_dir / "diffusion_pytorch_model.safetensors"
            
            save_file(state_dict, str(weight_path))
            
            if weight_path.exists() and weight_path.stat().st_size > 0:
                file_size_mb = weight_path.stat().st_size / (1024 * 1024)
                print(f"  UNet weights saved successfully: {weight_path.name} ({file_size_mb:.2f} MB)")
                logger.info(f"UNet weights saved: {weight_path} ({file_size_mb:.2f} MB)")
            else:
                raise RuntimeError(f"Weight file {weight_path} was not created or is empty")
            
            config_path = unet_dir / "config.json"
            if not config_path.exists():
                logger.warning("config.json missing, saving it now...")
                config_dict = unet_to_save.config.to_dict()
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"UNet config.json saved to {config_path}")
                
        except ImportError:
            logger.error("safetensors library not available - cannot save UNet weights")
            raise RuntimeError("safetensors library is required to save UNet weights")
        except Exception as e:
            logger.error(f"Failed to save UNet weights: {e}", exc_info=True)
            print(f"  ERROR: Failed to save UNet weights: {e}")
            raise
        finally:
            if was_checkpointing and hasattr(unet_to_save, "enable_gradient_checkpointing"):
                unet_to_save.enable_gradient_checkpointing()
        
        print("  Saving full pipeline (this may take a few minutes)...")
        pipeline.unet = unet_to_save
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.save_pretrained(
            final_dir,
            safe_serialization=True,
            is_main_process=True
        )
        print("  Pipeline saved (UNet, VAE, text_encoder, tokenizer, scheduler)")
        
        training_end_time = datetime.now()
        training_duration = training_end_time - training_start_time
        logger.info(f"Training complete! Model saved to: {final_dir}")
        logger.info(f"Final model contains: UNet, VAE, text_encoder, tokenizer, scheduler")
        logger.info(f"Training completed at: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total training duration: {training_duration}")
        logger.info(f"Average time per epoch: {training_duration / num_epochs}")
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
    parser.add_argument("--lambda_img", type=float, default=0.05,
                       help="Weight for image-space L1 loss (0.0 to disable, default: 0.05)")
    parser.add_argument("--base_model", type=str, default="sd-legacy/stable-diffusion-v1-5",
                       help="Base model to fine-tune from. Options: "
                            "'sd-legacy/stable-diffusion-v1-5' (default, publicly accessible), "
                            "'stabilityai/stable-diffusion-xl-base-1.0' (better quality, 3.5B parameters, publicly accessible).")
    
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
        args.base_model,
        args.lambda_img
    )

