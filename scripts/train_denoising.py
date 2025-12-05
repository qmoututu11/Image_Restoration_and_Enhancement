#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion for denoising on your synthetic pairs.
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
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from datetime import datetime
from src.metrics import MetricsCalculator


class DenoisingDataset(Dataset):
    """Dataset for denoising: noisy input -> clean output."""
    
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
        
        print(f"Found {len(self.files)} denoising pairs" + (f" (limited to {max_samples} for quick test)" if max_samples else ""))
        
        # Transform: resize to image_size, convert to tensor, normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.LANCZOS),
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
        input_tensor = self.transform(input_img)
        gt_tensor = self.transform(gt_img)
        
        return {
            "input": input_tensor,
            "gt": gt_tensor
        }


def train_denoising(
    train_input_dir: Path,
    train_gt_dir: Path,
    val_input_dir: Path,
    val_gt_dir: Path,
    output_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 5e-6,  # Lower LR for fine-tuning
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    resume_from: str = None,
    image_size: int = 256,  # Reduced from 512 to save memory
    max_train_samples: int = None,  # Limit training samples for quick test
    max_val_samples: int = None,  # Limit validation samples for quick test
    base_model: str = "runwayml/stable-diffusion-v1-5"  # Base model to fine-tune
):
    """Fine-tune Stable Diffusion for denoising."""
    
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
    training_start_time = datetime.now()
    logger.info("="*60)
    logger.info("Fine-tuning Stable Diffusion for Denoising")
    logger.info("="*60)
    logger.info(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Max train samples: {max_train_samples if max_train_samples else 'all'}")
    logger.info(f"Max val samples: {max_val_samples if max_val_samples else 'all'}")
    # Force initial flush to verify file writing works
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    
    print("="*60)
    print("Fine-tuning Stable Diffusion for Denoising")
    print("="*60)
    print(f"Log file: {log_file.absolute()}")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Clear GPU cache before starting and verify GPU availability
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
        print("Make sure you selected GPU runtime in Colab: Runtime -> Change runtime type -> GPU")
    
    # Setup accelerator
    # Use "no" mixed precision when model is already in fp16 to avoid gradient scaling issues
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="no"  # Disable since model is loaded in fp16
    )
    
    # Verify accelerator device
    if torch.cuda.is_available() and accelerator.device.type != "cuda":
        logger.warning(f"CUDA available but accelerator using {accelerator.device.type}")
        print(f"WARNING: CUDA available but accelerator using {accelerator.device.type}")
    
    # Enable cuDNN benchmark for faster training (if input size is constant)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Load pre-trained model
        logger.info("Loading pre-trained Stable Diffusion Img2Img...")
        print("Loading pre-trained Stable Diffusion Img2Img...")
        model_id = base_model
        logger.info(f"Using base model: {model_id}")
        print(f"Using base model: {model_id}")
        
        # Check for HF token
        hf_token = os.getenv("HF_TOKEN", None)
        if hf_token:
            logger.info("Hugging Face token found in environment")
            print("Hugging Face token detected")
        else:
            logger.info("No HF_TOKEN found - using public access")
            print("No HF_TOKEN set - using public model access")
        
        # Check if using SD-XL model
        is_sdxl = "xl" in model_id.lower() or "stable-diffusion-xl" in model_id.lower()
        
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            print(f"Resuming from checkpoint: {resume_from}")
            # Try to detect if checkpoint is SD-XL or SD v1.5
            try:
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(resume_from)
                is_sdxl = True
            except:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(resume_from)
                is_sdxl = False
        else:
            # Get Hugging Face token from environment variable
            hf_token = os.getenv("HF_TOKEN", None)
            
            # Load appropriate pipeline based on model
            if is_sdxl:
                logger.info("Detected SD-XL model, using StableDiffusionXLImg2ImgPipeline")
                print("Using SD-XL pipeline")
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if torch.cuda.is_available() else None,
                    token=hf_token,
                )
            else:
                logger.info("Using standard SD v1.5 pipeline")
                print("Using SD v1.5 pipeline")
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=False,
                    token=hf_token,  # Use token if available, None for public models
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
        
        # Handle SD-XL (dual text encoders) vs SD v1.5 (single text encoder)
        if is_sdxl:
            text_encoder = pipeline.text_encoder
            text_encoder_2 = pipeline.text_encoder_2
            tokenizer = pipeline.tokenizer
            tokenizer_2 = pipeline.tokenizer_2
            # Freeze both text encoders
            text_encoder.requires_grad_(False)
            text_encoder_2.requires_grad_(False)
        else:
            text_encoder = pipeline.text_encoder
            tokenizer = pipeline.tokenizer
            text_encoder_2 = None
            tokenizer_2 = None
            text_encoder.requires_grad_(False)
        
        # Freeze VAE and train only UNet
        vae.requires_grad_(False)
        unet.requires_grad_(True)
        
        # Enable gradient checkpointing to save memory
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing to save memory")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Setup scheduler
        # Calculate steps based on actual dataset size
        train_dataset_temp = DenoisingDataset(train_input_dir, train_gt_dir, image_size=image_size)
        num_train_steps = max(
            1,
            len(train_dataset_temp) * num_epochs // (batch_size * gradient_accumulation_steps)
        )
        # Use cosine schedule with warmup for better convergence
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=int(num_train_steps * 0.1),  # 10% warmup
            num_training_steps=num_train_steps
        )
        
        # Setup datasets
        train_dataset = DenoisingDataset(train_input_dir, train_gt_dir, image_size=image_size, max_samples=max_train_samples)
        val_dataset = DenoisingDataset(val_input_dir, val_gt_dir, image_size=image_size, max_samples=max_val_samples) if val_input_dir.exists() else None
        
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
        
        logger.info(f"DataLoader batches per epoch: {len(train_loader)}")
        print(f"Batches per epoch: {len(train_loader)}")
        
        # Setup noise scheduler
        noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        
        vae = vae.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)
        if is_sdxl and text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(accelerator.device)
        
        vae.eval()
        text_encoder.eval()
        if is_sdxl and text_encoder_2 is not None:
            text_encoder_2.eval()
        
        # Prepare with accelerator
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )
        
        # Log model configuration
        logger.info(f"UNet parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,} trainable")
        logger.info(f"UNet input channels: {unet.config.in_channels}")
        logger.info(f"UNet output channels: {unet.config.out_channels}")
        logger.info(f"VAE scaling factor: {vae.config.scaling_factor}")
        logger.info(f"Text encoder hidden size: {text_encoder.config.hidden_size}")
        if is_sdxl and text_encoder_2 is not None:
            logger.info(f"Text encoder 2 hidden size: {text_encoder_2.config.hidden_size}")
        print(f"Trainable UNet parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
        # Validation function
        def run_validation(epoch: int, val_dataset, pipeline, unet_model, output_dir: Path, num_samples: int = 4, is_sdxl_model: bool = False):
            """Run validation: sample images, compute metrics, save comparisons."""
            if val_dataset is None or len(val_dataset) == 0:
                logger.warning("Validation dataset is None or empty, skipping validation")
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
            print("Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]")
            sys.stdout.flush()
            device_for_metrics = accelerator.device if torch.cuda.is_available() else "cpu"
            metrics_calc = MetricsCalculator(use_lpips=True, use_fid=False, device=device_for_metrics)
            
            # Temporarily update pipeline with current UNet
            pipeline.unet = accelerator.unwrap_model(unet_model)
            
            # All components are already on GPU (set at initialization)
            # Just ensure pipeline is on the correct device
            device = accelerator.device
            
            # Only move if not already on correct device (shouldn't happen, but safety check)
            unet_device = next(pipeline.unet.parameters()).device
            vae_device = next(pipeline.vae.parameters()).device
            text_encoder_device = next(pipeline.text_encoder.parameters()).device
            if unet_device != device:
                pipeline.unet = pipeline.unet.to(device)
            if vae_device != device:
                pipeline.vae = pipeline.vae.to(device)
            if text_encoder_device != device:
                pipeline.text_encoder = pipeline.text_encoder.to(device)
            # Handle SD-XL second text encoder
            if is_sdxl_model and hasattr(pipeline, 'text_encoder_2'):
                text_encoder_2_device = next(pipeline.text_encoder_2.parameters()).device
                if text_encoder_2_device != device:
                    pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)
            
            # CRITICAL: Set pipeline's internal device state
            # This ensures pipeline.encode_prompt() and other internal methods use the correct device
            pipeline = pipeline.to(device)
            
            # Disable safety checker for validation (it's blocking legitimate images)
            # More aggressive disabling - set to None and also disable in config
            if hasattr(pipeline, 'safety_checker'):
                pipeline.safety_checker = None
            if hasattr(pipeline, 'feature_extractor'):
                pipeline.feature_extractor = None
            # Also disable in pipeline config if it exists
            if hasattr(pipeline, 'safety_checker') and hasattr(pipeline, '_safety_checker'):
                pipeline._safety_checker = None
            # Disable safety checker requirement
            if hasattr(pipeline, 'requires_safety_checker'):
                pipeline.requires_safety_checker = False
            
            # Set UNet to eval mode
            pipeline.unet.eval()
            pipeline.vae.eval()
            pipeline.text_encoder.eval()
            if is_sdxl_model and hasattr(pipeline, 'text_encoder_2'):
                pipeline.text_encoder_2.eval()
            
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
                        
                        # Run inference with lower strength for denoising
                        # Lower strength = more preservation of input structure, less generation
                        prompt = "a photograph, high quality, detailed, sharp"
                        result = pipeline(
                            prompt=prompt,
                            image=input_pil,
                            strength=0.3,  # Lower strength for denoising (preserve more input)
                            num_inference_steps=20,  # Fewer steps for faster validation
                            guidance_scale=5.0  # Lower guidance for less hallucination
                        ).images[0]
                        
                        # Check if result is black (safety checker might have blocked it)
                        result_np_check = np.array(result)
                        if result_np_check.sum() < 1000:  # Very dark/black image (threshold adjusted)
                            logger.warning(f"Sample {idx} produced dark output (sum={result_np_check.sum()}) - safety checker may still be active")
                        
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
                        
                        # Create comparison image: [Input | Model Output | Ground Truth]
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
                sys.stdout.flush()
        
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
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                with accelerator.accumulate(unet):
                    # Get images
                    input_images = batch["input"].to(accelerator.device)
                    gt_images = batch["gt"].to(accelerator.device)
                    
                    # Encode images with VAE (always on GPU)
                    with torch.no_grad():
                        # Ensure input images match VAE dtype
                        input_images_encoded = input_images.to(vae.dtype)
                        gt_images_encoded = gt_images.to(vae.dtype)
                        input_latents = vae.encode(input_images_encoded).latent_dist.sample()
                        gt_latents = vae.encode(gt_images_encoded).latent_dist.sample()
                        input_latents = input_latents * vae.config.scaling_factor
                        gt_latents = gt_latents * vae.config.scaling_factor
                    
                    # For denoising: train model to predict noise that needs to be removed
                    # Standard diffusion training: add noise to clean GT, predict that noise
                    # But condition on the noisy input to learn denoising transformation
                    
                    # Sample timesteps - use full range for better learning
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (gt_latents.shape[0],),
                        device=gt_latents.device
                    ).long()
                    
                    # Add noise to the CLEAN ground truth (standard diffusion training)
                    # This simulates the denoising process: we'll learn to remove this noise
                    noise = torch.randn_like(gt_latents)
                    noisy_gt_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)
                    
                    # Blend noisy input with noisy GT for conditioning
                    alpha = timesteps.float() / noise_scheduler.config.num_train_timesteps
                    alpha = alpha.view(-1, 1, 1, 1)
                    model_input = (1 - alpha) * input_latents + alpha * noisy_gt_latents
                    
                    prompt = "clean high quality photo, no noise, sharp details"
                    added_cond_kwargs = None
                    with torch.no_grad():
                        if is_sdxl and text_encoder_2 is not None:
                            # SD-XL: encode with both text encoders
                            text_inputs = tokenizer(
                                prompt,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            text_inputs_2 = tokenizer_2(
                                prompt,
                                padding="max_length",
                                max_length=tokenizer_2.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            # Extract sequence embeddings from both encoders
                            encoder_output_1 = text_encoder(text_inputs.input_ids.to(accelerator.device))
                            encoder_output_2 = text_encoder_2(text_inputs_2.input_ids.to(accelerator.device))
                            
                            if hasattr(encoder_output_1, 'last_hidden_state'):
                                prompt_embeds = encoder_output_1.last_hidden_state  # (batch, seq_len, 768)
                            elif isinstance(encoder_output_1, tuple):
                                prompt_embeds = encoder_output_1[0]  # (batch, seq_len, 768)
                            else:
                                prompt_embeds = encoder_output_1  # (batch, seq_len, 768)
                            
                            if hasattr(encoder_output_2, 'last_hidden_state'):
                                prompt_embeds_2 = encoder_output_2.last_hidden_state  # (batch, seq_len, 1280)
                                # Get pooled embeddings for added_cond_kwargs
                                if hasattr(encoder_output_2, 'pooler_output') and encoder_output_2.pooler_output is not None:
                                    pooled_prompt_embeds = encoder_output_2.pooler_output  # (batch, 1280)
                                else:
                                    pooled_prompt_embeds = None
                            elif isinstance(encoder_output_2, tuple):
                                prompt_embeds_2 = encoder_output_2[0]  # (batch, seq_len, 1280)
                                pooled_prompt_embeds = encoder_output_2[1] if len(encoder_output_2) > 1 else None
                            else:
                                prompt_embeds_2 = encoder_output_2  # (batch, seq_len, 1280)
                                pooled_prompt_embeds = None
                            
                            # Ensure both have the same sequence length
                            seq_len_1 = prompt_embeds.shape[1]
                            seq_len_2 = prompt_embeds_2.shape[1]
                            if seq_len_1 != seq_len_2:
                                min_len = min(seq_len_1, seq_len_2)
                                if seq_len_1 > min_len:
                                    prompt_embeds = prompt_embeds[:, :min_len, :]
                                if seq_len_2 > min_len:
                                    prompt_embeds_2 = prompt_embeds_2[:, :min_len, :]
                            
                            text_embeddings = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                            
                            # Prepare added_cond_kwargs for SD-XL UNet
                            if pooled_prompt_embeds is None:
                                pooled_prompt_embeds = prompt_embeds_2.mean(dim=1)
                            
                            if pooled_prompt_embeds is None or pooled_prompt_embeds.numel() == 0:
                                batch_size = text_embeddings.shape[0]
                                pooled_prompt_embeds = torch.zeros(batch_size, 1280, dtype=text_embeddings.dtype, device=text_embeddings.device)
                            
                            batch_size = text_embeddings.shape[0]
                            time_ids = torch.tensor(
                                [[image_size, image_size, 0, 0, image_size, image_size]],
                                dtype=text_embeddings.dtype,
                                device=text_embeddings.device
                            ).repeat(batch_size, 1)
                            
                            if pooled_prompt_embeds.shape[0] == 1 and batch_size > 1:
                                pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
                            
                            added_cond_kwargs = {
                                "text_embeds": pooled_prompt_embeds.to(unet.dtype),
                                "time_ids": time_ids.to(unet.dtype)
                            }
                        else:
                            text_inputs = tokenizer(
                                prompt,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            encoder_output = text_encoder(text_inputs.input_ids.to(accelerator.device))
                            if hasattr(encoder_output, 'last_hidden_state'):
                                text_embeddings = encoder_output.last_hidden_state
                            elif isinstance(encoder_output, tuple):
                                text_embeddings = encoder_output[0]
                            else:
                                text_embeddings = encoder_output
                    
                    # Expand to match batch size if needed
                    batch_size = model_input.shape[0]
                    if text_embeddings.shape[0] == 1 and batch_size > 1:
                        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
                    
                    # Predict the noise that was added to GT (conditioned on noisy input)
                    # Model learns: given noisy input, predict noise to remove from noisy GT
                    model_input = model_input.to(unet.dtype)
                    
                    # SD-XL UNet requires added_cond_kwargs, SD v1.5 does not
                    if is_sdxl and added_cond_kwargs is not None:
                        noise_pred = unet(
                            model_input,
                            timesteps,
                            encoder_hidden_states=text_embeddings.to(unet.dtype),
                            added_cond_kwargs=added_cond_kwargs
                        ).sample
                    else:
                        noise_pred = unet(
                            model_input,
                            timesteps,
                            encoder_hidden_states=text_embeddings.to(unet.dtype)
                        ).sample
                    
                    noise = noise.to(noise_pred.dtype)
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss detected (NaN/Inf) at step {global_step}, skipping...")
                        optimizer.zero_grad()
                        continue
                    
                    # Backward pass
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    if global_step % 20 == 0:
                        torch.cuda.empty_cache()
                
                train_loss += loss.item()
                global_step += 1
                num_batches += 1
                
                # Show running average loss in progress bar
                avg_loss_so_far = train_loss / num_batches
                progress_bar.set_postfix({"loss": f"{avg_loss_so_far:.4f}", "batch_loss": f"{loss.item():.4f}"})
            
            # Save checkpoint (if save_steps > 0, save every N steps; if -1, skip)
            if save_steps > 0 and global_step % save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save UNet
                    unet_to_save = accelerator.unwrap_model(unet)
                    unet_to_save.eval()
                    
                    checkpoint_unet_dir = checkpoint_dir / "unet"
                    checkpoint_unet_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save config
                    try:
                        unet_to_save.save_pretrained(
                            str(checkpoint_unet_dir),
                            safe_serialization=True,
                            is_main_process=True
                        )
                    except Exception:
                        pass  # Will save config manually if needed
                    
                    # Always save weights directly
                    from safetensors.torch import save_file
                    import json
                    state_dict = unet_to_save.state_dict()
                    weight_path = checkpoint_unet_dir / "diffusion_pytorch_model.safetensors"
                    save_file(state_dict, str(weight_path))
                    
                    # Ensure config exists
                    config_path = checkpoint_unet_dir / "config.json"
                    if not config_path.exists():
                        config_dict = unet_to_save.config.to_dict()
                        with open(config_path, 'w') as f:
                            json.dump(config_dict, f, indent=2)
                    
                    logger.info(f"Saved checkpoint at step {global_step} to {checkpoint_dir}")
                    logger.info(f"Checkpoint contains: UNet weights and config")
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
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")
            logger.info(f"Epoch {epoch+1} processed {len(train_loader)} batches")
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            sys.stdout.flush()
            
            # Run validation every epoch for consistent monitoring across all tasks
            if val_dataset is not None and accelerator.is_main_process:
                run_validation(epoch, val_dataset, pipeline, unet, output_dir, num_samples=2, is_sdxl_model=is_sdxl)
            
            # Save checkpoint at end of epoch (if save_steps == 0)
            if save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save UNet
                    unet_to_save = accelerator.unwrap_model(unet)
                    unet_to_save.eval()
                    
                    checkpoint_unet_dir = checkpoint_dir / "unet"
                    checkpoint_unet_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save config
                    try:
                        unet_to_save.save_pretrained(
                            str(checkpoint_unet_dir),
                            safe_serialization=True,
                            is_main_process=True
                        )
                    except Exception:
                        pass  # Will save config manually if needed
                    
                    # Always save weights directly
                    from safetensors.torch import save_file
                    import json
                    state_dict = unet_to_save.state_dict()
                    weight_path = checkpoint_unet_dir / "diffusion_pytorch_model.safetensors"
                    save_file(state_dict, str(weight_path))
                    
                    # Ensure config exists
                    config_path = checkpoint_unet_dir / "config.json"
                    if not config_path.exists():
                        config_dict = unet_to_save.config.to_dict()
                        with open(config_path, 'w') as f:
                            json.dump(config_dict, f, indent=2)
                    
                    logger.info(f"Saved checkpoint at end of epoch {epoch+1} to {checkpoint_dir}")
                    logger.info(f"Checkpoint contains: UNet weights and config")
                    for handler in logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            handler.flush()
                    print(f"\nSaved checkpoint at end of epoch {epoch+1}")
        
        # Save final model (OUTSIDE the training loop - only once after all epochs)
        print("\nSaving final model...")
        logger.info("Saving final model...")
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        print("  Saving UNet...")
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.eval()
        
        # Disable gradient checkpointing temporarily for saving (if enabled)
        was_checkpointing = False
        if hasattr(unet_to_save, "gradient_checkpointing") and unet_to_save.gradient_checkpointing:
            was_checkpointing = True
            unet_to_save.disable_gradient_checkpointing()
        
        # Save UNet with safetensors format (keep on GPU, no need to move to CPU)
        unet_dir = final_dir / "unet"
        unet_dir.mkdir(parents=True, exist_ok=True)
        
        # First, save config.json using save_pretrained (this always works)
        try:
            unet_to_save.save_pretrained(
                str(unet_dir),
                safe_serialization=True,
                is_main_process=True
            )
            logger.info(f"UNet config saved to {unet_dir}")
        except Exception as e:
            logger.warning(f"Failed to save UNet config via save_pretrained: {e}")
            # Continue anyway - we'll save config manually if needed
        
        # Always save weights directly using safetensors (more reliable)
        print("  Saving UNet weights directly...")
        try:
            from safetensors.torch import save_file
            import json
            
            # Get state dict
            state_dict = unet_to_save.state_dict()
            weight_path = unet_dir / "diffusion_pytorch_model.safetensors"
            
            # Save weights
            save_file(state_dict, str(weight_path))
            
            # Verify file was created and has size > 0
            if weight_path.exists() and weight_path.stat().st_size > 0:
                file_size_mb = weight_path.stat().st_size / (1024 * 1024)
                print(f"  UNet weights saved successfully: {weight_path.name} ({file_size_mb:.2f} MB)")
                logger.info(f"UNet weights saved: {weight_path} ({file_size_mb:.2f} MB)")
            else:
                raise RuntimeError(f"Weight file {weight_path} was not created or is empty")
            
            # Ensure config.json exists (save it if save_pretrained didn't work)
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
            # Re-enable gradient checkpointing if it was enabled
            if was_checkpointing and hasattr(unet_to_save, "enable_gradient_checkpointing"):
                unet_to_save.enable_gradient_checkpointing()
        
        # Save full pipeline (all components: UNet, VAE, text_encoder, tokenizer, scheduler)
        print("  Saving full pipeline (this may take a few minutes)...")
        pipeline.unet = unet_to_save
        # Ensure all components are in eval mode before saving
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        if hasattr(pipeline, 'text_encoder_2'):
            pipeline.text_encoder_2.eval()
        # Save entire pipeline with all components
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
        # Final flush
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        
        print(f"\nTraining complete! Model saved to: {final_dir}")
        print(f"To use the fine-tuned model, update inference.py to load from: {final_dir}")
        print(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion for denoising")
    parser.add_argument("--train_input", type=str, default="data/pairs/denoise/train/input",
                       help="Training input directory (noisy images)")
    parser.add_argument("--train_gt", type=str, default="data/pairs/denoise/train/gt",
                       help="Training ground truth directory (clean images)")
    parser.add_argument("--val_input", type=str, default="data/pairs/denoise/val/input",
                       help="Validation input directory")
    parser.add_argument("--val_gt", type=str, default="data/pairs/denoise/val/gt",
                       help="Validation ground truth directory")
    parser.add_argument("--output_dir", type=str, default="outputs/models/denoising",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (use 1 if GPU memory is limited)")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate (default: 5e-6 for fine-tuning)")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base model to fine-tune from. Options: "
                            "'runwayml/stable-diffusion-v1-5' (default, recommended - good quality, fast training, publicly accessible), "
                            "'stabilityai/stable-diffusion-xl-base-1.0'")
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
    
    train_denoising(
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

