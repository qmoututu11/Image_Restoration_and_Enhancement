#!/usr/bin/env python3
"""
Inference pipelines for image restoration tasks using Hugging Face models.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import sys
import warnings
import logging
from typing import Literal, Any
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Type aliases for better type hints
Task = Literal["denoise", "sr", "super_resolution", "colorize", "inpaint"]

try:
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionUpscalePipeline
    )
except ImportError:
    print("Missing dependencies. Install: pip install diffusers transformers huggingface_hub", file=sys.stderr)
    sys.exit(1)


class RestorationPipeline:
    """Unified pipeline for image restoration tasks using Hugging Face models."""
    
    def __init__(self, device: str = "auto", config: dict | None = None, seed: int = 42):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.models: dict[str, object] = {}
        self.seed = seed
        logger.info(f"Using device: {self.device} ({self.dtype}), seed: {seed}")
        
        default_config = {
            "denoise": {
                "fine_tuned_dir": "outputs/models/denoising/final",
                "pretrained_id": "runwayml/stable-diffusion-v1-5",
                "default_backend": "auto",  # "auto" | "diffusion" | "opencv"
            },
            "sr": {
                "fine_tuned_dir": "outputs/models/super_resolution/final",
                "pretrained_id": "stabilityai/stable-diffusion-x4-upscaler",
                "default_backend": "auto",  # "auto" | "sd_upscaler" | "realesrgan" | "lanczos"
            },
            "colorize": {
                "fine_tuned_dir": "outputs/models/colorization/final",
                "pretrained_id": "runwayml/stable-diffusion-v1-5",
            },
            "inpaint": {
                "fine_tuned_dir": "outputs/models/inpainting/final",
                "pretrained_id": "runwayml/stable-diffusion-inpainting",
            },
        }
        
        self.config = default_config if config is None else {**default_config, **config}
        
        # Default prompts for each task
        self.prompts = {
            "denoise": "clean high quality photo, no noise, sharp details",
            "sr": "high quality, detailed, sharp",
            "colorize": "realistic natural colors, high quality photo, detailed",
            "inpaint": "high quality detailed photo",
        }
    
    def _load_sd_pipeline(
        self,
        pipe_class,
        model_path: str,
        task_name: str,
        fine_tuned_path: Path | None = None,
    ):
        """
        Helper method to load Stable Diffusion pipeline with consistent error handling.
        
        Args:
            pipe_class: The pipeline class to instantiate (e.g., StableDiffusionImg2ImgPipeline)
            model_path: Path to model (fine-tuned directory or Hugging Face model ID)
            task_name: Name of the task for logging (e.g., "Denoising")
            fine_tuned_path: Optional Path object to check if model is fine-tuned
        
        Returns:
            Loaded pipeline instance
        """
        model_type = "fine-tuned" if fine_tuned_path and fine_tuned_path.exists() else "pre-trained"
        try:
            if self.device == "cuda":
                pipe = pipe_class.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    use_safetensors=False,
                )
                # Explicitly move all components to GPU (no CPU offloading)
                pipe = pipe.to("cuda")
                pipe.unet = pipe.unet.to("cuda")
                pipe.vae = pipe.vae.to("cuda")
                pipe.text_encoder = pipe.text_encoder.to("cuda")
                logger.info(f"{task_name} model ready ({model_type}, GPU)")
                
                # Verify components are on GPU
                if torch.cuda.is_available():
                    unet_device = next(pipe.unet.parameters()).device
                    vae_device = next(pipe.vae.parameters()).device
                    text_encoder_device = next(pipe.text_encoder.parameters()).device
                    logger.info(f"  UNet on: {unet_device}, VAE on: {vae_device}, Text encoder on: {text_encoder_device}")
                    
                    # Warn if any component is not on GPU
                    if unet_device.type != "cuda" or vae_device.type != "cuda" or text_encoder_device.type != "cuda":
                        logger.warning(f"  WARNING: Some components are not on GPU! This may cause slow inference.")
            else:
                pipe = pipe_class.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=False,
                )
                pipe = pipe.to("cpu")
                logger.info(f"{task_name} model ready ({model_type}, CPU)")
            return pipe
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU out of memory for {task_name}, retrying on CPU...")
                pipe = pipe_class.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=False,
                )
                pipe = pipe.to("cpu")
                logger.info(f"{task_name} model ready ({model_type}, using CPU)")
                return pipe
            raise
    
    def load_denoise_model(self):
        """Load denoising model - Fine-tuned or pre-trained Stable Diffusion from Hugging Face."""
        if "denoise" in self.models:
            return
        
        logger.info("Loading denoising model...")
        cfg = self.config["denoise"]
        backend = cfg.get("default_backend", "auto")
        
        # Try diffusion-based denoising (Stable Diffusion)
        if backend in ("auto", "diffusion"):
            try:
                fine_tuned_path = Path(cfg["fine_tuned_dir"])
                model_path = str(fine_tuned_path) if fine_tuned_path.exists() else cfg["pretrained_id"]
                
                if fine_tuned_path.exists():
                    logger.info("Found fine-tuned model, loading...")
                else:
                    logger.info("Using pre-trained model from Hugging Face (fine-tuned not found)")
                    logger.info("To use fine-tuned model, train with: python3 scripts/train_denoising.py")
                
                self.models["denoise"] = self._load_sd_pipeline(
                    StableDiffusionImg2ImgPipeline,
                    model_path,
                    task_name="Denoising",
                    fine_tuned_path=fine_tuned_path,
                )
                if backend == "diffusion":
                    return
            except Exception as e:
                if backend == "diffusion":
                    raise RuntimeError(f"Diffusion-based denoising failed: {e}")
                logger.warning(f"Could not load diffusion-based denoising model: {e}")
                if backend == "auto":
                    logger.info("Denoising will use OpenCV fallback")
        
        # Fallback to OpenCV
        if backend in ("auto", "opencv"):
            self.models["denoise"] = None
            logger.info("Denoising model ready (OpenCV fallback)")
    
    def load_sr_model(self):
        """Load super-resolution model - Fine-tuned or pre-trained Stable Diffusion, with fallbacks."""
        if "sr" in self.models:
            return
        
        logger.info("Loading super-resolution model...")
        cfg = self.config["sr"]
        backend = cfg.get("default_backend", "auto")
        fine_tuned_path = Path(cfg["fine_tuned_dir"])
        
        # Try Stable Diffusion Upscaler
        if backend in ("auto", "sd_upscaler"):
            try:
                if fine_tuned_path.exists():
                    logger.info("Found fine-tuned model, loading...")
                    model_path = str(fine_tuned_path)
                else:
                    logger.info("Using pre-trained model (fine-tuned not found)")
                    logger.info("To use fine-tuned model, train with: python3 scripts/train_super_resolution.py")
                    model_path = cfg["pretrained_id"]
                
                self.models["sr"] = self._load_sd_pipeline(
                    StableDiffusionUpscalePipeline,
                    model_path,
                    task_name="Super-resolution",
                    fine_tuned_path=fine_tuned_path,
                )
                if backend == "sd_upscaler":
                    return
            except Exception as e:
                if backend == "sd_upscaler":
                    raise RuntimeError(f"Stable Diffusion Upscaler failed: {e}")
                logger.warning(f"Stable Diffusion Upscaler failed: {e}")
                if backend == "auto":
                    logger.info("Trying Real-ESRGAN as fallback...")
            
            # Try Real-ESRGAN
            if backend in ("auto", "realesrgan"):
                try:
                    from realesrgan import RealESRGANer
                    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                    
                    model_name = 'RealESRGAN_x4plus'
                    model_path = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
                    
                    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                          num_conv=32, upscale=4, act_type='prelu')
                    # Real-ESRGAN automatically uses GPU if available and half=True
                    # Explicitly move model to GPU if CUDA is available
                    if self.device == "cuda" and torch.cuda.is_available():
                        model = model.to("cuda")
                    realesrgan = RealESRGANer(
                        scale=4,
                        model_path=model_path,
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=self.device == "cuda"
                    )
                    self.models["sr"] = realesrgan
                    device_info = "GPU" if self.device == "cuda" else "CPU"
                    logger.info(f"Super-resolution model ready (Real-ESRGAN, {device_info})")
                    if backend == "realesrgan":
                        return
                except ImportError:
                    if backend == "realesrgan":
                        raise ImportError("Real-ESRGAN library not installed. Install with: pip install realesrgan")
                    logger.warning("Real-ESRGAN library not installed")
                    logger.info("Install with: pip install realesrgan")
                    if backend == "auto":
                        logger.info("Using LANCZOS upscaling")
                except Exception as e2:
                    if backend == "realesrgan":
                        raise RuntimeError(f"Real-ESRGAN loading failed: {e2}")
                    logger.warning(f"Real-ESRGAN loading failed: {e2}")
                    if backend == "auto":
                        logger.info("Using LANCZOS upscaling")
            
            # Fallback to LANCZOS
            if backend in ("auto", "lanczos"):
                self.models["sr"] = "lanczos"
                logger.info("Super-resolution model ready (LANCZOS fallback)")
    
    def load_colorize_model(self):
        """Load colorization model - using fine-tuned or pre-trained Stable Diffusion."""
        if "colorize" in self.models:
            return
        
        logger.info("Loading colorization model...")
        cfg = self.config["colorize"]
        fine_tuned_path = Path(cfg["fine_tuned_dir"])
        model_path = str(fine_tuned_path) if fine_tuned_path.exists() else cfg["pretrained_id"]
        
        try:
            if fine_tuned_path.exists():
                logger.info("Found fine-tuned model, loading...")
            else:
                logger.info("Using pre-trained model (fine-tuned not found)")
                logger.info("To use fine-tuned model, train with: python3 scripts/train_colorization.py")
            
            self.models["colorize"] = self._load_sd_pipeline(
                StableDiffusionImg2ImgPipeline,
                model_path,
                task_name="Colorization",
                fine_tuned_path=fine_tuned_path,
            )
        except Exception as e:
            logger.warning(f"Could not load Stable Diffusion: {e}")
            logger.info("Using improved colorization fallback")
            self.models["colorize"] = "improved"
    
    def load_inpaint_model(self):
        """Load inpainting model - fine-tuned or pre-trained Stable Diffusion."""
        if "inpaint" in self.models:
            return
        
        logger.info("Loading inpainting model...")
        cfg = self.config["inpaint"]
        fine_tuned_path = Path(cfg["fine_tuned_dir"])
        model_path = str(fine_tuned_path) if fine_tuned_path.exists() else cfg["pretrained_id"]
        
        try:
            if fine_tuned_path.exists():
                logger.info("Found fine-tuned model, loading...")
            else:
                logger.info("Using pre-trained model (fine-tuned not found)")
                logger.info("To use fine-tuned model, train with: python3 scripts/train_inpainting.py")
            
            self.models["inpaint"] = self._load_sd_pipeline(
                StableDiffusionInpaintPipeline,
                model_path,
                task_name="Inpainting",
                fine_tuned_path=fine_tuned_path,
            )
        except Exception as e:
            logger.error("Could not load inpainting model", exc_info=True)
            logger.warning("Inpainting will be disabled")
            self.models["inpaint"] = None
    
    def denoise(self, image: Image.Image, strength: float = 0.5, **kwargs) -> Image.Image:
        """
        Denoise an image using fine-tuned or pre-trained Stable Diffusion from Hugging Face.
        
        Args:
            image: Input image to denoise
            strength: Denoising strength (0.0-1.0)
            **kwargs: Additional parameters including 'prompt' for custom prompt
        """
        if "denoise" not in self.models:
            self.load_denoise_model()
        
        model = self.models.get("denoise")
        
        # Use Stable Diffusion if available
        if isinstance(model, StableDiffusionImg2ImgPipeline):
            return self._denoise_sd(image, model, strength=strength, **kwargs)
        
        # Classical fallback
        return self._denoise_opencv(image, strength=strength)
    
    def _denoise_sd(self, image: Image.Image, model, strength: float, **kwargs) -> Image.Image:
        """Denoise using Stable Diffusion."""
        try:
            prompt = kwargs.get("prompt", self.prompts["denoise"])
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            result = model(
                prompt=prompt,
                image=image.convert("RGB"),
                strength=strength,
                num_inference_steps=20,
                guidance_scale=5.0,
                generator=generator
            )
            return result.images[0]
        except Exception as e:
            logger.warning(f"Stable Diffusion denoising failed: {e}, using OpenCV fallback")
            return self._denoise_opencv(image, strength=strength)
    
    def _denoise_opencv(self, image: Image.Image, strength: float) -> Image.Image:
        """Denoise using OpenCV (classical computer vision)."""
        img_np = np.array(image.convert("RGB"))
        
        # Adjust denoising strength based on input
        h = float(np.clip(strength, 0.1, 1.0))
        h_value = h * 10 if h < 0.6 else 20
        h_color = h * 10 if h < 0.6 else 20
        
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np, None,
            h=h_value,
            hColor=h_color,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        if strength > 0.6:
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        if strength > 0.8:
            denoised = cv2.medianBlur(denoised, 5)
        
        return Image.fromarray(denoised)
    
    def super_resolve(self, image: Image.Image, scale: int = 4, **kwargs) -> Image.Image:
        """
        Super-resolve an image using Real-ESRGAN or alternatives.
        
        Args:
            image: Input image to super-resolve
            scale: Upscaling factor
            **kwargs: Additional parameters including 'prompt' for custom prompt
        """
        if "sr" not in self.models:
            self.load_sr_model()
        
        model = self.models["sr"]
        
        # Use Stable Diffusion Upscaler if available
        if isinstance(model, StableDiffusionUpscalePipeline):
            return self._sr_sd(image, model, scale=scale, **kwargs)
        
        # Use Real-ESRGAN if available
        if hasattr(model, 'enhance'):
            return self._sr_realesrgan(image, model, scale=scale)
        
        # Classical fallback
        return self._sr_lanczos(image, scale=scale)
    
    def _sr_sd(self, image: Image.Image, model, scale: int, **kwargs) -> Image.Image:
        """Super-resolve using Stable Diffusion Upscaler."""
        try:
            w, h = image.size
            if w * h > 1024 * 1024:
                max_dim = 1024
                if w > h:
                    new_w, new_h = max_dim, int(h * max_dim / w)
                else:
                    new_w, new_h = int(w * max_dim / h), max_dim
                image = image.resize((new_w, new_h), Image.LANCZOS)
            
            prompt = kwargs.get("prompt", self.prompts["sr"])
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            result = model(
                prompt=prompt,
                image=image,
                num_inference_steps=20,
                guidance_scale=0,
                generator=generator
            )
            return result.images[0]
        except Exception as e:
            logger.warning(f"Stable Diffusion upscaling failed: {e}, falling back to LANCZOS")
            return self._sr_lanczos(image, scale=scale)
    
    def _sr_realesrgan(self, image: Image.Image, model, scale: int) -> Image.Image:
        """Super-resolve using Real-ESRGAN."""
        try:
            # Real-ESRGAN expects BGR format (OpenCV convention)
            img_np = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            output, _ = model.enhance(img_bgr, outscale=scale)
            # Convert back to RGB for PIL
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output_rgb)
        except Exception as e:
            logger.warning(f"Real-ESRGAN upscaling failed: {e}, falling back to LANCZOS")
            return self._sr_lanczos(image, scale=scale)
    
    def _sr_lanczos(self, image: Image.Image, scale: int) -> Image.Image:
        """Super-resolve using LANCZOS interpolation (classical fallback)."""
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.LANCZOS)
    
    def colorize(self, image: Image.Image, **kwargs) -> Image.Image:
        """
        Colorize a grayscale image using Stable Diffusion or improved fallback.
        
        Args:
            image: Input grayscale image to colorize
            **kwargs: Additional parameters including 'prompt' for custom prompt
        """
        if "colorize" not in self.models:
            self.load_colorize_model()
        
        model = self.models["colorize"]
        
        # Check if already color
        img_np = np.array(image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Check if it's actually grayscale (all channels same)
            if not (np.allclose(img_np[:,:,0], img_np[:,:,1]) and np.allclose(img_np[:,:,1], img_np[:,:,2])):
                # Already has color
                return image
        
        # Convert to RGB if grayscale
        if len(img_np.shape) == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_rgb)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Grayscale in RGB format - use first channel
            img_rgb = cv2.cvtColor(img_np[:,:,0], cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_rgb)
        
        # Use Stable Diffusion if available
        if isinstance(model, StableDiffusionImg2ImgPipeline):
            return self._colorize_sd(image, model, **kwargs)
        
        # Classical fallback
        return self._colorize_lab(image)
    
    def _colorize_sd(self, image: Image.Image, model, **kwargs) -> Image.Image:
        """Colorize using Stable Diffusion."""
        try:
            prompt = kwargs.get("prompt", self.prompts["colorize"])
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            result = model(
                prompt=prompt,
                image=image,
                strength=0.4,  # Reduced from 0.7 to preserve structure better
                num_inference_steps=20,
                guidance_scale=5.0,  # Reduced from 7.5 to be less aggressive
                generator=generator
            )
            return result.images[0]
        except Exception as e:
            logger.warning(f"Stable Diffusion colorization failed: {e}, using fallback", exc_info=True)
            return self._colorize_lab(image)
    
    def _colorize_lab(self, image: Image.Image) -> Image.Image:
        """Colorize using LAB color space (classical fallback)."""
        try:
            img_np = np.array(image.convert("RGB"))
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel = lab[:,:,0]
            
            # Simple colorization: add slight color variation based on luminance
            # This is a placeholder - real colorization needs learned color priors
            a_channel = np.clip(l_channel * 0.1 - 10, -127, 127).astype(np.int8)
            b_channel = np.clip(l_channel * 0.1 - 5, -127, 127).astype(np.int8)
            
            lab_colored = np.stack([l_channel, a_channel, b_channel], axis=2)
            rgb_colored = cv2.cvtColor(lab_colored.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(rgb_colored)
        except Exception as e:
            logger.warning(f"LAB colorization failed: {e}, returning grayscale as RGB")
            return image
    
    def inpaint(self, image: Image.Image, mask: Image.Image = None, prompt: str = None, **kwargs) -> Image.Image:
        """
        Inpaint missing/damaged parts using Stable Diffusion.
        
        Args:
            image: Input image to inpaint
            mask: Mask image (white areas will be inpainted)
            prompt: Text prompt for inpainting (defaults to self.prompts["inpaint"])
            **kwargs: Additional parameters
        """
        if "inpaint" not in self.models:
            self.load_inpaint_model()
        
        model = self.models.get("inpaint")
        if model is None:
            logger.warning("Inpainting model not available, returning original")
            return image
        
        # Use provided prompt or default from config
        if prompt is None:
            prompt = kwargs.get("prompt", self.prompts["inpaint"])
        
        # Auto-detect mask if not provided
        if mask is None:
            mask = self._auto_mask_from_image(image)
            if mask is None:
                return image
        
        # Normalize mask (resize and ensure correct polarity)
        mask = self._normalize_mask(mask, image.size)
        
        # Use Stable Diffusion if available
        if isinstance(model, StableDiffusionInpaintPipeline):
            return self._inpaint_sd(image, model, mask, prompt=prompt)
        
        # No classical fallback for inpainting - return original
        return image
    
    def _inpaint_sd(self, image: Image.Image, model, mask: Image.Image, prompt: str) -> Image.Image:
        """Inpaint using Stable Diffusion."""
        try:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            result = model(
                prompt=prompt,
                image=image.convert("RGB"),
                mask_image=mask,
                num_inference_steps=30,
                guidance_scale=5.0,
                strength=0.6,
                generator=generator
            )
            return result.images[0]
        except Exception as e:
            logger.error("Error in inpainting", exc_info=True)
            return image
    
    def _normalize_mask(self, mask: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """
        Normalize mask: resize to target size and ensure correct polarity (white = inpaint).
        
        Args:
            mask: Input mask image
            target_size: Target (width, height) tuple
        
        Returns:
            Normalized mask with correct size and polarity
        """
        # Resize mask to match target size
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.LANCZOS)
        
        # Ensure mask is correct format (white = area to inpaint)
        mask_np = np.array(mask.convert("L"))
        
        # Invert mask if needed (some masks use black for inpaint area)
        # Check if mask is mostly black (inverted) or mostly white
        white_ratio = np.sum(mask_np > 128) / mask_np.size
        if white_ratio < 0.1:  # If less than 10% white, probably inverted
            mask_np = 255 - mask_np
            mask = Image.fromarray(mask_np).convert("L")
        
        return mask
    
    def _auto_mask_from_image(self, image: Image.Image) -> Image.Image | None:
        """
        Auto-detect damaged areas in image to create mask (classical CV).
        
        Detects very dark regions (scratches/damage) and very bright regions (tears/holes).
        
        Args:
            image: Input image to analyze
        
        Returns:
            Mask image if significant damage detected, None otherwise
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect very dark regions (potential damage/scratches)
        _, mask_dark = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Detect very bright regions (potential damage/tears)
        _, mask_bright = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        mask_combined = cv2.bitwise_or(mask_dark, mask_bright)
        
        # Clean up mask (remove small noise)
        kernel = np.ones((5,5), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
        
        # Only use mask if significant area is detected
        mask_ratio = np.sum(mask_combined > 0) / (mask_combined.shape[0] * mask_combined.shape[1])
        if mask_ratio < 0.01:  # Less than 1% of image
            logger.info("No significant damage detected, skipping inpainting")
            return None
        
        return Image.fromarray(mask_combined).convert("L")
    
    def process(self, image: Image.Image, tasks: list[Task], **kwargs: Any) -> dict[str, Image.Image]:
        """
        Process image through multiple tasks in sequence.
        
        Args:
            image: Input PIL Image
            tasks: List of tasks to apply, e.g., ["denoise", "sr", "colorize", "inpaint"]
            **kwargs: Additional parameters (mask for inpainting, etc.)
        
        Returns:
            Dictionary with intermediate and final results (all values are Image.Image)
        """
        results = {"original": image, "final": image}
        current = image
        
        for task in tasks:
            try:
                if task == "denoise":
                    denoise_prompt = kwargs.get("denoise_prompt", None)
                    current = self.denoise(
                        current, 
                        strength=kwargs.get("denoise_strength", 0.5),
                        prompt=denoise_prompt
                    )
                    results["denoised"] = current
                elif task == "sr" or task == "super_resolution":
                    scale = kwargs.get("sr_scale", 4)
                    sr_prompt = kwargs.get("sr_prompt", None)
                    current = self.super_resolve(current, scale=scale, prompt=sr_prompt)
                    results["super_resolved"] = current
                elif task == "colorize":
                    colorize_prompt = kwargs.get("colorize_prompt", None)
                    current = self.colorize(current, prompt=colorize_prompt)
                    results["colorized"] = current
                elif task == "inpaint":
                    mask = kwargs.get("mask", None)
                    inpaint_prompt = kwargs.get("inpaint_prompt", None)
                    current = self.inpaint(current, mask=mask, prompt=inpaint_prompt)
                    results["inpainted"] = current
            except Exception as e:
                logger.error(f"Error processing task {task}", exc_info=True)
                continue
        
        results["final"] = current
        return results
