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

# Standardized model directory mapping for fine-tuned models
# All training scripts save the best model (based on validation PSNR) to:
#   outputs/models/{task}/best/
# This directory contains the full pipeline saved via pipeline.save_pretrained():
#   - unet/ (fine-tuned UNet weights)
#   - vae/ (VAE weights)
#   - text_encoder/ (text encoder weights)
#   - tokenizer/ (tokenizer config)
#   - scheduler/ (scheduler config)
TASK_MODEL_DIRS = {
    "denoise": "outputs/models/denoising/best",
    "sr": "outputs/models/super_resolution/best",
    "colorize": "outputs/models/colorization/best",
    "inpaint": "outputs/models/inpainting/best",
}

try:
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionImg2ImgPipeline,
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
                "fine_tuned_dir": TASK_MODEL_DIRS["denoise"],
                "pretrained_id": "sd-legacy/stable-diffusion-v1-5",
                "default_backend": "auto",  # "auto" | "diffusion" | "opencv"
            },
            "sr": {
                "fine_tuned_dir": TASK_MODEL_DIRS["sr"],
                "pretrained_id": "sd-legacy/stable-diffusion-v1-5",
                "default_backend": "auto",  # "auto" | "sd_img2img" | "realesrgan" | "lanczos"
            },
            "colorize": {
                "fine_tuned_dir": TASK_MODEL_DIRS["colorize"],
                "pretrained_id": "sd-legacy/stable-diffusion-v1-5",
            },
            "inpaint": {
                "fine_tuned_dir": TASK_MODEL_DIRS["inpaint"],
                "pretrained_id": "runwayml/stable-diffusion-inpainting",  # Note: Still using runwayml for inpainting (no direct stabilityai replacement)
            },
        }
        
        self.config = default_config if config is None else {**default_config, **config}
        
        # Default prompts for each task
        self.prompts = {
            "denoise": "clean high quality photo, no noise, sharp details",
            "sr": "high quality, detailed, sharp",
            "colorize": "vibrant realistic natural colors, colorful, high quality photo, detailed, full color, rich colors",
            "inpaint": "high quality detailed photo",
        }
    
    def _find_latest_checkpoint(self, output_dir: Path) -> Path | None:
        """
        Find the latest checkpoint directory that contains UNet weights.
        
        Args:
            output_dir: Base output directory (e.g., outputs/models/denoising)
        
        Returns:
            Path to latest checkpoint with UNet weights, or None if not found
        """
        if not output_dir.exists():
            return None
        
        # Look for checkpoint directories (checkpoint-epoch-* or checkpoint-*)
        checkpoints = []
        for checkpoint_dir in output_dir.glob("checkpoint-*"):
            unet_dir = checkpoint_dir / "unet"
            if unet_dir.exists():
                # Check if UNet weights exist
                unet_weights = list(unet_dir.glob("*.safetensors")) + list(unet_dir.glob("*.bin"))
                if unet_weights:
                    checkpoints.append(checkpoint_dir)
        
        if not checkpoints:
            return None
        
        # Sort by directory name (checkpoint-epoch-N or checkpoint-N)
        # Extract number for sorting
        def get_checkpoint_num(path: Path) -> int:
            name = path.name
            if "epoch" in name:
                # checkpoint-epoch-N
                try:
                    return int(name.split("-")[-1])
                except:
                    return 0
            else:
                # checkpoint-N
                try:
                    return int(name.split("-")[-1])
                except:
                    return 0
        
        checkpoints.sort(key=get_checkpoint_num, reverse=True)
        return checkpoints[0]
    
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
            # Load pipeline - let diffusers handle dtype automatically
            try:
                pipe = pipe_class.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                )
            except (TypeError, OSError, EnvironmentError):
                # Fallback if torch_dtype not supported
                pipe = pipe_class.from_pretrained(
                    model_path,
                    use_safetensors=True,
                )
            
            # Move to appropriate device
            if self.device == "cuda":
                pipe = pipe.to("cuda")
                # Set components to eval mode for inference
                pipe.unet.eval()
                pipe.vae.eval()
                pipe.text_encoder.eval()
                logger.info(f"{task_name} model ready ({model_type}, GPU)")
            else:
                pipe = pipe.to("cpu")
                logger.info(f"{task_name} model ready ({model_type}, CPU)")
            
            return pipe
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU out of memory for {task_name}, retrying on CPU...")
                pipe = pipe_class.from_pretrained(
                    model_path,
                    use_safetensors=True,
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
                
                # Check if we're in fine-tuned mode (not using "nonexistent" as a fallback indicator)
                is_pretrained_mode = cfg["fine_tuned_dir"] == "nonexistent"
                
                if fine_tuned_path.exists():
                    logger.info("Found fine-tuned model, loading...")
                    try:
                        self.models["denoise"] = self._load_sd_pipeline(
                            StableDiffusionImg2ImgPipeline,
                            str(fine_tuned_path),
                            task_name="Denoising",
                            fine_tuned_path=fine_tuned_path,
                        )
                        if backend == "diffusion":
                            return
                    except (OSError, EnvironmentError) as e:
                        # Fine-tuned model directory exists but is incomplete
                        error_msg = f"Fine-tuned denoising model directory exists but is incomplete: {e}"
                        logger.error(error_msg)
                        if is_pretrained_mode:
                            # In pretrained mode, fall back to pretrained
                            logger.info(f"Falling back to pretrained model: {cfg['pretrained_id']}")
                            model_path = cfg["pretrained_id"]
                            self.models["denoise"] = self._load_sd_pipeline(
                                StableDiffusionImg2ImgPipeline,
                                model_path,
                                task_name="Denoising",
                                fine_tuned_path=None,
                            )
                            if backend == "diffusion":
                                return
                        else:
                            # In fine-tuned mode, raise error
                            raise FileNotFoundError(
                                f"Fine-tuned denoising model not found or incomplete at {fine_tuned_path}. "
                                f"Please train the model first with: python3 scripts/train_denoising.py"
                            )
                else:
                    # Fine-tuned model doesn't exist
                    if is_pretrained_mode:
                        # In pretrained mode, use pretrained model
                        logger.info("Using pre-trained model from Hugging Face")
                        model_path = cfg["pretrained_id"]
                        self.models["denoise"] = self._load_sd_pipeline(
                            StableDiffusionImg2ImgPipeline,
                            model_path,
                            task_name="Denoising",
                            fine_tuned_path=None,
                        )
                        if backend == "diffusion":
                            return
                    else:
                        # In fine-tuned mode, raise error
                        raise FileNotFoundError(
                            f"Fine-tuned denoising model not found at {fine_tuned_path}. "
                            f"Please train the model first with: python3 scripts/train_denoising.py"
                        )
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
        
        # Try Stable Diffusion Img2Img (aligned with training script)
        if backend in ("auto", "sd_img2img"):
            try:
                is_pretrained_mode = cfg["fine_tuned_dir"] == "nonexistent"
                
                if fine_tuned_path.exists():
                    logger.info("Found fine-tuned model, loading...")
                    model_path = str(fine_tuned_path)
                else:
                    if is_pretrained_mode:
                        # In pretrained mode, use pretrained model
                        logger.info("Using pre-trained model from Hugging Face")
                        model_path = cfg["pretrained_id"]
                    else:
                        # In fine-tuned mode, raise error
                        raise FileNotFoundError(
                            f"Fine-tuned super-resolution model not found at {fine_tuned_path}. "
                            f"Please train the model first with: python3 scripts/train_super_resolution.py"
                        )
                
                self.models["sr"] = self._load_sd_pipeline(
                    StableDiffusionImg2ImgPipeline,
                    model_path,
                    task_name="Super-resolution",
                    fine_tuned_path=fine_tuned_path if fine_tuned_path.exists() else None,
                )
                if backend == "sd_img2img":
                    return
            except Exception as e:
                if backend == "sd_img2img":
                    raise RuntimeError(f"Stable Diffusion Img2Img failed: {e}")
                logger.warning(f"Stable Diffusion Img2Img failed: {e}")
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
        is_pretrained_mode = cfg["fine_tuned_dir"] == "nonexistent"
        
        try:
            if fine_tuned_path.exists():
                logger.info("Found fine-tuned model, loading...")
                model_path = str(fine_tuned_path)
            else:
                if is_pretrained_mode:
                    # In pretrained mode, use pretrained model
                    logger.info("Using pre-trained model from Hugging Face")
                    model_path = cfg["pretrained_id"]
                else:
                    # In fine-tuned mode, raise error
                    raise FileNotFoundError(
                        f"Fine-tuned colorization model not found at {fine_tuned_path}. "
                        f"Please train the model first with: python3 scripts/train_colorization.py"
                    )
            
            self.models["colorize"] = self._load_sd_pipeline(
                StableDiffusionImg2ImgPipeline,
                model_path,
                task_name="Colorization",
                fine_tuned_path=fine_tuned_path if fine_tuned_path.exists() else None,
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
        is_pretrained_mode = cfg["fine_tuned_dir"] == "nonexistent"
        
        try:
            if fine_tuned_path.exists():
                logger.info("Found fine-tuned model, loading...")
                model_path = str(fine_tuned_path)
            else:
                if is_pretrained_mode:
                    # In pretrained mode, use pretrained model
                    logger.info("Using pre-trained model from Hugging Face")
                    model_path = cfg["pretrained_id"]
                else:
                    # In fine-tuned mode, raise error
                    raise FileNotFoundError(
                        f"Fine-tuned inpainting model not found at {fine_tuned_path}. "
                        f"Please train the model first with: python3 scripts/train_inpainting.py"
                    )
            
            self.models["inpaint"] = self._load_sd_pipeline(
                StableDiffusionInpaintPipeline,
                model_path,
                task_name="Inpainting",
                fine_tuned_path=fine_tuned_path if fine_tuned_path.exists() else None,
            )
            
            # Disable safety checker for inpainting (it blocks legitimate restoration images)
            if hasattr(self.models["inpaint"], 'safety_checker'):
                self.models["inpaint"].safety_checker = None
            if hasattr(self.models["inpaint"], 'feature_extractor'):
                self.models["inpaint"].feature_extractor = None
            if hasattr(self.models["inpaint"], 'requires_safety_checker'):
                self.models["inpaint"].requires_safety_checker = False
            logger.info("Safety checker disabled for inpainting")
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
            model_device = next(model.unet.parameters()).device
            generator = torch.Generator(device=model_device).manual_seed(self.seed)
            
            with torch.no_grad():
                result = model(
                    prompt=prompt,
                    image=image.convert("RGB"),
                    strength=strength,
                    num_inference_steps=20,
                    guidance_scale=5.0,
                    generator=generator,
                    output_type="pil"
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
        
        # Use Stable Diffusion Img2Img if available (aligned with training)
        if isinstance(model, StableDiffusionImg2ImgPipeline):
            return self._sr_sd(image, model, scale=scale, **kwargs)
        
        # Use Real-ESRGAN if available
        if hasattr(model, 'enhance'):
            return self._sr_realesrgan(image, model, scale=scale)
        
        # Classical fallback
        return self._sr_lanczos(image, scale=scale)
    
    def _sr_sd(self, image: Image.Image, model, scale: int, **kwargs) -> Image.Image:
        """Super-resolve using Stable Diffusion Img2Img (aligned with training)."""
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
            model_device = next(model.unet.parameters()).device
            generator = torch.Generator(device=model_device).manual_seed(self.seed)
            
            with torch.no_grad():
                result = model(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=20,
                    guidance_scale=0,
                    generator=generator,
                    output_type="pil"
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
        
        # Check if already color - be more lenient for grayscale detection
        img_np = np.array(image)
        
        # Convert to grayscale for comparison
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Calculate per-pixel differences between channels
            diff_rg = np.abs(img_np[:,:,0].astype(np.float32) - img_np[:,:,1].astype(np.float32))
            diff_gb = np.abs(img_np[:,:,1].astype(np.float32) - img_np[:,:,2].astype(np.float32))
            diff_rb = np.abs(img_np[:,:,0].astype(np.float32) - img_np[:,:,2].astype(np.float32))
            
            # Calculate mean difference across all pixels
            mean_diff = (np.mean(diff_rg) + np.mean(diff_gb) + np.mean(diff_rb)) / 3.0
            
            # If mean difference is > 10, image likely has color
            # (allowing for JPEG compression artifacts)
            if mean_diff > 10.0:
                logger.info(f"Image already has color (mean channel diff: {mean_diff:.2f}), skipping colorization")
                return image
            else:
                logger.info(f"Detected grayscale image (mean channel diff: {mean_diff:.2f}), proceeding with colorization")
        
        # Convert to RGB if grayscale
        if len(img_np.shape) == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_rgb)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Grayscale in RGB format - use first channel
            img_rgb = cv2.cvtColor(img_np[:,:,0], cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_rgb)
        
        logger.info(f"Colorizing grayscale image, model type: {type(model)}")
        
        # Use Stable Diffusion if available
        if isinstance(model, StableDiffusionImg2ImgPipeline):
            result = self._colorize_sd(image, model, **kwargs)
            logger.info("Colorization completed using Stable Diffusion")
            return result
        
        # Classical fallback
        logger.info("Using LAB colorization fallback")
        return self._colorize_lab(image)
    
    def _colorize_sd(self, image: Image.Image, model, **kwargs) -> Image.Image:
        """Colorize using Stable Diffusion."""
        try:
            prompt = kwargs.get("prompt") or self.prompts.get("colorize", "vibrant realistic natural colors, colorful, high quality photo, detailed, full color, rich colors")
            if not prompt:
                prompt = "vibrant realistic natural colors, colorful, high quality photo, detailed, full color, rich colors"
            
            model_device = next(model.unet.parameters()).device
            generator = torch.Generator(device=model_device).manual_seed(self.seed)
            
            with torch.no_grad():
                result = model(
                    prompt=prompt,
                    image=image,
                    strength=0.75,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                    output_type="pil"
                )
            
            if hasattr(result, 'images') and result.images:
                return result.images[0]
            
            logger.warning("Colorization failed to produce valid output, returning original")
            return image
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
            # Ensure safety checker is disabled before inference
            if hasattr(model, 'safety_checker'):
                model.safety_checker = None
            if hasattr(model, 'feature_extractor'):
                model.feature_extractor = None
            if hasattr(model, 'requires_safety_checker'):
                model.requires_safety_checker = False
            
            model_device = next(model.unet.parameters()).device
            generator = torch.Generator(device=model_device).manual_seed(self.seed)
            
            with torch.no_grad():
                result = model(
                    prompt=prompt,
                    image=image.convert("RGB"),
                    mask_image=mask,
                    num_inference_steps=30,
                    guidance_scale=5.0,
                    strength=0.6,
                    generator=generator,
                    output_type="pil"
                )
            
            if hasattr(result, 'images') and result.images:
                return result.images[0]
            
            logger.warning("Inpainting failed to produce valid output, returning original")
            return image
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
                    # Only pass prompt if it's explicitly provided and not None
                    colorize_prompt = kwargs.get("colorize_prompt")
                    if colorize_prompt:
                        current = self.colorize(current, prompt=colorize_prompt)
                    else:
                        current = self.colorize(current)
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
