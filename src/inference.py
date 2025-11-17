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
warnings.filterwarnings("ignore")

try:
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionImg2ImgPipeline,
        DDPMPipeline,
        StableDiffusionUpscalePipeline
    )
    from transformers import pipeline as hf_pipeline
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Missing dependencies. Install: pip install diffusers transformers huggingface_hub", file=sys.stderr)
    sys.exit(1)


class RestorationPipeline:
    """Unified pipeline for image restoration tasks using Hugging Face models."""
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.models = {}
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Using device: {self.device} ({self.dtype})")
    
    def load_denoise_model(self):
        """Load denoising model - Hybrid: OpenCV for light noise, SwinIR/DDPM for heavy noise."""
        if "denoise" not in self.models:
            print("Loading denoising models (hybrid approach)...")
            
            # Try SwinIR for denoising (more reliable than DDPM)
            try:
                print("  Loading SwinIR model for heavy noise...")
                from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution
                
                processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
                model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
                model = model.to(self.device)
                self.models["denoise_swinir"] = {"processor": processor, "model": model}
                print("  SwinIR model ready")
            except Exception as e:
                print(f"  WARNING: Could not load SwinIR: {e}")
                # Try DDPM as alternative (but it may not exist)
                try:
                    print("  Trying DDPM model...")
                    self.models["denoise_ddpm"] = DDPMPipeline.from_pretrained(
                        "google/ddpm-celebahq-256",  # Alternative DDPM model
                        dtype=self.dtype
                    ).to(self.device)
                    print("  DDPM model ready")
                except Exception as e2:
                    print(f"  WARNING: Could not load DDPM either: {e2}")
                    print("  Will use OpenCV only for all noise levels")
                    self.models["denoise_swinir"] = None
                    self.models["denoise_ddpm"] = None
            
            # OpenCV is always available (no model loading needed)
            self.models["denoise"] = "hybrid"
            print("Denoising ready (hybrid: OpenCV for light noise, SwinIR/DDPM for heavy noise)")
    
    def load_sr_model(self):
        """Load super-resolution model - Real-ESRGAN or alternatives."""
        if "sr" not in self.models:
            print("Loading super-resolution model...")
            
            # Try Real-ESRGAN library first (if available)
            try:
                from realesrgan import RealESRGANer
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                
                # Real-ESRGAN model paths
                model_name = 'RealESRGAN_x4plus'
                model_path = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
                
                # Create Real-ESRGAN upsampler
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                      num_conv=32, upscale=4, act_type='prelu')
                self.models["sr"] = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    model=model,
                    tile=0,  # Tile size, 0 for no tiling
                    tile_pad=10,
                    pre_pad=0,
                    half=self.device == "cuda"  # Use half precision on GPU
                )
                print("Super-resolution model ready (Real-ESRGAN)")
            except ImportError:
                print("  WARNING: Real-ESRGAN library not installed")
                print("  Install with: pip install realesrgan")
                print("  Trying alternative: Stable Diffusion Upscaler...")
                try:
                    if self.device == "cuda":
                        try:
                            self.models["sr"] = StableDiffusionUpscalePipeline.from_pretrained(
                                "stabilityai/stable-diffusion-x4-upscaler",
                                dtype=self.dtype,
                                use_safetensors=False
                            )
                            self.models["sr"].enable_model_cpu_offload()
                            print(" Using Stable Diffusion Upscaler (CPU offloading)")
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"  WARNING: GPU out of memory, using CPU...")
                                self.models["sr"] = StableDiffusionUpscalePipeline.from_pretrained(
                                    "stabilityai/stable-diffusion-x4-upscaler",
                                    dtype=torch.float32,
                                    use_safetensors=False
                                )
                                self.models["sr"] = self.models["sr"].to("cpu")
                                print("Using Stable Diffusion Upscaler (CPU)")
                            else:
                                raise
                    else:
                        self.models["sr"] = StableDiffusionUpscalePipeline.from_pretrained(
                            "stabilityai/stable-diffusion-x4-upscaler",
                            dtype=torch.float32,
                            use_safetensors=False
                        )
                        print("Using Stable Diffusion Upscaler (CPU)")
                except Exception as e2:
                    print(f"  WARNING: Stable Diffusion Upscaler failed: {e2}")
                    print("  Using LANCZOS upscaling")
                    self.models["sr"] = "lanczos"
            except Exception as e:
                print(f"  WARNING: Real-ESRGAN loading failed: {e}")
                print("  Trying alternative: Stable Diffusion Upscaler...")
                try:
                    if self.device == "cuda":
                        try:
                            self.models["sr"] = StableDiffusionUpscalePipeline.from_pretrained(
                                "stabilityai/stable-diffusion-x4-upscaler",
                                dtype=self.dtype,
                                use_safetensors=False
                            )
                            self.models["sr"].enable_model_cpu_offload()
                            print(" Using Stable Diffusion Upscaler (CPU offloading)")
                        except RuntimeError as e_mem:
                            if "out of memory" in str(e_mem).lower():
                                print(f"  WARNING: GPU out of memory, using CPU...")
                                self.models["sr"] = StableDiffusionUpscalePipeline.from_pretrained(
                                    "stabilityai/stable-diffusion-x4-upscaler",
                                    dtype=torch.float32,
                                    use_safetensors=False
                                )
                                self.models["sr"] = self.models["sr"].to("cpu")
                                print("Using Stable Diffusion Upscaler (CPU)")
                            else:
                                raise
                    else:
                        self.models["sr"] = StableDiffusionUpscalePipeline.from_pretrained(
                            "stabilityai/stable-diffusion-x4-upscaler",
                            dtype=torch.float32,
                            use_safetensors=False
                        )
                        print("Using Stable Diffusion Upscaler (CPU)")
                except Exception as e2:
                    print(f"  WARNING: Fallback also failed: {e2}")
                    print("  Using LANCZOS upscaling")
                    self.models["sr"] = "lanczos"
    
    def load_colorize_model(self):
        """Load colorization model - using fine-tuned or pre-trained Stable Diffusion."""
        if "colorize" not in self.models:
            print("Loading colorization model...")
            
            # Try using Stable Diffusion for colorization (text-to-image with grayscale conditioning)
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                
                # Check if fine-tuned model exists
                fine_tuned_path = Path("outputs/models/colorization/final")
                model_path = str(fine_tuned_path) if fine_tuned_path.exists() else "runwayml/stable-diffusion-v1-5"
                
                if fine_tuned_path.exists():
                    print("  Found fine-tuned model, loading...")
                else:
                    print("  Using pre-trained model (fine-tuned not found)")
                    print("  To use fine-tuned model, train with: python3 scripts/train_colorization.py")
                
                if self.device == "cuda":
                    try:
                        self.models["colorize"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                            model_path,
                            dtype=self.dtype,
                            use_safetensors=False
                        )
                        # Enable CPU offloading to save GPU memory
                        self.models["colorize"].enable_model_cpu_offload()
                        model_type = "fine-tuned" if fine_tuned_path.exists() else "pre-trained"
                        print(f"Colorization model ready ({model_type}, CPU offloading enabled)")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            print(f"  WARNING: GPU out of memory, trying CPU...")
                            self.models["colorize"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                                model_path,
                                dtype=torch.float32,
                                use_safetensors=False
                            )
                            self.models["colorize"] = self.models["colorize"].to("cpu")
                            model_type = "fine-tuned" if fine_tuned_path.exists() else "pre-trained"
                            print(f"Colorization model ready ({model_type}, using CPU)")
                        else:
                            raise
                else:
                    self.models["colorize"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model_path,
                        dtype=torch.float32,
                        use_safetensors=False
                    )
                    model_type = "fine-tuned" if fine_tuned_path.exists() else "pre-trained"
                    print(f"Colorization model ready ({model_type}, CPU)")
            except Exception as e:
                print(f"  WARNING: Could not load Stable Diffusion: {e}")
                print("  Using improved colorization fallback")
                # Use improved colorization (better than simple placeholder)
                self.models["colorize"] = "improved"
    
    def load_inpaint_model(self):
        """Load inpainting model - fine-tuned or pre-trained Stable Diffusion."""
        if "inpaint" not in self.models:
            print("Loading inpainting model...")
            try:
                from diffusers import StableDiffusionInpaintPipeline
                
                # Check if fine-tuned model exists
                fine_tuned_path = Path("outputs/models/inpainting/final")
                model_path = str(fine_tuned_path) if fine_tuned_path.exists() else "runwayml/stable-diffusion-inpainting"
                
                if fine_tuned_path.exists():
                    print("  Found fine-tuned model, loading...")
                else:
                    print("  Using pre-trained model (fine-tuned not found)")
                    print("  To use fine-tuned model, train with: python3 scripts/train_inpainting.py")
                
                # Try loading with CPU offloading to save GPU memory
                if self.device == "cuda":
                    try:
                        self.models["inpaint"] = StableDiffusionInpaintPipeline.from_pretrained(
                            model_path,
                            dtype=self.dtype,
                            use_safetensors=False  # Allow pickle fallback
                        )
                        # Enable CPU offloading to save GPU memory
                        self.models["inpaint"].enable_model_cpu_offload()
                        model_type = "fine-tuned" if fine_tuned_path.exists() else "pre-trained"
                        print(f"Inpainting model ready ({model_type}, CPU offloading enabled)")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            print(f"WARNING: GPU out of memory, trying CPU...")
                            # Fallback to CPU
                            self.models["inpaint"] = StableDiffusionInpaintPipeline.from_pretrained(
                                model_path,
                                dtype=torch.float32,
                                use_safetensors=False
                            )
                            self.models["inpaint"] = self.models["inpaint"].to("cpu")
                            model_type = "fine-tuned" if fine_tuned_path.exists() else "pre-trained"
                            print(f"Inpainting model ready ({model_type}, using CPU)")
                        else:
                            raise
                else:
                    # CPU mode
                    self.models["inpaint"] = StableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        dtype=torch.float32,
                        use_safetensors=False
                    )
                    model_type = "fine-tuned" if fine_tuned_path.exists() else "pre-trained"
                    print(f"Inpainting model ready ({model_type}, CPU)")
            except Exception as e:
                print(f"WARNING: Could not load inpainting model: {e}")
                print("  Inpainting will be disabled")
                self.models["inpaint"] = None
    
    def denoise(self, image: Image.Image, strength: float = 0.5) -> Image.Image:
        """
        Denoise an image using hybrid approach:
        - OpenCV for light noise (strength < 0.6)
        - SwinIR/DDPM for heavy noise (strength >= 0.6)
        """
        if "denoise" not in self.models:
            self.load_denoise_model()
        
        # Hybrid approach: use OpenCV for light noise, advanced models for heavy noise
        if strength < 0.6:
            # Light noise: Use OpenCV (fast and effective)
            img_np = np.array(image.convert("RGB"))
            
            # Non-local means denoising (better than bilateral for light noise)
            denoised = cv2.fastNlMeansDenoisingColored(
                img_np, None, 
                h=10 * strength,  # Filter strength
                hColor=10 * strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            return Image.fromarray(denoised)
        else:
            # Heavy noise: Use SwinIR or DDPM (slower but more powerful)
            swinir_model = self.models.get("denoise_swinir")
            ddpm_model = self.models.get("denoise_ddpm")
            
            if swinir_model is None and ddpm_model is None:
                # Fallback to enhanced OpenCV for heavy noise
                print("WARNING: Advanced models not available, using enhanced OpenCV for heavy noise")
                img_np = np.array(image.convert("RGB"))
                
                # Multi-stage denoising for heavy noise
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_np, None, h=20, hColor=20, templateWindowSize=7, searchWindowSize=21
                )
                denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
                if strength > 0.8:
                    denoised = cv2.medianBlur(denoised, 5)
                return Image.fromarray(denoised)
            
            # Try SwinIR first (more reliable)
            if swinir_model is not None:
                try:
                    processor = swinir_model["processor"]
                    model = swinir_model["model"]
                    
                    # Process image
                    inputs = processor(image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get output image
                    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1)
                    output = (output.numpy() * 255).astype(np.uint8)
                    output = np.transpose(output, (1, 2, 0))
                    
                    return Image.fromarray(output)
                except Exception as e:
                    print(f"WARNING: SwinIR denoising failed: {e}, trying fallback...")
            
            # Try DDPM as fallback
            if ddpm_model is not None:
                try:
                    original_size = image.size
                    # Resize if needed (DDPM models have size constraints)
                    if image.size[0] > 256 or image.size[1] > 256:
                        max_dim = 256
                        w, h = image.size
                        if w > h:
                            new_w, new_h = max_dim, int(h * max_dim / w)
                        else:
                            new_w, new_h = int(w * max_dim / h), max_dim
                        image = image.resize((new_w, new_h), Image.LANCZOS)
                    
                    result = ddpm_model(
                        image,
                        num_inference_steps=50,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    ).images[0]
                    
                    if result.size != original_size:
                        result = result.resize(original_size, Image.LANCZOS)
                    
                    return result
                except Exception as e:
                    print(f"WARNING: DDPM denoising failed: {e}, falling back to OpenCV")
            
            # Final fallback to OpenCV
            img_np = np.array(image.convert("RGB"))
            denoised = cv2.fastNlMeansDenoisingColored(
                img_np, None, h=20, hColor=20, templateWindowSize=7, searchWindowSize=21
            )
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            if strength > 0.8:
                denoised = cv2.medianBlur(denoised, 5)
            return Image.fromarray(denoised)
    
    def super_resolve(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """Super-resolve an image using Real-ESRGAN or alternatives."""
        if "sr" not in self.models:
            self.load_sr_model()
        
        model = self.models["sr"]
        
        if isinstance(model, str) and model == "lanczos":
            # Fallback: High-quality LANCZOS upscaling
            w, h = image.size
            upscaled = image.resize((w * scale, h * scale), Image.LANCZOS)
            return upscaled
        
        # Use Real-ESRGAN, Stable Diffusion Upscaler, or other
        try:
            # Check if it's Real-ESRGANer (from realesrgan library)
            if hasattr(model, 'enhance'):
                # Real-ESRGAN library
                img_np = np.array(image.convert("RGB"))
                output, _ = model.enhance(img_np, outscale=scale)
                result = Image.fromarray(output)
                return result
            elif isinstance(model, StableDiffusionUpscalePipeline):
                # Stable Diffusion Upscaler (fallback)
                w, h = image.size
                if w * h > 1024 * 1024:
                    max_dim = 1024
                    if w > h:
                        new_w, new_h = max_dim, int(h * max_dim / w)
                    else:
                        new_w, new_h = int(w * max_dim / h), max_dim
                    image = image.resize((new_w, new_h), Image.LANCZOS)
                
                result = model(
                    prompt="high quality, detailed, sharp",
                    image=image,
                    num_inference_steps=20,
                    guidance_scale=0
                )
                return result.images[0]
            else:
                raise ValueError("Unknown SR model type")
        except Exception as e:
            print(f"WARNING: Error in super-resolution: {e}, falling back to LANCZOS")
            w, h = image.size
            return image.resize((w * scale, h * scale), Image.LANCZOS)
    
    def colorize(self, image: Image.Image) -> Image.Image:
        """Colorize a grayscale image using Stable Diffusion or improved fallback."""
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
        from diffusers import StableDiffusionImg2ImgPipeline
        if isinstance(model, StableDiffusionImg2ImgPipeline):
            try:
                # Use Stable Diffusion Img2Img for colorization
                # Lower strength to preserve structure and avoid over-generation
                prompt = "realistic natural colors, high quality photo, detailed"
                result = model(
                    prompt=prompt,
                    image=image,
                    strength=0.4,  # Reduced from 0.7 to preserve structure better
                    num_inference_steps=20,
                    guidance_scale=5.0  # Reduced from 7.5 to be less aggressive
                ).images[0]
                return result
            except Exception as e:
                print(f"WARNING: Stable Diffusion colorization failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
        
        # Improved fallback: Use LAB color space colorization
        if isinstance(model, str) and model == "improved":
            try:
                img_np = np.array(image.convert("RGB"))
                
                # Convert to LAB color space
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l_channel = lab[:,:,0]
                
                # Apply colorization using bilateral filter in LAB space
                # This creates a simple but effective colorization
                # In practice, you'd use a trained model, but this is a reasonable fallback
                
                # Create color hints (simple approach)
                # For better results, you could use a pre-trained colorization network
                # For now, we'll use a simple color transfer approach
                
                # Use the L channel and create a/b channels with slight variation
                # This is a placeholder - real colorization needs learned color priors
                a_channel = np.zeros_like(l_channel)
                b_channel = np.zeros_like(l_channel)
                
                # Simple colorization: add slight color variation based on luminance
                # This is very basic - a real model would learn color priors
                a_channel = np.clip(l_channel * 0.1 - 10, -127, 127).astype(np.int8)
                b_channel = np.clip(l_channel * 0.1 - 5, -127, 127).astype(np.int8)
                
                lab_colored = np.stack([l_channel, a_channel, b_channel], axis=2)
                rgb_colored = cv2.cvtColor(lab_colored.astype(np.uint8), cv2.COLOR_LAB2RGB)
                
                return Image.fromarray(rgb_colored)
            except Exception as e:
                print(f"WARNING: Improved colorization failed: {e}, returning grayscale as RGB")
                return image
        
        # Final fallback: return grayscale as RGB
        return image
    
    def inpaint(self, image: Image.Image, mask: Image.Image = None, prompt: str = "high quality detailed photo") -> Image.Image:
        """Inpaint missing/damaged parts using Stable Diffusion."""
        if "inpaint" not in self.models:
            self.load_inpaint_model()
        
        if self.models["inpaint"] is None:
            print("Inpainting model not available, returning original")
            return image
        
        # If no mask provided, try to detect damaged areas
        if mask is None:
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
                print("No significant damage detected, skipping inpainting")
                return image
            
            mask = Image.fromarray(mask_combined).convert("L")
        
        # Ensure mask is same size as image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        # Run inpainting
        try:
            # Ensure mask is correct format (white = area to inpaint)
            # Stable Diffusion expects white areas to be inpainted
            mask_np = np.array(mask.convert("L"))
            
            # Invert mask if needed (some masks use black for inpaint area)
            # Check if mask is mostly black (inverted) or mostly white
            white_ratio = np.sum(mask_np > 128) / mask_np.size
            if white_ratio < 0.1:  # If less than 10% white, probably inverted
                mask_np = 255 - mask_np
                mask = Image.fromarray(mask_np).convert("L")
            
            result = self.models["inpaint"](
                prompt=prompt,
                image=image.convert("RGB"),
                mask_image=mask,
                num_inference_steps=30,
                guidance_scale=5.0,  # Reduced from 7.5 to be less aggressive
                strength=0.6  # Reduced from 0.75 to preserve more original content
            )
            return result.images[0]
        except Exception as e:
            print(f"WARNING: Error in inpainting: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def process(self, image: Image.Image, tasks: list, **kwargs) -> dict:
        """
        Process image through multiple tasks in sequence.
        
        Args:
            image: Input PIL Image
            tasks: List of tasks to apply, e.g., ["denoise", "sr", "colorize", "inpaint"]
            **kwargs: Additional parameters (mask for inpainting, etc.)
        
        Returns:
            Dictionary with intermediate and final results
        """
        results = {"original": image, "final": image}
        current = image
        
        for task in tasks:
            try:
                if task == "denoise":
                    current = self.denoise(current, strength=kwargs.get("denoise_strength", 0.5))
                    results["denoised"] = current
                elif task == "sr" or task == "super_resolution":
                    scale = kwargs.get("sr_scale", 4)
                    current = self.super_resolve(current, scale=scale)
                    results["super_resolved"] = current
                elif task == "colorize":
                    current = self.colorize(current)
                    results["colorized"] = current
                elif task == "inpaint":
                    mask = kwargs.get("mask", None)
                    prompt = kwargs.get("inpaint_prompt", "high quality detailed photo")
                    current = self.inpaint(current, mask=mask, prompt=prompt)
                    results["inpainted"] = current
            except Exception as e:
                print(f"Error processing task {task}: {e}")
                continue
        
        results["final"] = current
        return results
