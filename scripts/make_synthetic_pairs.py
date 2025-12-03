#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import random
import sys

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Please run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(root: Path):
    """List all image files recursively from root directory."""
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(img: np.ndarray, sigma_range=(3, 15)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_jpeg_compression(img: np.ndarray, quality_range=(30, 90)) -> np.ndarray:
    """Add JPEG compression artifacts to simulate real-world compression."""
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec


def add_motion_blur(img: np.ndarray, kernel_size_range=(5, 15), angle_range=(0, 360)) -> np.ndarray:
    """Add motion blur to simulate camera shake."""
    # Random kernel size (length of motion)
    kernel_size = random.randint(*kernel_size_range)
    # Random angle (direction of motion)
    angle = random.uniform(*angle_range)
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    # Calculate center
    center = kernel_size // 2
    # Calculate direction vector
    angle_rad = math.radians(angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    # Draw line in kernel
    for i in range(kernel_size):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply motion blur
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def degrade_sr(img: np.ndarray, scale=4) -> np.ndarray:
    """Degrade image for super-resolution: blur (Gaussian or motion) -> downsample -> JPEG compression."""
    # Sometimes use motion blur instead of Gaussian blur to simulate camera shake (30% chance)
    if random.random() < 0.3:
        blur = add_motion_blur(img, kernel_size_range=(5, 12))
    else:
        k = random.choice([3, 5, 7])
        blur = cv2.GaussianBlur(img, (k, k), sigmaX=0)
    
    h, w = blur.shape[:2]
    lr = cv2.resize(blur, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    # Add JPEG compression artifacts to simulate real-world low-res images
    lr = add_jpeg_compression(lr)
    return lr


def to_grayscale(img: np.ndarray, mode: str = "simple") -> np.ndarray:
    """Convert image to grayscale.
    
    Args:
        img: BGR image
        mode: "simple" (BGR2GRAY) or "lab" (LAB L channel, preserves luminance better)
    """
    if mode == "lab":
        # Convert to LAB and use L channel (luminance)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]  # L channel
    else:  # simple
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_max_size(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image to max_size if larger, maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def random_free_form_mask(h: int, w: int, num_strokes=(5, 15)) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(random.randint(*num_strokes)):
        pts = []
        num_pts = random.randint(4, 8)
        for _ in range(num_pts):
            pts.append((random.randint(0, w - 1), random.randint(0, h - 1)))
        thickness = random.randint(10, 40)
        for i in range(len(pts) - 1):
            cv2.line(mask, pts[i], pts[i + 1], color=255, thickness=thickness)
    return mask


def process_split(clean_dir: Path, out_root: Path, sr_scale: int, max_size: int = 1024, 
                  tasks: list[str] = None, grayscale_mode: str = "simple"):
    """Process a split (train/val/test) and generate synthetic pairs.
    
    Args:
        clean_dir: Directory containing clean images
        out_root: Output root directory
        sr_scale: Super-resolution scale factor
        max_size: Maximum image dimension
        tasks: List of tasks to generate (denoise, sr, colorize, inpaint). If None, all tasks.
        grayscale_mode: "simple" or "lab" for colorization grayscale conversion
    """
    if tasks is None:
        tasks = ["denoise", "sr", "colorize", "inpaint"]
    
    imgs = list_images(clean_dir)
    if not imgs:
        print(f"No images found in {clean_dir}")
        return

    # Define output paths for all tasks (only create dirs for selected tasks)
    denoise_in = out_root / "denoise" / clean_dir.name / "input"
    denoise_gt = out_root / "denoise" / clean_dir.name / "gt"
    sr_in = out_root / "sr_x{}".format(sr_scale) / clean_dir.name / "input"
    sr_gt = out_root / "sr_x{}".format(sr_scale) / clean_dir.name / "gt"
    color_in = out_root / "colorize" / clean_dir.name / "input"
    color_gt = out_root / "colorize" / clean_dir.name / "gt"
    inpaint_in = out_root / "inpaint" / clean_dir.name / "input"
    inpaint_mask = out_root / "inpaint" / clean_dir.name / "mask"
    inpaint_gt = out_root / "inpaint" / clean_dir.name / "gt"

    # Create directories only for selected tasks
    dirs_to_create = []
    if "denoise" in tasks:
        dirs_to_create.extend([denoise_in, denoise_gt])
    if "sr" in tasks:
        dirs_to_create.extend([sr_in, sr_gt])
    if "colorize" in tasks:
        dirs_to_create.extend([color_in, color_gt])
    if "inpaint" in tasks:
        dirs_to_create.extend([inpaint_in, inpaint_mask, inpaint_gt])

    for d in dirs_to_create:
        ensure_dir(d)

    for p in tqdm(imgs, desc=f"{clean_dir.name}"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Resize to max_size if image is too large (before all degradations)
        img = resize_to_max_size(img, max_size=max_size)

        # Denoising pair: Gaussian noise + sometimes JPEG compression + sometimes motion blur
        if "denoise" in tasks:
            # Simulates sensor noise + compression artifacts + camera shake
            noisy = add_gaussian_noise(img)
            # Sometimes add JPEG compression in addition to Gaussian noise (30% chance)
            if random.random() < 0.3:
                noisy = add_jpeg_compression(noisy, quality_range=(40, 85))
            # Sometimes add motion blur to simulate camera shake (20% chance)
            if random.random() < 0.2:
                noisy = add_motion_blur(noisy, kernel_size_range=(3, 8))
            cv2.imwrite(str(denoise_in / p.name), noisy)
            cv2.imwrite(str(denoise_gt / p.name), img)

        # SR pair
        if "sr" in tasks:
            lr = degrade_sr(img, scale=sr_scale)
            cv2.imwrite(str(sr_in / p.name), lr)
            cv2.imwrite(str(sr_gt / p.name), img)

        # Colorization pair (input: 1-channel gray saved as png; gt: color)
        if "colorize" in tasks:
            # Note: Models expect 3-channel RGB, so training/inference code should expand
            # grayscale to RGB (e.g., PIL's .convert("RGB") or np.stack([gray]*3, axis=-1))
            gray = to_grayscale(img, mode=grayscale_mode)
            gray_name = Path(p.stem + ".png")  # ensure lossless for gray
            cv2.imwrite(str(color_in / gray_name), gray)
            cv2.imwrite(str(color_gt / p.name), img)

        # Inpainting pair: masked image + mask + gt
        if "inpaint" in tasks:
            # Mask convention: white (255) = region to inpaint (matches Stable Diffusion Inpaint)
            # Masked image: original image with inpaint regions set to black (0)
            h, w = img.shape[:2]
            mask = random_free_form_mask(h, w)
            masked = img.copy()
            masked[mask == 255] = 0
            cv2.imwrite(str(inpaint_in / p.name), masked)
            cv2.imwrite(str(inpaint_mask / p.name), mask)
            cv2.imwrite(str(inpaint_gt / p.name), img)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pairs for denoise, SR, colorize, and inpaint from clean images")
    parser.add_argument("--clean_root", type=str, default="data/clean", help="Directory containing clean/train|val|test")
    parser.add_argument("--out_root", type=str, default="data/pairs", help="Output root for task pairs")
    parser.add_argument("--sr_scale", type=int, default=4, choices=[2, 3, 4], help="Super-resolution downscale factor")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum dimension for images (resize if larger, maintains aspect ratio)")
    parser.add_argument("--tasks", type=str, default="denoise,sr,colorize,inpaint", 
                       help="Comma-separated list of tasks to generate: denoise, sr, colorize, inpaint")
    parser.add_argument("--grayscale_mode", type=str, default="simple", choices=["simple", "lab"],
                       help="Grayscale conversion mode: 'simple' (BGR2GRAY) or 'lab' (LAB L channel)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse tasks
    tasks = [t.strip().lower() for t in args.tasks.split(",")]
    valid_tasks = {"denoise", "sr", "colorize", "inpaint"}
    invalid_tasks = [t for t in tasks if t not in valid_tasks]
    if invalid_tasks:
        parser.error(f"Invalid tasks: {invalid_tasks}. Valid tasks are: {', '.join(valid_tasks)}")
    
    if not tasks:
        parser.error("At least one task must be specified")

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_root = Path(args.clean_root).resolve()
    out_root = Path(args.out_root).resolve()
    for split in ["train", "val", "test"]:
        process_split(clean_root / split, out_root, args.sr_scale, max_size=args.max_size, 
                     tasks=tasks, grayscale_mode=args.grayscale_mode)

    print(f"Done. Generated pairs for tasks: {', '.join(tasks)} under {out_root}")


if __name__ == "__main__":
    main()


