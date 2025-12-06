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


def add_gaussian_noise(img: np.ndarray, sigma_range=(5, 8)) -> np.ndarray:
    """Add Gaussian noise for denoising task."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_jpeg_compression(img: np.ndarray, quality_range=(30, 90)) -> np.ndarray:
    """Add JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def add_motion_blur(img: np.ndarray, kernel_size_range=(5, 15), angle_range=(0, 360)) -> np.ndarray:
    """Add motion blur to simulate camera shake."""
    kernel_size = random.randint(*kernel_size_range)
    angle = random.uniform(*angle_range)
    
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    angle_rad = math.radians(angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    for i in range(kernel_size):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)


def degrade_sr(img: np.ndarray, scale=4, use_jpeg: bool = False, use_motion_blur: bool = False) -> np.ndarray:
    """Degrade image for super-resolution."""
    if use_motion_blur and random.random() < 0.3:
        blur = add_motion_blur(img, kernel_size_range=(5, 12))
    else:
        k = random.choice([3, 5, 7])
        blur = cv2.GaussianBlur(img, (k, k), sigmaX=0)
    
    h, w = blur.shape[:2]
    lr = cv2.resize(blur, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    
    if use_jpeg:
        lr = add_jpeg_compression(lr, quality_range=(40, 85))
    
    return lr


def to_grayscale(img: np.ndarray, mode: str = "lab") -> np.ndarray:
    """Convert image to grayscale. LAB mode preserves luminance better."""
    if mode == "lab":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]
    else:
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


def random_free_form_mask(h: int, w: int, num_strokes=(5, 15), thickness_range=(10, 40)) -> np.ndarray:
    """Generate random free-form mask composed of line strokes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(random.randint(*num_strokes)):
        pts = []
        for _ in range(random.randint(4, 8)):
            pts.append((random.randint(0, w - 1), random.randint(0, h - 1)))
        thickness = random.randint(*thickness_range)
        for i in range(len(pts) - 1):
            cv2.line(mask, pts[i], pts[i + 1], color=255, thickness=thickness)
    return mask

def process_split(clean_dir: Path, out_root: Path, sr_scale: int, max_size: int = 1024, 
                  tasks: list[str] = None, grayscale_mode: str = "lab",
                  denoise_with_artifacts: bool = False, sr_with_jpeg: bool = False,
                  sr_with_motion_blur: bool = False, inpaint_easy_ratio: float = 0.7):
    """Process a split (train/val/test) and generate synthetic pairs."""
    if tasks is None:
        tasks = ["denoise", "sr", "colorize", "inpaint"]
    
    imgs = list_images(clean_dir)
    if not imgs:
        print(f"WARNING: No images found in {clean_dir}")
        print(f"  Looking for files with extensions: {IMG_EXTS}")
        return
    
    print(f"  Found {len(imgs)} images in {clean_dir.name}")

    denoise_in = out_root / "denoise" / clean_dir.name / "input"
    denoise_gt = out_root / "denoise" / clean_dir.name / "gt"
    sr_in = out_root / f"sr_x{sr_scale}" / clean_dir.name / "input"
    sr_gt = out_root / f"sr_x{sr_scale}" / clean_dir.name / "gt"
    color_in = out_root / "colorize" / clean_dir.name / "input"
    color_gt = out_root / "colorize" / clean_dir.name / "gt"
    inpaint_in = out_root / "inpaint" / clean_dir.name / "input"
    inpaint_mask = out_root / "inpaint" / clean_dir.name / "mask"
    inpaint_gt = out_root / "inpaint" / clean_dir.name / "gt"

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

        img = resize_to_max_size(img, max_size=max_size)

        if "denoise" in tasks:
            if denoise_with_artifacts:
                noisy = add_gaussian_noise(img, sigma_range=(3, 15))
                if random.random() < 0.3:
                    noisy = add_jpeg_compression(noisy, quality_range=(40, 85))
                if random.random() < 0.2:
                    noisy = add_motion_blur(noisy, kernel_size_range=(3, 8))
            else:
                noisy = add_gaussian_noise(img, sigma_range=(5, 8))
            cv2.imwrite(str(denoise_in / p.name), noisy)
            cv2.imwrite(str(denoise_gt / p.name), img)

        if "sr" in tasks:
            lr = degrade_sr(img, scale=sr_scale, use_jpeg=sr_with_jpeg, use_motion_blur=sr_with_motion_blur)
            cv2.imwrite(str(sr_in / p.name), lr)
            cv2.imwrite(str(sr_gt / p.name), img)

        if "colorize" in tasks:
            gray = to_grayscale(img, mode=grayscale_mode)
            gray_name = Path(p.stem + ".png")
            cv2.imwrite(str(color_in / gray_name), gray)
            cv2.imwrite(str(color_gt / p.name), img)

        if "inpaint" in tasks:
            h, w = img.shape[:2]
            if random.random() < inpaint_easy_ratio:
                mask = random_free_form_mask(h, w, num_strokes=(3, 7), thickness_range=(5, 20))
            else:
                mask = random_free_form_mask(h, w, num_strokes=(8, 15), thickness_range=(20, 40))
            masked = img.copy()
            masked[mask == 255] = 0
            cv2.imwrite(str(inpaint_in / p.name), masked)
            cv2.imwrite(str(inpaint_mask / p.name), mask)
            cv2.imwrite(str(inpaint_gt / p.name), img)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pairs for denoise, SR, colorize, and inpaint")
    parser.add_argument("--clean_root", type=str, default="data/clean", help="Directory containing clean/train|val|test")
    parser.add_argument("--out_root", type=str, default="data/pairs", help="Output root for task pairs")
    parser.add_argument("--sr_scale", type=int, default=4, choices=[2, 3, 4], help="Super-resolution downscale factor")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum image dimension")
    parser.add_argument("--tasks", type=str, default="denoise,sr,colorize,inpaint", 
                       help="Comma-separated list of tasks: denoise, sr, colorize, inpaint")
    parser.add_argument("--grayscale_mode", type=str, default="lab", choices=["simple", "lab"],
                       help="Grayscale mode: 'lab' (default) or 'simple'")
    parser.add_argument("--denoise_with_artifacts", action="store_true",
                       help="Add JPEG compression and motion blur to denoise (default: clean Gaussian only)")
    parser.add_argument("--sr_with_jpeg", action="store_true",
                       help="Add JPEG compression to SR (default: clean SR only)")
    parser.add_argument("--sr_with_motion_blur", action="store_true",
                       help="Use motion blur for SR (default: Gaussian only)")
    parser.add_argument("--inpaint_easy_ratio", type=float, default=0.7,
                       help="Ratio of easy vs hard masks (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = [t.strip().lower() for t in args.tasks.split(",")]
    valid_tasks = {"denoise", "sr", "colorize", "inpaint"}
    invalid_tasks = [t for t in tasks if t not in valid_tasks]
    if invalid_tasks:
        parser.error(f"Invalid tasks: {invalid_tasks}. Valid: {', '.join(valid_tasks)}")
    
    if not tasks:
        parser.error("At least one task must be specified")

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_root = Path(args.clean_root).resolve()
    out_root = Path(args.out_root).resolve()
    
    print(f"Clean root: {clean_root}")
    print(f"Output root: {out_root}")
    print(f"Tasks: {', '.join(tasks)}")
    
    if not clean_root.exists():
        print(f"ERROR: Clean root directory does not exist: {clean_root}")
        return
    
    splits_found = []
    for split in ["train", "val", "test"]:
        split_dir = clean_root / split
        if split_dir.exists():
            splits_found.append(split)
            print(f"Processing {split} split: {split_dir}")
            process_split(split_dir, out_root, args.sr_scale, max_size=args.max_size, 
                         tasks=tasks, grayscale_mode=args.grayscale_mode,
                         denoise_with_artifacts=args.denoise_with_artifacts,
                         sr_with_jpeg=args.sr_with_jpeg,
                         sr_with_motion_blur=args.sr_with_motion_blur,
                         inpaint_easy_ratio=args.inpaint_easy_ratio)
        else:
            print(f"WARNING: Split directory not found: {split_dir} (skipping)")
    
    if not splits_found:
        print(f"ERROR: No valid split directories found in {clean_root}")
        print(f"Expected structure: {clean_root}/train, {clean_root}/val, {clean_root}/test")
        return
    
    print(f"Done. Generated pairs for tasks: {', '.join(tasks)} under {out_root}")
    print(f"Processed splits: {', '.join(splits_found)}")


if __name__ == "__main__":
    main()


