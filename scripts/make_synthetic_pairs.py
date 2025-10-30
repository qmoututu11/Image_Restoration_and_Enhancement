#!/usr/bin/env python3
import argparse
import math
import os
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
    return [p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(img: np.ndarray, sigma_range=(5, 25)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def degrade_sr(img: np.ndarray, scale=4) -> np.ndarray:
    # blur -> downsample -> (optional) noise + jpeg like
    k = random.choice([3, 5, 7])
    blur = cv2.GaussianBlur(img, (k, k), sigmaX=0)
    h, w = blur.shape[:2]
    lr = cv2.resize(blur, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return lr


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def process_split(clean_dir: Path, out_root: Path, sr_scale: int):
    imgs = list_images(clean_dir)
    if not imgs:
        print(f"No images found in {clean_dir}")
        return

    # Prepare output dirs
    denoise_in = out_root / "denoise" / clean_dir.name / "input"
    denoise_gt = out_root / "denoise" / clean_dir.name / "gt"
    sr_in = out_root / "sr_x{}".format(sr_scale) / clean_dir.name / "input"
    sr_gt = out_root / "sr_x{}".format(sr_scale) / clean_dir.name / "gt"
    color_in = out_root / "colorize" / clean_dir.name / "input"
    color_gt = out_root / "colorize" / clean_dir.name / "gt"
    inpaint_in = out_root / "inpaint" / clean_dir.name / "input"
    inpaint_mask = out_root / "inpaint" / clean_dir.name / "mask"
    inpaint_gt = out_root / "inpaint" / clean_dir.name / "gt"

    for d in [denoise_in, denoise_gt, sr_in, sr_gt, color_in, color_gt, inpaint_in, inpaint_mask, inpaint_gt]:
        ensure_dir(d)

    for p in tqdm(imgs, desc=f"{clean_dir.name}"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Denoising pair
        noisy = add_gaussian_noise(img)
        cv2.imwrite(str(denoise_in / p.name), noisy)
        cv2.imwrite(str(denoise_gt / p.name), img)

        # SR pair
        lr = degrade_sr(img, scale=sr_scale)
        cv2.imwrite(str(sr_in / p.name), lr)
        cv2.imwrite(str(sr_gt / p.name), img)

        # Colorization pair (input: 1-channel gray saved as png; gt: color)
        gray = to_grayscale(img)
        gray_name = Path(p.stem + ".png")  # ensure lossless for gray
        cv2.imwrite(str(color_in / gray_name), gray)
        cv2.imwrite(str(color_gt / p.name), img)

        # Inpainting pair: masked image + mask + gt
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_root = Path(args.clean_root).resolve()
    out_root = Path(args.out_root).resolve()
    for split in ["train", "val", "test"]:
        process_split(clean_root / split, out_root, args.sr_scale)

    print(f"Done. Generated pairs under {out_root}")


if __name__ == "__main__":
    main()


