#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Please run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    # COCO zips contain a directory named the same as split
    return out_dir


def sample_and_copy(images_dir: Path, output_root: Path, num_images: int, seed: int = 42, split_name: str = "train") -> None:
    rng = random.Random(seed)
    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not all_images:
        raise RuntimeError(f"No images found in {images_dir}")

    if num_images > 0 and num_images < len(all_images):
        selected = rng.sample(all_images, num_images)
    else:
        selected = all_images

    # Create simple train/val/test split from the sampled set: 85/10/5
    rng.shuffle(selected)
    n = len(selected)
    n_train = int(0.85 * n)
    n_val = int(0.10 * n)
    splits = {
        "train": selected[:n_train],
        "val": selected[n_train:n_train + n_val],
        "test": selected[n_train + n_val:],
    }

    for sname, files in splits.items():
        out_dir = output_root / "clean" / sname
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = out_dir / src.name
            shutil.copy2(src, dst)

    print(f"Saved: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} under {output_root/'clean'}")


def main():
    parser = argparse.ArgumentParser(description="Download a COCO 2017 split and sample images into clean/train|val|test")
    parser.add_argument("--split", choices=["val2017", "train2017"], default="val2017", help="COCO split to download")
    parser.add_argument("--num_images", type=int, default=1500, help="Number of images to sample (0 = use all in split)")
    parser.add_argument("--out_dir", type=str, default="data", help="Root output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    url = COCO_URLS[args.split]
    zip_name = Path(url).name
    zip_path = out_root / zip_name

    if not zip_path.exists():
        print(f"Downloading {args.split} from {url} ...")
        download_file(url, zip_path)
    else:
        print(f"Found existing archive: {zip_path}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        print("Extracting archive ...")
        extract_zip(zip_path, td_path)
        images_dir = td_path / args.split.replace("2017", "2017")  # keep name e.g., val2017
        if not images_dir.exists():
            # COCO zip extracts to a folder named exactly split
            images_dir = td_path / args.split
        if not images_dir.exists():
            # Fallback: search for directory with many jpgs
            candidates = [p for p in td_path.iterdir() if p.is_dir()]
            if candidates:
                images_dir = candidates[0]
        if not images_dir.exists():
            raise RuntimeError("Could not locate extracted images directory")

        print(f"Sampling and copying images from {images_dir} ...")
        sample_and_copy(images_dir, out_root, args.num_images, seed=args.seed)

    print("Done.")


if __name__ == "__main__":
    main()


