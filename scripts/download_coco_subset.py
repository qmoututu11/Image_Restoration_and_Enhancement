#!/usr/bin/env python3
import argparse
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


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)


def sample_and_copy(images_dir: Path, output_root: Path, num_images: int, seed: int = 42, target_split: str = None) -> None:
    """Sample images and copy to output directory."""
    rng = random.Random(seed)
    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not all_images:
        raise RuntimeError(f"No images found in {images_dir}")

    if num_images > 0 and num_images < len(all_images):
        selected = rng.sample(all_images, num_images)
    else:
        selected = all_images

    if target_split:
        out_dir = output_root / "clean" / target_split
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in selected:
            dst = out_dir / src.name
            shutil.copy2(src, dst)
        print(f"Saved {len(selected)} images to {output_root/'clean'/target_split}")
    else:
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
    parser = argparse.ArgumentParser(description="Download COCO 2017 splits: train2017 for training, val2017 for validation")
    parser.add_argument("--train_images", type=int, default=2000, help="Number of training images from train2017 (0 = use all)")
    parser.add_argument("--val_images", type=int, default=200, help="Number of validation images from val2017 (0 = use all)")
    parser.add_argument("--test_images", type=int, default=100, help="Number of test images (sampled from val2017)")
    parser.add_argument("--out_dir", type=str, default="data", help="Root output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--skip_train", action="store_true", help="Skip downloading train2017 (use existing)")
    parser.add_argument("--skip_val", action="store_true", help="Skip downloading val2017 (use existing)")
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Download train2017 for training
    if not args.skip_train:
        print("\n=== Downloading train2017 for training ===")
        train_url = COCO_URLS["train2017"]
        train_zip = out_root / "train2017.zip"
        
        if not train_zip.exists():
            print(f"Downloading train2017 from {train_url} ...")
            print(f"Warning: This is ~19GB. Make sure you have enough disk space!")
            download_file(train_url, train_zip)
        else:
            print(f"Found existing archive: {train_zip}")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            print("Extracting train2017 archive ...")
            extract_zip(train_zip, td_path)
            images_dir = td_path / "train2017"
            if not images_dir.exists():
                candidates = [p for p in td_path.iterdir() if p.is_dir()]
                if candidates:
                    images_dir = candidates[0]
            if not images_dir.exists():
                raise RuntimeError("Could not locate extracted train2017 images directory")

            print(f"Sampling {args.train_images} training images from train2017...")
            sample_and_copy(images_dir, out_root, args.train_images, seed=args.seed, target_split="train")

    # Download val2017 for validation and test
    if not args.skip_val:
        print("\n=== Downloading val2017 for validation ===")
        val_url = COCO_URLS["val2017"]
        val_zip = out_root / "val2017.zip"
        
        if not val_zip.exists():
            print(f"Downloading val2017 from {val_url} ...")
            download_file(val_url, val_zip)
        else:
            print(f"Found existing archive: {val_zip}")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            print("Extracting val2017 archive ...")
            extract_zip(val_zip, td_path)
            images_dir = td_path / "val2017"
            if not images_dir.exists():
                candidates = [p for p in td_path.iterdir() if p.is_dir()]
                if candidates:
                    images_dir = candidates[0]
            if not images_dir.exists():
                raise RuntimeError("Could not locate extracted val2017 images directory")

            all_val_images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            rng = random.Random(args.seed)
            rng.shuffle(all_val_images)
            
            total_needed = args.train_images + args.val_images + args.test_images
            if total_needed > len(all_val_images):
                total_needed = len(all_val_images)
            
            selected = all_val_images[:total_needed]
            
            train_selected = selected[:args.train_images] if args.train_images > 0 else []
            val_start = args.train_images
            val_end = val_start + args.val_images if args.val_images > 0 else val_start
            val_selected = selected[val_start:val_end] if args.val_images > 0 else []
            test_start = val_end
            test_end = test_start + args.test_images if args.test_images > 0 else test_start
            test_selected = selected[test_start:test_end] if args.test_images > 0 else []
            
            if train_selected:
                train_dir = out_root / "clean" / "train"
                train_dir.mkdir(parents=True, exist_ok=True)
                for src in train_selected:
                    shutil.copy2(src, train_dir / src.name)
                print(f"Saved {len(train_selected)} training images to {train_dir}")
            
            if val_selected:
                val_dir = out_root / "clean" / "val"
                val_dir.mkdir(parents=True, exist_ok=True)
                for src in val_selected:
                    shutil.copy2(src, val_dir / src.name)
                print(f"Saved {len(val_selected)} validation images to {val_dir}")
            
            if test_selected:
                test_dir = out_root / "clean" / "test"
                test_dir.mkdir(parents=True, exist_ok=True)
                for src in test_selected:
                    shutil.copy2(src, test_dir / src.name)
                print(f"Saved {len(test_selected)} test images to {test_dir}")

    print("\nDone!")
    print(f"\nFinal dataset structure:")
    for split in ["train", "val", "test"]:
        split_dir = out_root / "clean" / split
        if split_dir.exists():
            count = len(list(split_dir.glob("*.jpg")))
            print(f"  {split}: {count} images")


if __name__ == "__main__":
    main()


