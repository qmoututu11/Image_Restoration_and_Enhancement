#!/usr/bin/env python3
import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_files(d: Path):
    if not d.exists():
        return []
    return [p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def check_disjoint(a, b):
    return set(a).isdisjoint(set(b))


def report_split(root: Path):
    train = list_files(root / "train")
    val = list_files(root / "val")
    test = list_files(root / "test")
    total = len(train) + len(val) + len(test)

    print(f"\nSplit at: {root}")
    print(f"  train: {len(train)} | val: {len(val)} | test: {len(test)} | total: {total}")

    ok_tv = check_disjoint(train, val)
    ok_tt = check_disjoint(train, test)
    ok_vt = check_disjoint(val, test)
    if ok_tv and ok_tt and ok_vt:
        print("  Disjoint check: OK (no filename overlaps)")
    else:
        print("  Disjoint check: FAIL")
        if not ok_tv:
            overlap = set(train).intersection(set(val))
            print(f"    Overlap train-val: {len(overlap)} (e.g., {list(overlap)[:5]})")
        if not ok_tt:
            overlap = set(train).intersection(set(test))
            print(f"    Overlap train-test: {len(overlap)} (e.g., {list(overlap)[:5]})")
        if not ok_vt:
            overlap = set(val).intersection(set(test))
            print(f"    Overlap val-test: {len(overlap)} (e.g., {list(overlap)[:5]})")


def main():
    parser = argparse.ArgumentParser(description="Verify train/val/test splits and disjointness")
    parser.add_argument("--clean_root", type=str, default="data/clean", help="Root of clean dataset with train/val/test")
    parser.add_argument("--pairs_root", type=str, default="data/pairs", help="Root of generated pairs")
    args = parser.parse_args()

    clean_root = Path(args.clean_root).resolve()
    pairs_root = Path(args.pairs_root).resolve()

    # Check clean split
    report_split(clean_root)

    # Check all task pairs that follow task/split/{input,gt} (and mask for inpaint)
    if pairs_root.exists():
        for task_dir in sorted([p for p in pairs_root.iterdir() if p.is_dir()]):
            print(f"\nTask: {task_dir.name}")
            for split in ["train", "val", "test"]:
                input_dir = task_dir / split / "input"
                gt_dir = task_dir / split / "gt"
                mask_dir = task_dir / split / "mask"

                input_files = list_files(input_dir)
                gt_files = list_files(gt_dir)
                mask_files = list_files(mask_dir) if mask_dir.exists() else []

                print(f"  {split}: input={len(input_files)} gt={len(gt_files)}" + (f" mask={len(mask_files)}" if mask_files else ""))

                # Basic consistency checks
                if len(input_files) != len(gt_files):
                    print(f"    WARNING: count mismatch input({len(input_files)}) vs gt({len(gt_files)})")

                # Disjointness check by filenames within the split
                if input_files and gt_files:
                    if not set(input_files) == set(gt_files) and not task_dir.name.startswith("colorize"):
                        # For colorize, gray inputs are .png while gt may be .jpg
                        print("    NOTE: Filenames differ (expected for colorize due to PNG inputs)")
            
    else:
        print(f"Pairs root not found: {pairs_root}")


if __name__ == "__main__":
    main()


