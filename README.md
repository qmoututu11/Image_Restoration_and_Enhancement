# Image_Restoration_and_Enhancement
RestoraGen: A Generative AI System for Image Restoration and Enhancement

## Quick start: build a clean image set and synthetic training pairs

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Download a COCO split and sample images (default: 1,500):

```
python3 scripts/download_coco_subset.py --split val2017 --num_images 1500 --out_dir data
```

This creates `data/clean/{train,val,test}` with sampled images.

3. Generate synthetic pairs for tasks (denoise, SR, colorize, inpaint):

```
python3 scripts/make_synthetic_pairs.py --clean_root data/clean --out_root data/pairs --sr_scale 4
```

Outputs live under:
- `data/pairs/denoise/{train,val,test}/{input,gt}`
- `data/pairs/sr_x4/{train,val,test}/{input,gt}`
- `data/pairs/colorize/{train,val,test}/{input,gt}` (inputs are grayscale PNG)
- `data/pairs/inpaint/{train,val,test}/{input,mask,gt}`

## Quick test: Stable Diffusion Inpainting (Hugging Face)

Install extra deps (already in requirements):

```
pip install -r requirements.txt
```

Run a one-sample inpainting test (uses data/pairs/inpaint/test):

```
python3 scripts/test_inference_sd_inpaint.py \
  --pairs_root data/pairs/inpaint \
  --output outputs/sd_inpaint_test.png
```

If you have a GPU, it will use it automatically; otherwise CPU works but is slower.

