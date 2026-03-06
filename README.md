# Prompted Segmentation for Drywall QA

Text-prompted binary segmentation of drywall defects. Given an image and a natural language prompt, the model produces a binary mask identifying the defect region.

**Two prompts supported:**
- `"segment crack"` — localises surface cracks
- `"segment taping area"` — localises drywall joints and seams

---

## Models

| Model | Backbone | Test IoU | Inference |
|-------|----------|----------|-----------|
| CLIPSeg | CLIP ViT-B/16 | 0.4395 | ~48ms |
| CrackCLIP | CLIP ViT-B/16 + Adapters | 0.4727 | 47.8ms |
| CrackSAM (**best**) | SAM2 Hiera-B+ + LoRA | 0.6168 | 330ms |

---

## Dataset

Two datasets from Roboflow combined:
- [Drywall-Join-Detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) — 246 images, YOLO bbox annotations
- [Cracks](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) — 1,112 images, COCO polygon annotations

After deduplication: **1,358 unique images** split 70/15/15 (train/val/test), seed=42.

---

## Project Structure

```
├── dataset/
│   └── processed/          # standardised images + binary masks
│       ├── drywall/
│       ├── cracks/
│       ├── splits.json
│       └── prompts.csv
├── src/
│   ├── dataset.py
│   ├── augmentations.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── 01_clipseg.ipynb
│   ├── 04_crackclip_v3.ipynb
│   └── 03_sam2.ipynb
└── outputs/
    ├── checkpoints/
    ├── predictions/
    └── visualisations/
```

---

## Reproducing Results

All notebooks were run on Kaggle with a T4 GPU. To reproduce:

1. Download both datasets from Roboflow and place them under `dataset/`
2. Run `data_preparation.ipynb` to generate `dataset/processed/`
3. Run the model notebooks in order — each notebook is self-contained

All experiments use `seed=42`.

---

## Output Format

Prediction masks are saved as single-channel PNG files at the original source image resolution with pixel values in `{0, 255}`.

Filename format: `{image_id}__{prompt}.png`
Example: `cracks_00042__segment_crack.png`

---

## Requirements

```
torch
transformers
albumentations
opencv-python
pandas
sam2 (from facebookresearch/sam2)
```