import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SegDataset(Dataset):
    """
    Prompted segmentation dataset for drywall joint and crack detection.

    Expects the processed/ folder produced by data_preparation.ipynb:
        processed/
            drywall/
                images/   drywall_00001.jpg, ...
                masks/    drywall_00001.png, ...
            cracks/
                images/   cracks_00001.jpg, ...
                masks/    cracks_00001.png, ...

    Args:
        stems          : list of stems e.g. ["drywall_00001", "cracks_00042"]
        processed_root : path to the processed/ folder
        prompts_df     : DataFrame loaded from prompts.csv
        split          : "train" | "val" | "test"
        image_size     : (H, W) — (352, 352) for CLIPSeg, (512, 512) for everything else
        transform      : albumentations pipeline for train, None for val/test
        processor      : HuggingFace processor for CLIPSeg/SAM, None for others
        neg_prob       : chance of using a cross-class negative prompt during training
    """

    def __init__(
        self,
        stems,
        processed_root,
        prompts_df,
        split,
        image_size=(512, 512),
        transform=None,
        processor=None,
        neg_prob=0.3,
    ):
        self.stems = stems
        self.processed_root = Path(processed_root)
        self.image_size = image_size
        self.transform = transform
        self.processor = processor
        self.neg_prob = neg_prob if split == "train" else 0.0
        self.split = split

        split_df = prompts_df[prompts_df["split"] == split].copy()
        self.prompt_map = split_df.set_index("stem")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # route to the correct subfolder based on stem prefix
        dataset = "drywall" if stem.startswith("drywall") else "cracks"
        img_path = self.processed_root / dataset / "images" / f"{stem}.jpg"
        mask_path = self.processed_root / dataset / "masks" / f"{stem}.png"

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # INTER_LINEAR for images, INTER_NEAREST for masks to preserve binary values
        H, W = self.image_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # augmentation only on train — val/test always get clean originals
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # pick a prompt for this sample
        row = self.prompt_map.loc[stem]
        pos_prompts = row["positive_prompts"].split(";")
        neg_prompts = row["negative_prompts"].split(";")

        use_negative = (self.neg_prob > 0.0) and (random.random() < self.neg_prob)

        if use_negative:
            # cross-class prompt — expected output is an empty mask
            prompt = random.choice(neg_prompts)
            target_mask = np.zeros_like(mask)
        else:
            prompt = random.choice(pos_prompts)
            target_mask = mask

        # convert to tensor
        if self.processor is not None:
            # CLIPSeg and SAM use the HuggingFace processor for normalisation
            inputs = self.processor(
                images=img,
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
            pixel_values = inputs["pixel_values"].squeeze(0)
            input_ids = inputs["input_ids"].squeeze(0)
        else:
            # SegFormer and Mask2Former — manual ImageNet normalisation
            pixel_values = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            pixel_values = _imagenet_normalise(pixel_values)
            input_ids = None

        mask_tensor = torch.from_numpy(target_mask).float() / 255.0

        return {
            "pixel_values": pixel_values,  # (C, H, W)
            "mask": mask_tensor,  # (H, W)  values in {0.0, 1.0}
            "prompt": prompt,  # str
            "input_ids": input_ids,  # tensor or None
            "stem": stem,  # useful for saving predictions
        }


def _imagenet_normalise(tensor):
    # standard ImageNet normalisation for models without a HuggingFace processor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _make_weighted_sampler(stems):
    # drywall is the minority class at ~18% of the dataset
    # give it 4.5x higher sampling probability to balance classes per epoch
    drywall_count = sum(1 for s in stems if s.startswith("drywall"))
    cracks_count = len(stems) - drywall_count
    ratio = cracks_count / max(drywall_count, 1)

    weights = []
    for stem in stems:
        if stem.startswith("drywall"):
            weights.append(ratio)  # minority class upsampled
        else:
            weights.append(1.0)

    weights = torch.tensor(weights, dtype=torch.float)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def build_loaders(cfg, prompts_df, splits, transform_fn, processor=None):
    """
    Builds train, val, and test DataLoaders from a config dict.

    Args:
        cfg          : dict with keys: processed_root, image_size, batch_size, neg_prob
        prompts_df   : DataFrame from prompts.csv
        splits       : dict from splits.json
        transform_fn : callable(split, image_size) -> albumentations pipeline or None
        processor    : HuggingFace processor or None

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    image_size = tuple(cfg["image_size"])
    datasets = {}
    train_stems = splits["drywall"]["train"] + splits["cracks"]["train"]

    for split in ("train", "val", "test"):
        stems = splits["drywall"][split] + splits["cracks"][split]
        datasets[split] = SegDataset(
            stems=stems,
            processed_root=cfg["processed_root"],
            prompts_df=prompts_df,
            split=split,
            image_size=image_size,
            transform=transform_fn(split, image_size),
            processor=processor,
            neg_prob=cfg.get("neg_prob", 0.3),
        )

    # weighted sampler on train so drywall and cracks are seen equally
    # shuffle=True is replaced by the sampler — they can't coexist
    train_sampler = _make_weighted_sampler(train_stems)

    drywall_n = sum(1 for s in train_stems if s.startswith("drywall"))
    cracks_n = len(train_stems) - drywall_n
    print(
        f"Train sampler  drywall={drywall_n}  cracks={cracks_n}  ratio={cracks_n / max(drywall_n, 1):.1f}:1"
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg["batch_size"],
        sampler=train_sampler,  # replaces shuffle=True
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate,
    )

    return train_loader, val_loader, test_loader


def _collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    prompts = [b["prompt"] for b in batch]
    stems = [b["stem"] for b in batch]

    input_ids = None
    if batch[0]["input_ids"] is not None:
        # pad to longest sequence in the batch — prompts tokenise to different lengths
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded = []
        for b in batch:
            ids = b["input_ids"]
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            padded.append(ids)
        input_ids = torch.stack(padded)

    return {
        "pixel_values": pixel_values,
        "mask": masks,
        "prompt": prompts,
        "input_ids": input_ids,
        "stem": stems,
    }
