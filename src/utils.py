import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# checkpoint helpers


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Save model weights, optimizer state, epoch, and val metrics.

    Args:
        model     : nn.Module
        optimizer : torch optimizer
        epoch     : int
        metrics   : dict from MetricTracker.compute()
        path      : str or Path — where to save the .pth file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    clean_metrics = {k: float(v) for k, v in metrics.items()}

    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": clean_metrics,
        },
        path,
    )


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """
    Load a checkpoint saved by save_checkpoint.

    Args:
        model     : nn.Module — weights will be loaded in-place
        path      : str or Path
        optimizer : torch optimizer — pass to also restore optimizer state
        device    : device to map tensors onto

    Returns:
        (epoch, metrics) tuple
    """

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# prediction saving


def save_predictions(preds, stems, prompts, out_dir, threshold=0.5):
    """
    Save predicted binary masks to disk.

    Filename format: {stem}__{prompt_underscored}.png
    Example: cracks_00042__segment_crack.png

    Args:
        preds     : torch.Tensor (B, 1, H, W) raw logits
        stems     : list of str — image stems
        prompts   : list of str — text prompts used for each prediction
        out_dir   : path to output directory
        threshold : binarisation threshold
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = (torch.sigmoid(preds).squeeze(1) >= threshold).cpu().numpy()

    for i, (stem, prompt) in enumerate(zip(stems, prompts)):
        mask = (binary[i] * 255).astype(np.uint8)
        prompt_slug = prompt.replace(" ", "_")
        filename = f"{stem}__{prompt_slug}.png"
        cv2.imwrite(str(out_dir / filename), mask)


# visualization


def visualise_predictions(
    images, masks_gt, masks_pred, prompts, n=4, threshold=0.5, save_path=None
):
    """
    Plot a grid of: original image | ground truth mask | predicted mask.

    Args:
        images     : torch.Tensor (B, C, H, W) normalised
        masks_gt   : torch.Tensor (B, H, W) ground truth
        masks_pred : torch.Tensor (B, 1, H, W) raw logits
        prompts    : list of str
        n          : number of examples to show (max = batch size)
        threshold  : binarisation threshold for prediction
        save_path  : if provided, saves figure here instead of showing
    """
    n = min(n, images.size(0))
    fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        ax = ax[np.newaxis, :]

    pred_bin = (torch.sigmoid(masks_pred).squeeze(1) >= threshold).cpu().numpy()

    for i in range(n):
        # denormalise image for display
        img = _denormalise(images[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)

        gt = masks_gt[i].cpu().numpy()
        pred = pred_bin[i]

        ax[i, 0].imshow(img)
        ax[i, 0].set_title(f"Image\n{prompts[i]}", fontsize=9)
        ax[i, 0].axis("off")

        ax[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[i, 1].set_title("Ground Truth", fontsize=9)
        ax[i, 1].axis("off")

        ax[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_title("Prediction", fontsize=9)
        ax[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualise_overlay(image, mask_gt, mask_pred, prompt, threshold=0.5, save_path=None):
    """
    Overlay ground truth (green) and prediction (red) on the image.
    Overlap shown in yellow.

    Args:
        image     : np.ndarray (H, W, 3) uint8 RGB
        mask_gt   : np.ndarray (H, W) uint8 {0, 255}
        mask_pred : torch.Tensor (1, H, W) or (H, W) raw logits
        prompt    : str
        threshold : binarisation threshold
        save_path : optional save path
    """
    pred_bin = (
        (torch.sigmoid(mask_pred.squeeze()) >= threshold).cpu().numpy().astype(np.uint8)
    )
    gt_bin = (mask_gt > 127).astype(np.uint8)

    overlay = image.copy().astype(np.float32)

    gt_only = (gt_bin == 1) & (pred_bin == 0)
    pred_only = (pred_bin == 1) & (gt_bin == 0)
    both = (gt_bin == 1) & (pred_bin == 1)

    overlay[gt_only, :] = overlay[gt_only, :] * 0.4 + np.array([0, 200, 0]) * 0.6
    overlay[pred_only, :] = overlay[pred_only, :] * 0.4 + np.array([200, 0, 0]) * 0.6
    overlay[both, :] = overlay[both, :] * 0.4 + np.array([200, 200, 0]) * 0.6

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title(f'Original — "{prompt}"', fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Green=GT  Red=Pred  Yellow=Both", fontsize=10)
    axes[1].axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# training utilities


def get_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model):
    """Prints trainable and frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"Parameters  trainable : {trainable:,}")
    print(f"            frozen    : {frozen:,}")
    print(f"            total     : {total:,}")
    return trainable, frozen


def set_trainable(model, trainable=True):
    """Freeze or unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = trainable


def freeze_encoder(model, encoder_attr="encoder"):
    """
    Freeze only the encoder of a model.
    Most HuggingFace segmentation models expose the backbone as model.encoder.

    Args:
        model        : nn.Module
        encoder_attr : attribute name of the encoder submodule
    """
    encoder = getattr(model, encoder_attr, None)
    if encoder is None:
        print(f"WARNING: model has no attribute '{encoder_attr}' — nothing frozen")
        return
    for param in encoder.parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in encoder.parameters())
    print(f"Frozen encoder '{encoder_attr}': {frozen:,} parameters")


# results logging


def save_results(results, path):
    """
    Save a results dict to JSON.

    Args:
        results : dict — typically {method: {split: {metric: value}}}
        path    : str or Path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


def load_results(path):
    with open(path) as f:
        return json.load(f)


def print_results_table(results):
    """
    Pretty-print a results dict as a comparison table.

    Expected format:
        {
            "segformer": {"val": {"iou": 0.72, "dice": 0.81}, "test": {...}},
            "clipseg":   {"val": {"iou": 0.65, "dice": 0.77}, "test": {...}},
            ...
        }
    """
    methods = list(results.keys())
    splits = list(next(iter(results.values())).keys())
    metrics = list(next(iter(next(iter(results.values())).values())).keys())

    for split in splits:
        print(f"\n{'─' * 60}")
        print(f"  {split.upper()} SET")
        print(f"{'─' * 60}")
        header = f"  {'Method':<16}" + "".join(f"  {m:>10}" for m in metrics)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for method in methods:
            row = f"  {method:<16}"
            for m in metrics:
                val = results[method].get(split, {}).get(m, float("nan"))
                row += f"  {val:>10.4f}"
            print(row)


# internal helpers


def _denormalise(tensor):
    """Reverse ImageNet normalisation for display purposes."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean
