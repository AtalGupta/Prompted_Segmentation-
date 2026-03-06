import numpy as np
import torch


# ── threshold for converting predicted probabilities to binary mask ────────────
DEFAULT_THRESHOLD = 0.5


def compute_iou(pred, target, threshold=DEFAULT_THRESHOLD, smooth=1e-6):
    """
    Intersection over Union for binary segmentation.

    Args:
        pred      : torch.Tensor (B, 1, H, W) raw logits
                    or (B, H, W) probabilities
        target    : torch.Tensor (B, H, W) float in {0.0, 1.0}
        threshold : binarisation threshold
        smooth    : smoothing to avoid div-by-zero on empty masks

    Returns:
        mean IoU across the batch (scalar float)
    """
    pred_bin = (torch.sigmoid(pred).squeeze(1) >= threshold).float()

    pred_flat = pred_bin.contiguous().view(pred_bin.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_dice(pred, target, threshold=DEFAULT_THRESHOLD, smooth=1e-6):
    """
    Dice coefficient (F1 score) for binary segmentation.

    Args:
        pred      : torch.Tensor (B, 1, H, W) raw logits
        target    : torch.Tensor (B, H, W) float in {0.0, 1.0}
        threshold : binarisation threshold
        smooth    : smoothing to avoid div-by-zero on empty masks

    Returns:
        mean Dice across the batch (scalar float)
    """
    pred_bin = (torch.sigmoid(pred).squeeze(1) >= threshold).float()

    pred_flat = pred_bin.contiguous().view(pred_bin.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def compute_precision_recall(pred, target, threshold=DEFAULT_THRESHOLD, smooth=1e-6):
    """
    Precision and Recall for binary segmentation.

    Precision = TP / (TP + FP)  — of all pixels predicted positive, how many are correct
    Recall    = TP / (TP + FN)  — of all actual positives, how many did we find

    Returns:
        (precision, recall) tuple of scalar floats
    """
    pred_bin = (torch.sigmoid(pred).squeeze(1) >= threshold).float()

    pred_flat = pred_bin.contiguous().view(pred_bin.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return precision.mean().item(), recall.mean().item()


def compute_metrics(pred, target, threshold=DEFAULT_THRESHOLD):
    """
    Compute all metrics in one call.

    Returns:
        dict with keys: iou, dice, precision, recall
    """
    iou = compute_iou(pred, target, threshold)
    dice = compute_dice(pred, target, threshold)
    precision, recall = compute_precision_recall(pred, target, threshold)

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


class MetricTracker:
    """
    Accumulates metrics across batches in an epoch, then averages.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            pred = model(batch)
            tracker.update(pred, batch["mask"])
        epoch_metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._iou = []
        self._dice = []
        self._precision = []
        self._recall = []
        self._loss = []

    def update(self, pred, target, loss=None):
        """
        Args:
            pred   : (B, 1, H, W) raw logits — kept on whatever device they're on
            target : (B, H, W) float masks
            loss   : scalar loss value (optional)
        """
        with torch.no_grad():
            metrics = compute_metrics(pred.detach(), target.detach())

        self._iou.append(metrics["iou"])
        self._dice.append(metrics["dice"])
        self._precision.append(metrics["precision"])
        self._recall.append(metrics["recall"])

        if loss is not None:
            self._loss.append(loss if isinstance(loss, float) else loss.item())

    def compute(self):
        """Returns averaged metrics dict for the epoch."""
        result = {
            "iou": round(float(np.mean(self._iou)), 4),
            "dice": round(float(np.mean(self._dice)), 4),
            "precision": round(float(np.mean(self._precision)), 4),
            "recall": round(float(np.mean(self._recall)), 4),
        }
        if self._loss:
            result["loss"] = round(float(np.mean(self._loss)), 4)
        return result

    def pretty(self, prefix=""):
        """One-line string for printing during training."""
        m = self.compute()
        parts = [f"{prefix}loss={m['loss']:.4f}"] if "loss" in m else []
        parts += [
            f"iou={m['iou']:.4f}",
            f"dice={m['dice']:.4f}",
            f"prec={m['precision']:.4f}",
            f"rec={m['recall']:.4f}",
        ]
        return "  ".join(parts)


class BestModelTracker:
    """
    Tracks whether the current epoch achieved a new best validation metric
    and whether early stopping should trigger.

    Usage:
        tracker = BestModelTracker(metric="iou", patience=7)
        for epoch in range(max_epochs):
            val_metrics = ...
            if tracker.is_best(val_metrics):
                save_checkpoint(model, "best.pth")
            if tracker.should_stop():
                break
    """

    def __init__(self, metric="iou", patience=7, min_delta=1e-4):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0

    def is_best(self, metrics):
        """Returns True if this is the best epoch so far."""
        current = metrics[self.metric]
        if current > self.best + self.min_delta:
            self.best = current
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self):
        """Returns True when patience is exhausted."""
        return self.counter >= self.patience

    def status(self):
        return f"best_{self.metric}={self.best:.4f}  no_improve={self.counter}/{self.patience}"
