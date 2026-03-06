import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    We use the soft version (no thresholding) so gradients flow
    through the predicted probabilities during training.

    Args:
        smooth : additive smoothing term to avoid division by zero
                 when both prediction and target are all-zero
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred   : (B, 1, H, W) raw logits  or  (B, H, W)
        # target : (B, H, W) float in {0.0, 1.0}

        pred = torch.sigmoid(pred).squeeze(1)  # (B, H, W) probabilities

        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.

    BCE handles per-pixel classification well but struggles with
    heavy class imbalance (most pixels are background).
    Dice directly optimises the overlap metric we care about.
    Combining them gives stable training with good mask quality.

    Loss = bce_weight * BCE + dice_weight * Dice

    Args:
        bce_weight  : weight for BCE term (default 0.5)
        dice_weight : weight for Dice term (default 0.5)
        smooth      : smoothing for Dice (default 1.0)
        pos_weight  : BCE positive class weight — use to counter class
                      imbalance within a mask (foreground << background).
                      Computed as (total_pixels / foreground_pixels) roughly.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)

        if pos_weight is not None:
            pw = torch.tensor([pos_weight])
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # pred   : (B, 1, H, W) raw logits
        # target : (B, H, W) float in {0.0, 1.0}

        # BCE expects same shape as pred
        bce_loss = self.bce(pred.squeeze(1), target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Focal loss + Dice loss.

    Use this instead of BCEDiceLoss when foreground pixels are very sparse
    (e.g. thin hairline cracks covering < 5% of the image).
    Focal loss down-weights easy background pixels so the model focuses
    on hard foreground examples.

    Args:
        alpha      : focal loss balance parameter (default 0.25)
        gamma      : focal loss focusing parameter (default 2.0)
        dice_weight: weight for Dice term (default 0.5)
        smooth     : smoothing for Dice (default 1.0)
    """

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = 1.0 - dice_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        # pred   : (B, 1, H, W) raw logits
        # target : (B, H, W) float in {0.0, 1.0}

        pred_squeezed = pred.squeeze(1)
        prob = torch.sigmoid(pred_squeezed)

        # focal weight
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_w = alpha_t * (1.0 - p_t) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(
            pred_squeezed, target, reduction="none"
        )
        focal_loss = (focal_w * bce).mean()
        dice_loss = self.dice(pred, target)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def get_loss(cfg):
    """
    Factory — returns the correct loss from config.

    Config keys:
        loss_type   : "bce_dice" | "focal_dice"
        bce_weight  : float (bce_dice only)
        dice_weight : float
        pos_weight  : float or null (bce_dice only)
        focal_alpha : float (focal_dice only)
        focal_gamma : float (focal_dice only)

    Usage in training notebook:
        from losses import get_loss
        criterion = get_loss(cfg)
    """

    loss_type = cfg.get("loss_type", "bce_dice")

    if loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=cfg.get("bce_weight", 0.5),
            dice_weight=cfg.get("dice_weight", 0.5),
            pos_weight=cfg.get("pos_weight", None),
        )

    if loss_type == "focal_dice":
        return FocalDiceLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
            dice_weight=cfg.get("dice_weight", 0.5),
        )

    raise ValueError(
        f"Unknown loss_type: {loss_type}. Choose 'bce_dice' or 'focal_dice'."
    )
