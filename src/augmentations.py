import albumentations as A


# CLIPSeg is sensitive to aggressive colour changes — use mild strength
# SAM benefits from elastic deformation for irregular crack shapes
# SegFormer and Mask2Former use the standard moderate suite
MODEL_CONFIGS = {
    "clipseg": {"colour_strength": "mild", "use_elastic": False},
    "segformer": {"colour_strength": "moderate", "use_elastic": False},
    "sam": {"colour_strength": "moderate", "use_elastic": True},
    "grounded_sam": {"colour_strength": "moderate", "use_elastic": True},
    "mask2former": {"colour_strength": "moderate", "use_elastic": False},
}


def get_transform(split, image_size, model="segformer"):
    """
    Returns an albumentations Compose pipeline for the given split and model.
    Val and test always return None — no augmentation on evaluation sets.

    Usage:
        transform = get_transform("train", (512, 512), model="segformer")
        augmented = transform(image=img, mask=mask)
        img, mask = augmented["image"], augmented["mask"]
    """
    if split in ("val", "test"):
        return None

    model_cfg = MODEL_CONFIGS.get(model, MODEL_CONFIGS["segformer"])
    colour_strength = model_cfg["colour_strength"]
    use_elastic = model_cfg["use_elastic"]

    transforms = (
        _build_spatial_transforms(image_size)
        + _build_colour_transforms(colour_strength)
        + _build_noise_transforms()
        + (_build_elastic_transforms() if use_elastic else [])
    )

    return A.Compose(transforms, additional_targets={"mask": "mask"})


def _build_spatial_transforms(image_size):
    # applied to both image and mask — albumentations keeps them aligned
    H, W = image_size
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        # replaces deprecated ShiftScaleRotate
        A.Affine(
            translate_percent=0.05,
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=0,
            p=0.4,
        ),
        # forces model to handle partial views of cracks and joints
        A.RandomResizedCrop(
            size=(H, W),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=0.3,
        ),
    ]


def _build_colour_transforms(strength="moderate"):
    # applied to image only — mask is binary so colour changes don't apply
    if strength == "mild":
        return [
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.05,
                p=0.4,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.2),
        ]

    return [
        A.ColorJitter(
            brightness=0.30,
            contrast=0.30,
            saturation=0.20,
            hue=0.10,
            p=0.5,
        ),
        A.RandomGamma(gamma_limit=(75, 125), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.ToGray(p=0.1),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.2),
    ]


def _build_noise_transforms():
    # simulate real-world camera conditions
    return [
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # updated API — std_range replaces var_limit
        A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
        # updated API — quality_range replaces quality_lower/quality_upper
        A.ImageCompression(
            compression_type="jpeg",
            quality_range=(70, 100),
            p=0.2,
        ),
        A.MotionBlur(blur_limit=5, p=0.1),
    ]


def _build_elastic_transforms():
    # used only for SAM and Grounded-SAM
    # helps with irregular crack shapes that benefit from non-linear deformation
    return [
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            border_mode=0,
            p=0.2,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=0,
            p=0.15,
        ),
    ]
