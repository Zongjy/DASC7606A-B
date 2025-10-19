import logging
import random
from pathlib import Path
import shutil
import hashlib
from typing import List, Tuple

import cv2
import numpy as np
import torch
import albumentations as A
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VAL_RATIO = 0.1


class ImageAugmenter:
    """Class to handle image augmentation operations using Albumentations."""

    def __init__(
        self,
        seed: int = 42,
    ):
        """
        Initialize the ImageAugmenter.

        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
        """
        self.seed = seed

        self._set_seed()

        self.pipeline = A.Compose([
            A.OneOf([
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(10, 15, 10, p=1.0),
                A.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.2), p=1.0),
            ], p=0.9),

            A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.RandomCrop(height=32, width=32, p=1.0),
            A.HorizontalFlip(p=0.5),
            # A.OneOf([
            #     A.GaussianBlur(blur_limit=3, p=1.0),
            #     A.MotionBlur(blur_limit=3, p=1.0),
            #     A.GaussNoise(std_range=(0.012, 0.022),   # ≈ 3~5.5 像素级 std
            #         mean_range=(0.0, 0.0),
            #         per_channel=False,
            #         noise_scale_factor=1.0,
            #         p=0.12,
            #     ),
            # ], p=0.15),

            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(6, 10),
                hole_width_range=(6, 10),
                fill=0,
                p=0.25
            ),
        ])

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to the input image.

        Args:
            image: Input image as a NumPy array (H x W x C).

        Returns:
            Augmented image as a NumPy array.
        """
        out = self.pipeline(image=image)
        return out["image"]

def _deterministic_split(files, val_ratio=VAL_RATIO, seed=42):
    """按文件名+seed 做稳定划分，避免同源图跨集合。"""
    def key(p: Path):
        h = hashlib.md5((p.name + str(seed)).encode()).hexdigest()
        return int(h, 16)
    files = sorted([p for p in files if p.is_file()], key=key)
    n_val = max(1, int(len(files) * val_ratio)) if len(files) > 0 else 0
    return files[n_val:], files[:n_val]   # train, val

def _norm_dir(out_dir: Path):
    """
    若 out_dir 以 'train' 结尾：使用其父目录做 augmented 根目录，
    在其下创建 train/ 与 val/；
    否则认为 out_dir 是 augmented 根目录，在其下创建 train/ 与 val/。
    """
    out_dir = Path(out_dir)
    if out_dir.name == "train":
        root = out_dir.parent
    else:
        root = out_dir
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    return root

def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> None:
    """
    Backward-compatible wrapper for legacy code.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
    """
    augmenter = ImageAugmenter(seed=seed)

    input_dir = Path(input_dir)
    output_root = _norm_dir(Path(output_dir))

    # 清空旧输出（可选：按需保留）
    if (output_root / "train").exists():
        shutil.rmtree(output_root / "train")
    if (output_root / "val").exists():
        shutil.rmtree(output_root / "val")
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    for cls_dir in classes:
        cls = cls_dir.name
        out_tr = output_root / "train" / cls
        out_va = output_root / "val" / cls
        out_tr.mkdir(parents=True, exist_ok=True)
        out_va.mkdir(parents=True, exist_ok=True)

        originals = sorted([p for p in cls_dir.iterdir() if p.is_file()])
        train_files, val_files = _deterministic_split(originals, 0.2, seed)


        # val：只复制原图（干净分布）
        for src in val_files:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img is None: 
                continue
            cv2.imwrite(str(out_va / src.name), img)

        # train：可复制原图 + 生成 K 份增强图
        for src in train_files:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img is None: 
                continue

            # save_original_train:
            cv2.imwrite(str(out_tr / src.name), img)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i in range(augmentations_per_image):
                aug_img = augmenter.transform(image=img_rgb)
                aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                out_name = f"{src.stem}_aug{i}.png"
                cv2.imwrite(str(out_tr / out_name), aug_bgr)

    print(f"[OK] Offline dataset ready at: {output_root} (train/ + val/)")
