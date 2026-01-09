from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils import IMG_SIZE

DATA_SET_DIR = "./data"


def find_pairs(data_dir: Path | str = DATA_SET_DIR):
    """vyextrahuje dvojice odpovídajících obrázků

    Args:
        data_dir (Path | str, optional): adresář datasetu. Defaults to DATA_SET_DIR.

    Raises:
        RuntimeError: pokud pro nějaký obraz nelze najít odpovídající masku a vice versa

    Returns:
        tuple: dvojice - obraz k jeho masce
    """
    data_dir = Path(data_dir)
    imgs = sorted((data_dir / "images").glob("*.png"))
    masks = sorted((data_dir / "masks").glob("*.png"))

    mask_by_stem = {m.stem.replace("_mask", ""): m for m in masks}
    pairs = []
    for img in imgs:
        stem = img.stem
        if stem in mask_by_stem:
            pairs.append((img, mask_by_stem[stem]))
    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found under {data_dir}")
    return pairs


class CiliaSegDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = Path(DATA_SET_DIR),
        angles: list[int] = np.arange(90, 270, 90).tolist(),
    ):
        super().__init__()
        self.pairs = find_pairs(data_dir)
        self.img_size = IMG_SIZE
        self.angles = list(angles) if len(angles) > 0 else [0]
        self.k = len(self.angles)

    def __len__(self):
        return len(self.pairs) * self.k

    def _load_resized_pil(self, img_path: Path, mask_path: Path):
        img = (
            Image.open(img_path)
            .convert("L") # L znamená, že se bitmapa načte jako greyscale
            .resize((self.img_size, self.img_size), Image.BILINEAR)
        )
        mask = (
            Image.open(mask_path)
            .convert("L")
            .resize((self.img_size, self.img_size), Image.NEAREST)
        )
        return img, mask

    def _rotate_pair(self, img_pil: Image.Image, mask_pil: Image.Image, angle: float):
        img_rot = img_pil.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=False,
            center=(self.img_size / 2, self.img_size / 2),
            fillcolor=0,
        )
        mask_rot = mask_pil.rotate(
            angle,
            resample=Image.NEAREST,
            expand=False,
            center=(self.img_size / 2, self.img_size / 2),
            fillcolor=0,
        )
        return img_rot, mask_rot

    def __getitem__(self, idx):
        base_idx = idx // self.k
        angle = self.angles[idx % self.k]

        img_path, mask_path = self.pairs[base_idx]
        img_pil, mask_pil = self._load_resized_pil(img_path, mask_path)
        img_pil, mask_pil = self._rotate_pair(img_pil, mask_pil, angle)

        img = np.asarray(img_pil, dtype=np.float32) / 255.0
        mask = (np.asarray(mask_pil) > 127).astype(np.float32)

        img_t = torch.from_numpy(img)[None, ...]
        mask_t = torch.from_numpy(mask)[None, ...]  # 1, H, W

        return img_t, mask_t
