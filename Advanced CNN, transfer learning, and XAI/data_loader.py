from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.params import IMG_SIZE


class CiliaClassificationDataset(Dataset):
    """
    Upravený Dataset, který přijímá seznam souborů (file_list)
    místo toho, aby sám prohledával složku.
    """
    def __init__(self, file_list, angles=[90, 180, 270]):
        super().__init__()
        
        self.files = file_list

        if len(self.files) == 0:
            raise RuntimeError("Dataset byl inicializován s prázdným file_list!")

        self.img_size = IMG_SIZE
        
        self.angles = list(angles) if angles else [0]
        self.k = len(self.angles)

    def __len__(self):
        return len(self.files) * self.k

    def _load_image(self, path):
        img = (Image.open(path)
               .convert("L")
               .resize((self.img_size, self.img_size), Image.BILINEAR))
        return img

    def __getitem__(self, idx):
        base_idx = idx // self.k
        angle = self.angles[idx % self.k]

        img_path, cls = self.files[base_idx]
        img_pil = self._load_image(img_path)

        if angle != 0:
            img_pil = img_pil.rotate(
                angle,
                resample=Image.BILINEAR,
                expand=False,
                center=(self.img_size / 2, self.img_size / 2),
                fillcolor=0,
            )

        img = np.asarray(img_pil, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img)[None, ...]   # shape (1, H, W)
        label_t = torch.tensor(cls, dtype=torch.long)

        return img_t, label_t