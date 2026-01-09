from PIL import Image
import numpy as np


def to_uint8(img_f: np.ndarray) -> np.ndarray:
    """Float ⟨0,1⟩ -> uint8 ⟨0,255⟩ (for display/save)."""
    if img_f.ndim == 2:
        pass
    elif img_f.ndim == 3 and img_f.shape[2] == 1:
        img_f = img_f[..., 0]
    else:
        raise ValueError("Expected single-channel grayscale image.")
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)


def from_uint8(img_u8: np.ndarray) -> np.ndarray:
    """uint8 ⟨0,255⟩ -> float ⟨0,1⟩."""
    if img_u8.ndim == 2:
        pass
    elif img_u8.ndim == 3 and img_u8.shape[2] == 1:
        img_u8 = img_u8[..., 0]
    else:
        raise ValueError("Expected single-channel grayscale image.")
    return img_u8.astype(np.float32) / 255.0



def pil_from_float(img_f: np.ndarray, max_side: int = 640) -> Image.Image:
    """
    Convert to PIL Image + scaled for UI (keep aspect).
    """
    u8 = to_uint8(img_f)
    pil = Image.fromarray(u8, mode="L")
    w, h = pil.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    if scale != 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.NEAREST)
    return pil