import numpy as np


def negate(img: np.ndarray)->np.ndarray:
    return np.subtract(1, img)

def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.1, gamma)
    out = np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)
    return np.clip(out, 0.0, 1.0)

def adjust_brightness(img: np.ndarray, brightness: float = 0.0) -> np.ndarray:
    if brightness < -1.0 or brightness > 1.0:
        raise ValueError("brightness musí být v rozsahu ⟨-1; 1⟩")
    return np.clip(img + float(brightness), 0.0, 1.0)

def adjust_contrast(img: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    return np.clip(img * contrast, 0.0, 1.0)


def non_linear_contrast(img:np.ndarray, alpha: float = 1.0) ->np.ndarray:
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha musí být v rozsahu (0; 1)")
    lt = np.zeros_like(img)
    gamma = 1/(1-alpha)
    left = (img<1/2)
    right = ~left
    lt[left]  = 0.5 * (2.0 * img[left]) ** gamma
    lt[right] = 1.0 - 0.5 * (2.0 - 2.0 * img[right]) ** gamma
    return lt

def logarithmic_scale(img:np.ndarray, s: float = 1.0) ->np.ndarray:
    if s <= -1:
        raise ValueError("s musí být v rozsahu (-1; inf)")
    return (np.log(1+img*s))/(np.log(1+s))

def quantization(img: np.ndarray, q: int = 1):
    if q <= 1:
        raise ValueError("q musí být v rozsahu (1; inf)")
    return (1/q)*np.floor(img*q)