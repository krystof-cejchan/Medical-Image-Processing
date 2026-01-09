from enum import StrEnum, auto
import cv2 as cv
import numpy as np

from utils.convertors import to_uint8

class BF(StrEnum):
    B_MEAN = auto()
    B_GAUSSIAN = auto()
    F_MEDIAN = auto()
    F_BILATERAL = auto()

    
def mean_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv.blur(img, (ksize, ksize))


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float | int = 0) -> np.ndarray:
    return cv.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv.medianBlur(img, ksize)

def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    return cv.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def canny(img:np.ndarray,threshold1:float,_, threshold2:float):
    img = to_uint8(img)
    edges= cv.Canny(img, threshold1, threshold2)
    return edges

def diff_of_gaus(img: np.ndarray, sigma: float = 1.0, k: float = 1.6) -> np.ndarray:
    s1 = max(0.1, float(sigma))
    s2 = max(s1 + 1e-6, float(k) * s1)

    g1 = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=s1, sigmaY=s1, borderType=cv.BORDER_REPLICATE)
    g2 = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=s2, sigmaY=s2, borderType=cv.BORDER_REPLICATE)
    return g1 - g2


def laplacian_of_gaus(img: np.ndarray, sigma: float = 1.0, ksize_lap: int = 3) -> np.ndarray:

    s = max(0.1, float(sigma))

    g = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=s, sigmaY=s, borderType=cv.BORDER_REPLICATE)
    lap = cv.Laplacian(g, ddepth=cv.CV_32F, ksize=ksize_lap, borderType=cv.BORDER_DEFAULT)
    return lap