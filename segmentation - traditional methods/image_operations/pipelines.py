from image_operations.filters import *
from image_operations.monadic_operations import *
from image_operations.advanced_operations import *
import cv2 as cv


def roi_binarization1(img: np.ndarray) -> np.ndarray:
    im = img.copy()
    im = erosion(im, erosion_size=1, interations_no=3)
    im = gaussian_blur(im, ksize=5, sigma=0)
    im = negate(im)
    im = adjust_gamma(im, gamma=0.16)
    im = otsu(im)
    im = opening(im, size=4, shape=cv.MORPH_RECT, iterations_no=1)
    return im


def pipeline1(img: np.ndarray) -> np.ndarray:
    im = img.copy()
    im = adjust_brightness(im, brightness=0.4)
    im = adjust_gamma(im, gamma=0.1)
    im = clahe(im, clip_limit=3, tile_grid=16)
    for _ in range(30):
        im = median_filter(im, ksize=3)
    im = negate(im)
    im = otsu(im)
    im = opening(im, 10)
    return im


def pipeline2(img: np.ndarray) -> np.ndarray:
    im = img.copy()
    im = adjust_brightness(im, brightness=0.6)
    im = logarithmic_scale(img, s=5.0)
    im = gaussian_blur(im, ksize=31, sigma=10)
    im = negate(im)
    im = otsu(im)
    im = opening(im, size=3, shape=cv.MORPH_RECT, iterations_no=13)
    return im


def roi_binarization2(img: np.ndarray) -> np.ndarray:
    im = img.copy()
    im = adjust_contrast(im, contrast=3)
    im = adjust_contrast(im, contrast=2)
    im = quantization(im, q=2.13)
    im = opening(im, size=3, iterations_no=1)
    im = negate(im)
    im = otsu(im)
    return im
