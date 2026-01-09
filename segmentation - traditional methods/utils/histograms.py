from matplotlib import pyplot as plt
import numpy as np


def compute_hist(img: np.ndarray, bins: int = 256) -> np.ndarray:
    counts, _ = np.histogram(img, bins=bins, range=(0.0, 1.0))
    hist = counts.astype(np.float32) / img.size
    return hist

def compute_cdf(img: np.ndarray, bins: int = 256) -> np.ndarray:
    h = compute_hist(img, bins)
    cdf = np.cumsum(h, dtype=np.float64).astype(np.float32)
    return cdf

def show_histograms_and_cdfs(img_original: np.ndarray, img_edited: np.ndarray,
                             title: str = "Histograms and CDFs"):
    hist   = compute_hist(img_original)
    cdf    = compute_cdf(img_original)
    hist_e = compute_hist(img_edited)
    cdf_e  = compute_cdf(img_edited)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    fig.suptitle(title)

    (ax1, ax2), (ax3, ax4) = axes

    # Original
    ax1.plot(hist)
    ax1.set_title("Histogram - Original")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, len(hist)-1)

    ax2.plot(cdf)
    ax2.set_title("CDF - Original")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(0, len(cdf)-1)

    # Edited
    ax3.plot(hist_e)
    ax3.set_title("Histogram - Edited")
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Count")
    ax3.set_xlim(0, len(hist_e)-1)

    ax4.plot(cdf_e)
    ax4.set_title("CDF – Edited")
    ax4.set_xlabel("Intensity")
    ax4.set_ylabel("Probability")
    ax4.set_xlim(0, len(cdf_e)-1)

    plt.show()
