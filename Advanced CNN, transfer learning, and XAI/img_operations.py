
from collections import deque
from datetime import datetime
import os
import cv2 as cv
import numpy as np
from utils.convertors import from_uint8, to_uint8
from utils.pixel import Pixel
from typing import Literal, Optional, Tuple


def hist_equalization(img: np.ndarray, bins: int = 256) -> np.ndarray:
    counts, _ = np.histogram(img, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(counts).astype(np.float32)
    if cdf[-1] == 0:
        return img.copy()

    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0.0
    denom = (cdf[-1] - cdf_min)
    cdf_norm = (cdf - cdf_min) / denom if denom > 0 else np.zeros_like(cdf)

    lut = cdf_norm
    idx = np.clip((img * (bins - 1)).astype(np.int32), 0, bins - 1)
    eq = lut[idx]
    return eq.astype(np.float32)

def clahe(img_f: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    tile_grid = max(2, int(tile_grid))
    clip_limit = max(0.01, float(clip_limit))

    img_u8 = to_uint8(img_f)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    out_u8 = clahe.apply(img_u8)
    return from_uint8(out_u8)

def erosion(img, erosion_size, erosion_shape:int=cv.MORPH_ELLIPSE, interations_no:int=1):
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    return cv.erode(img, element, iterations= interations_no)
    
def dilatation(img,dilatation_size,dilation_shape:int=cv.MORPH_ELLIPSE, iterations_no:int=1):
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    return cv.dilate(img, element, iterations=iterations_no)

def opening(img,size,shape:int=cv.MORPH_ELLIPSE, iterations_no:int=1):
    element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv.morphologyEx(img, cv.MORPH_OPEN, element, iterations=iterations_no)

def closing(img,size,shape:int=cv.MORPH_ELLIPSE, iterations_no:int=1):
    element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv.morphologyEx(img, cv.MORPH_CLOSE, element, iterations=iterations_no)


def otsu(img_f: np.ndarray):
    img_u8 = to_uint8(img_f)

    _, bin_u8 = cv.threshold(img_u8, 0, 255, cv.THRESH_OTSU)
    bin_f = from_uint8(bin_u8)
    return bin_f

def find_contours(img_f: np.ndarray,
                  min_area: int = 0,
                  mode: int = cv.RETR_EXTERNAL,
                  method: int = cv.CHAIN_APPROX_NONE) -> tuple[np.ndarray, list]:
    img_u8 = to_uint8(img_f)

    contours, _ = cv.findContours(img_u8, mode, method)

    if min_area and min_area > 0:
        contours = [c for c in contours if cv.contourArea(c) >= min_area]

    return from_uint8(img_u8), contours

def split_middle_cell(img, line_thickness=2):
    _, contours = find_contours(img, 30) #najde jádra
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0  

    best = None
    best_abs_d = float("inf")

    for c in contours: # pomocí pointPolygonTest, najde tu konturu, která je nejblíže středu obrázku
        d = cv.pointPolygonTest(c, (cx, cy), True)
        if abs(d) < best_abs_d:
            best_abs_d = abs(d)
            best = c

    x, y, bw, bh = cv.boundingRect(best)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)

    y_line = int((y0 + y1) / 2)
    cv.line(img, (x0, y_line), (x1, y_line), (0, 0, 0), thickness=line_thickness) #do středu "namaluje" černou úsečku

    return img

        


def contours_to_rect_images(img_f: np.ndarray,
                            contours: list,
                            pad: int = 0) -> list[np.ndarray]:
    h, w = img_f.shape[:2]
    rect_imgs = []
    img_u8 = to_uint8(img_f)

    for c in contours:
        x, y, bw, bh = cv.boundingRect(c)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)

        crop_u8 = img_u8[y0:y1, x0:x1].copy()
        crop_f = from_uint8(crop_u8)
        if crop_f.size > 0:
            rect_imgs.append(crop_f)

    return rect_imgs

def save_rect_images(rect_imgs: list[np.ndarray], out_dir: str, prefix: str = "roi"):
    os.makedirs(out_dir, exist_ok=True)
    for i, roi in enumerate(rect_imgs, start=1):
        cv.imwrite(os.path.join(out_dir, f"{prefix}_{i:03d}.png"), to_uint8(roi))
    return len(rect_imgs)

def save_image(img: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().timestamp()
    return cv.imwrite(os.path.join(out_dir, f"edited_img_${timestamp}.png"), to_uint8(img))

def image_reconstruct(marker: np.ndarray | None, img: np.ndarray) -> np.ndarray:
   """
   Perform morphological reconstruction by dilation.

   Parameters:
       marker (np.ndarray): The marker image (seed).
       mask (np.ndarray): The mask image (constraint).

   Returns:
       np.ndarray: Reconstructed image.
   """
   if not marker:
       marker = cv.erode(img, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), iterations=2)
   if marker.shape != img.shape:
       raise ValueError("Marker and mask must have the same dimensions")

   # Ensure images are in the same type
   marker = marker.astype(img.dtype)

   # Create a working copy
   reconstructed = marker.copy()

   # Use a queue for pixels to update
   q = deque()
   rows, cols = reconstructed.shape

   # Initialize queue with pixels where marker < mask
   for r in range(rows):
       for c in range(cols):
           if reconstructed[r, c] < img[r, c]:
               q.append(Pixel(r, c))

   # Neighbor offsets (4-connectivity)
   neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

   while q:
       p = q.popleft()
       for dr, dc in neighbors:
           rr, cc = p.row + dr, p.col + dc
           if 0 <= rr < rows and 0 <= cc < cols:
               new_val = min(img[rr, cc], max(reconstructed[rr, cc], reconstructed[p.row, p.col]))
               if new_val > reconstructed[rr, cc]:
                   reconstructed[rr, cc] = new_val
                   q.append(Pixel(rr, cc))

   return reconstructed


def separate_cell_with_black_line(
    img: np.ndarray,
    *,
    rel_peak: float = 0.5,              # 0.35–0.60: lower → more seeds → easier split
    line_width: int = 1,                # separator thickness in pixels
    component: Literal["largest","centered"] = "largest",
    center: Optional[Tuple[int,int]] = None  # if component="centered", you can pass (cx, cy)
) -> tuple[np.ndarray, np.ndarray]:
  
    bin_u8 = to_uint8(img)
    H, W = bin_u8.shape
    bin_out = bin_u8.copy()

    # ---- pick the target component (the merged 'middle cell') ----
    num, labels, stats, centroids = cv.connectedComponentsWithStats(bin_u8, connectivity=8)
    if num <= 2:
        # Nothing or only one blob — nothing to split
        return bin_out, np.zeros_like(bin_u8)

    # choose component id
    if component == "largest":
        areas = stats[1:, cv.CC_STAT_AREA]
        cid = 1 + int(np.argmax(areas))
    else:
        if center is None:
            center = (W // 2, H // 2)
        cx, cy = center
        d2 = []
        for i in range(1, num):
            cxi, cyi = centroids[i]
            d2.append((i, (cxi - cx) ** 2 + (cyi - cy) ** 2))
        cid = min(d2, key=lambda t: t[1])[0]

    cell_mask = (labels == cid).astype(np.uint8) * 255
    if cv.countNonZero(cell_mask) == 0:
        return bin_out, np.zeros_like(bin_u8)

    # ---- distance transform on that component only ----
    dist = cv.distanceTransform(cell_mask, cv.DIST_L2, 5).astype(np.float32)
    if dist.max() <= 0:
        return bin_out, np.zeros_like(bin_u8)
    dist_norm = cv.normalize(dist, None, 0.0, 1.0, cv.NORM_MINMAX)

    # ---- seed extraction: local maxima ∩ high distance ----
    se3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dil = cv.dilate(dist, se3)
    local_max = dist >= (dil - 1e-6)

    thr = max(0.0, min(1.0, rel_peak)) * float(dist_norm.max())
    seeds = (local_max & (dist_norm >= thr) & (cell_mask > 0)).astype(np.uint8) * 255
    seeds = cv.morphologyEx(seeds, cv.MORPH_OPEN, se3)

    # if too few seeds (need at least 2), relax
    if cv.countNonZero(seeds) < 2:
        seeds = (local_max & (dist_norm >= 0.35 * dist_norm.max()) & (cell_mask > 0)).astype(np.uint8) * 255
        seeds = cv.morphologyEx(seeds, cv.MORPH_OPEN, se3)

    nseeds, markers = cv.connectedComponents(seeds)
    if nseeds < 2:
        # cannot define a cut; return as-is
        return bin_out, np.zeros_like(bin_u8)

    # ---- watershed domain: just around the chosen component ----
    # Use negative distance as topography (basins at object centers, ridge at boundaries)
    neg_dist = cv.normalize(-dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    color_topo = cv.merge([neg_dist, neg_dist, neg_dist])

    # Unknown zone where watershed can place the cut
    sure_bg = cv.dilate(cell_mask, se3, iterations=1)
    unknown = cv.subtract(sure_bg, seeds)

    markers_ws = markers.astype(np.int32)
    markers_ws[unknown == 255] = 0  # 0 = unknown

    cv.watershed(color_topo, markers_ws)  # in-place; -1 are watershed boundaries

    # ---- internal boundary only (inside the chosen component) ----
    line_mask = ((markers_ws == -1) & (cell_mask > 0)).astype(np.uint8) * 255

    if cv.countNonZero(line_mask) == 0:
        # No internal boundary found → nothing drawn
        return bin_out, line_mask

    # ---- draw separator: thicken if requested, then subtract from that component ----
    if line_width > 1:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (line_width, line_width))
        thick_line = cv.dilate(line_mask, k, iterations=1)
    else:
        thick_line = line_mask

    # apply only within that component
    thick_line &= cell_mask
    bin_out[thick_line > 0] = 0  # draw black separator

    return from_uint8(bin_out), from_uint8(line_mask)
