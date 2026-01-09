from pathlib import Path
import numpy as np
import cv2 as cv
from utils.convertors import *
from img_operations import *
import random

DATA_SET_DIR = "./data"

def find_pairs(data_dir: Path | str = DATA_SET_DIR):
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

def split_list(a_list):
    s = len(a_list)//2
    return a_list[:s], a_list[s:]


pairs_left, pairs_right = split_list(find_pairs())

out_inp = Path(f"{DATA_SET_DIR}/orig_inpainted/inpainted")
out_ori = Path(f"{DATA_SET_DIR}/orig_inpainted/original")
out_inp.mkdir(parents=True, exist_ok=True)
out_ori.mkdir(parents=True, exist_ok=True)

for i, (img_path, mask_path) in enumerate(pairs_left):
    img = cv.imread(str(img_path))
    mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    mask_01 = from_uint8(mask)
    mask_01 = dilatation(mask_01, 20)


    h, w = mask.shape

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    selected_contours = []
    if contours: 
        num_to_select = min(5, len(contours))
        selected_contours = random.sample(contours, num_to_select)

    random_contour_mask = np.zeros((h, w), dtype=np.uint8)

    if selected_contours:
        cv.drawContours(random_contour_mask, selected_contours, -1, (255), thickness=cv.FILLED)

    random_contour_mask = from_uint8(random_contour_mask)
    random_contour_mask = dilatation(random_contour_mask, 4)
    random_contour_mask = to_uint8(random_contour_mask)
    dst = cv.inpaint(img, random_contour_mask, 8, cv.INPAINT_TELEA)

    cv.imwrite(str(out_inp / f"{i}.png"), dst)
    cv.imwrite(str(out_ori / f"{i}.png"), img)