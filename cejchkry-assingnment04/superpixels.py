import cv2 as cv
import numpy as np
from cv2.ximgproc import (createSuperpixelSLIC, createSuperpixelLSC, 
                          createSuperpixelSEEDS, SLIC, SLICO)

def get_reference_color(img):
    # nechá uživatele vybrat roi
    roi_rect = cv.selectROI("Select Reference Sample", img, showCrosshair=True, fromCenter=False)
    cv.destroyWindow("Select Reference Sample")

    if roi_rect[2] == 0 or roi_rect[3] == 0:
        raise ValueError("nebyl vybrán roi")

    # z vstupního obrázku se vyřízne čtverec pixelů
    x, y, w, h = roi_rect
    roi_sample = img[int(y):int(y+h), int(x):int(x+w)]
    
    roi_sample_lab = cv.cvtColor(roi_sample, cv.COLOR_BGR2LAB)
    mean_val = cv.mean(roi_sample_lab)[:3]
    
    # spočítá průměr
    return np.array(mean_val)

def generate_heatmap(dist_map):
    #normaliyace vstupu
    norm_dist = cv.normalize(dist_map, None, 0, 255, cv.NORM_MINMAX)
    norm_dist = norm_dist.astype(np.uint8)
    
    # vytvoří heatmap a vrátí ji
    return cv.applyColorMap(norm_dist, cv.COLORMAP_JET)

def run_segmentation(path):
    # načte obraz
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"kde je obraz? no není.\t {path}")

    # referenční barva z uživatelova vstupu
    try:
        sample_mean_lab = get_reference_color(img)
    except ValueError as e:
        print(e)
        return

    # obraz b LABu
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    height, width, channels = lab_img.shape

    # jednotlivé algoritmy pro superpixely a jejich parametry -- nutno odkomentovat
    methods = {
        #"SLICO": createSuperpixelSLIC(lab_img, algorithm=SLICO, region_size=30, ruler=15.0),
        #"LSC": createSuperpixelLSC(lab_img, region_size=30, ratio=0.075),
        #"SEEDS": createSuperpixelSEEDS(width, height, channels, num_superpixels=8000, num_levels=20)#num_superpixels, int num_levels

        #"SLICO": createSuperpixelSLIC(lab_img, algorithm=SLICO, region_size=20, ruler=100.0),
        #"LSC": createSuperpixelLSC(lab_img, region_size=20, ratio=0.5),
        #"SEEDS": createSuperpixelSEEDS(width, height, channels, 6000, 10) 
    }

    threshold_dist = 85.0  # threshold pro segmentaci

    for name, sp in methods.items():
        print(name)

        #spustí algoritmy pro superpixely
        if name == 'SEEDS':
            sp.iterate(lab_img, 450)
        else:
            sp.iterate()
        
        labels = sp.getLabels()
        num_superpixels = sp.getNumberOfSuperpixels()

        # inicializace mask a map
        mask_selected = np.zeros((height, width), dtype=np.uint8)
        distance_map = np.zeros((height, width), dtype=np.float32)

        # vypočte vzdálenost (pro heatmap) a segmentaci
        for i in range(num_superpixels):
            mask_current = (labels == i)
            
            # průměrná barva superpixelu
            mean_color = cv.mean(lab_img, mask=mask_current.astype(np.uint8))[:3]
            
            # euklidovská vzdálenost
            dist = np.linalg.norm(np.array(mean_color) - sample_mean_lab)
            
            # mapa vzdáleností pro heatmap
            distance_map[mask_current] = dist

            # threshold
            if dist < threshold_dist:
                mask_selected[mask_current] = 255

        # následuje vizualizace
        result_visual = img.copy()
        result_visual[mask_selected == 0] = (result_visual[mask_selected == 0] * 0.4).astype(np.uint8)
        contours, _ = cv.findContours(mask_selected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(result_visual, contours, -1, (0, 255, 0), 2)
        
        heatmap_visual = generate_heatmap(distance_map)

        cv.imshow(f'{name} - Segmentation', result_visual)
        cv.imshow(f'{name} - Distance Heatmap', heatmap_visual)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_segmentation('files/stainski67Roychowdhury03.jpg')