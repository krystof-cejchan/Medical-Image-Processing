import cv2 as cv
import numpy as np


def convert_to_lab(img):
    return cv.cvtColor(img, cv.COLOR_BGR2LAB)

def get_avg_lab_vector(image_path):
    img = cv.imread(image_path)

    lab_image = convert_to_lab(img)

    l_mean, a_mean, b_mean, _ = cv.mean(lab_image)
    
    return (l_mean, a_mean, b_mean), lab_image


