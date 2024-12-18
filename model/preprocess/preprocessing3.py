import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
#import math
from math import sqrt

def normalize_pixel(x, v0, v, m, m0):
    """
    From Handbook of Fingerprint Recognition pg 133
    Normalize job used by Hong, Wan and Jain(1998)
    similar to https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf equation 21
    :param x: pixel value
    :param v0: desired variance
    :param v: global image variance
    :param m: global image mean
    :param m0: desired mean
    :return: normilized pixel
    """
    dev_coeff = sqrt((v0 * ((x - m)**2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff

def normalize(im, m0, v0):
    m = np.mean(im)
    v = np.std(im) ** 2
    (y, x) = im.shape
    normilize_image = im.copy()
    for i in range(x):
        for j in range(y):
            normilize_image[j, i] = normalize_pixel(im[j, i], v0, v, m, m0)

    return normilize_image

def clahe_contrast(a):
    clahe = cv2.createCLAHE(clipLimit=5.0)
    a = clahe.apply(a)
    return a

def crop_using_morphology(image):
    
    # Apply thresholding
    _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing to connect ridge lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours and crop based on the largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    return image

def preprocessing_normal(a):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if len(a.shape) > 2 else a#1

    a = normalize(a.copy(), float(100), float(100))#2

    a = clahe_contrast(a)#3

    #4 binarization
    thresh, a= cv2.threshold(a, 255, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    a = crop_using_morphology(a)#5

    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA) #6 inter_AREA phù hợp giảm kích thước ảnh
     
    return a