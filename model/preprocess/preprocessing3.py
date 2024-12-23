import cv2
import numpy as np
#import math
def clahe_contrast(a):
    clahe = cv2.createCLAHE(clipLimit=5.0)
    a = clahe.apply(a)
    return a

def preprocessing_normal(a):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if len(a.shape) > 2 else a#1

    a = cv2.GaussianBlur(a, (3,3), 0)#2
    
    a = clahe_contrast(a)#3

    #4 binarization
    thresh, a= cv2.threshold(a, 255, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA) #5 inter_AREA phù hợp giảm kích thước ảnh
     
    return a