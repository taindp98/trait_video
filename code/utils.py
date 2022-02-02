import cv2
import numpy as np

def resize_aspect_ratio(img, size = 1280, interp=cv2.INTER_LINEAR):
    """
    resize image and keep aspect ratio
    """
    if len(img.shape) == 2:
        h,w = img.shape
    elif len(img.shape) == 3:
        h,w,_ = img.shape
    else:
        return None

    new_w = size
    new_h = h*new_w//w    
    img_rs = cv2.resize(img.copy(), (new_w, new_h), interpolation=interp)
    return img_rs