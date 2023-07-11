import cv2 as cv
import numpy as np

def rescale(img, size, target = 'min'):
    h, w = img.shape[:2]
    if target == 'min':
        factor = max(size / h, size / w)
    elif target == 'max':
        factor = min(size / h, size / w)
    else:
        raise NotImplementedError
    return cv.resize(img, None, fx = factor, fy = factor)

def centralPad(img, size = (28, 28)):
    h, w = img.shape[:2]
    vp = size[0] - h
    hp = size[1] - w
    padding = ((vp // 2, vp - vp // 2), (hp // 2, hp - hp // 2))
    img = np.pad(img, padding, constant_values = 0)
    return img