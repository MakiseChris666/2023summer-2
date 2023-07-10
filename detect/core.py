import cv2 as cv
from detect import window

def preprocess(img, size = 224):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape
    factor = max(size / h, size / w)
    img = cv.resize(img, None, fx = factor, fy = factor)
    edge = cv.Canny(img, 100, 200)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)
    return img

def detectFromRois(rois, model):
    pass