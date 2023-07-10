import cv2 as cv

def rescale(img, size, target = 'min'):
    h, w = img.shape[0, 1]
    if target == 'min':
        factor = max(size / h, size / w)
    elif target == 'max':
        factor = min(size / h, size / w)
    return cv.resize(img, None, fx = factor, fy = factor)

def centralPad(img, size = (28, 28)):
    h, w = img.shape[0, 1]
    vp = size[0] - h
    hp = size[1] - w
    padding = ((vp // 2, size[0] - vp // 2), (hp // 2, size[1] - hp // 2))
    img = np.pad(img, padding, constant_values = 0)
    return img