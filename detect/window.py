import numpy as np
import utils

def generate(minSize = 20, maxSize = 120, hwratio = 1.0, steps = 6, method = 'constant'):
    if method == 'constant':
        h = np.linspace(minSize * hwratio, maxSize * hwratio, steps).reshape((-1, 1))
        w = np.linspace(minSize, maxSize, steps).reshape((-1, 1))
        return np.concatenate([h, w], axis = 1)
    elif method == 'exponential':
        h = np.logspace(minSize * hwratio, maxSize * hwratio, steps).reshape((-1, 1))
        w = np.logspace(minSize, maxSize, steps).reshape((-1, 1))
        return np.concatenate([h, w], axis = 1)
    else:
        raise NotImplementedError


def getWindows(img, windows, stride = 10):
    h, w = img.shape
    res = []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            for window in windows:
                roi = img[i:i + window[0], j:j + window[1]]
                roi = utils.rescale(roi, 28, 'max')
                roi = utils.centralPad(roi)
                res.append(roi[None, None, ...])
    res = np.concatenate(res, axis = 0)
    return res