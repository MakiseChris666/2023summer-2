import numpy as np
import utils

def generate(minSize = 10, maxSize = 30, hwratio = 2.0, steps = 5, method = 'constant'):
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


def getWindows(img, windows, stride = 4):
    h, w = img.shape
    res = []
    loc = []
    for window in windows:
        for i in range(0, h - int(window[0]), stride):
            for j in range(0, w - int(window[1]), stride):
                roi = img[i:i + int(window[0]), j:j + int(window[1])]
                print(roi.shape)
                roi = utils.rescale(roi, 28, 'max')
                roi = utils.centralPad(roi)
                if np.sum(roi) / (roi.shape[0] * roi.shape[1]) < 0.01:
                    continue
                res.append(roi[None, None, ...])
                loc.append([i, j, i + int(window[0]), j + int(window[1])])
    res = np.concatenate(res, axis = 0)
    loc = np.array(loc)
    return res, loc