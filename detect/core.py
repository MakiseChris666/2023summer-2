import cv2 as cv
import numpy as np

from detect import window
import torch
from torchvision.ops.boxes import nms, box_iou
from tqdm import tqdm
import torch.nn.functional as F

templateVecs = []
templateVecsLen = []

def preprocess(img, size = 224):
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.GaussianBlur(img, (5, 5), 0)
    h, w = img.shape
    factor = max(size / h, size / w)
    img = cv.resize(img, None, fx = factor, fy = factor)
    edge = cv.Canny(img, 100, 200)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)
    return img

def _readTemplate():
    for i in range(10):
        templateVecs.append(torch.load('./data/templates/' + str(i) + '.pth').reshape((1, -1))) # 1 * 2048
        templateVecsLen.append(torch.sqrt(torch.sum(templateVecs[-1] * templateVecs[-1], dim = 1, keepdim = True))) # 1 * 1

@torch.no_grad()
def detectFromRois(img, model, thres = 0.99, needPreproccess = True):
    """
    model: need to be headless; output shape (256,)
    """
    if len(templateVecs) == 0:
        _readTemplate()
    device = next(model.parameters()).device
    if needPreproccess:
        img = preprocess(img) / 255
    else:
        img = img / 255
    windows = window.generate()
    rois, locs = window.getWindows(img, windows)
    rois = torch.Tensor(rois)
    rois = rois.to(device)
    ftmaps = []
    for i in tqdm(range(0, rois.shape[0], 32)):
        ftmaps.append(model(rois[i:min(i + 32, rois.shape[0])]))
    ftmaps = torch.concat(ftmaps, dim = 0)
    ftvecs = torch.flatten(ftmaps, start_dim = 1).cpu() # N * 2048
    print(ftvecs.shape)
    ftvecsLen = torch.sqrt(torch.sum(ftvecs * ftvecs, dim = 1, keepdim = True)) # N * 1
    print(ftvecsLen.shape)
    res = []
    for v, vl in zip(templateVecs, templateVecsLen):
        print(v.shape, vl.shape)
        res.append(torch.sum(v * ftvecs, dim = 1, keepdim = True) / (ftvecsLen * vl)) # N * 1
    res = torch.concat(res, dim = 1) # N * 10
    prob, num = torch.max(res, dim = 1)
    selected = prob > thres

    prob = prob[selected]
    locsT = torch.Tensor(locs)[selected]
    nmsed = nms(locsT, prob, 0.1)

    num = num[selected][nmsed].numpy()
    rois = rois[selected][nmsed].numpy()
    for r in rois:
        cv.imshow('', (r[0] * 255).astype('uint8'))
        cv.waitKey(0)
    print(num)
    selected = selected.numpy()
    locs = locs[selected][nmsed]
    resImg = img.copy()
    for loc in locs:
        cv.rectangle(resImg, (loc[1], loc[0]), (loc[3], loc[2]), color = (255, 0, 0))
    return resImg

class UnionSet:

    def __init__(self, n):
        self.num = n
        self.fa = np.arange(n)
        self.size = np.ones(n)

    def find(self, a):
        while a != self.fa[a]:
            a = self.fa[a]
        return a

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            self.size[b] += self.size[a]
            self.fa[a] = b
            return b
        else:
            self.size[a] += self.size[b]
            self.fa[b] = a
            return a

    def isRoot(self, a):
        return self.fa[a] == a

@torch.no_grad()
def detectFromRois2(img, model, thres = 0.9, needPreproccess = True, windows = None, verbose = True, weights = None,
                    combine = True):
    device = next(model.parameters()).device
    resImg = img.copy()
    if needPreproccess:
        img = preprocess(img) / 255
    else:
        img = img / 255
    if windows is None:
        windows = window.generate()
    rois, locs, roiweights = window.getWindows(img, windows, weights = weights)
    # for i in range(rois.shape[0]):
    #     roi = rois[i][0]
    #     x, y = np.where(roi > 0)
    #     x = x.reshape((-1, 1))
    #     y = y.reshape((-1, 1))
    #     coor = np.concatenate([x, y], axis = 1)
    #     center = np.mean(coor, axis = 0)
    #     center = (np.array([13.5, 13.5]) - center).astype('int32')
    #
    #     res = np.zeros_like(roi)
    #     res[max(center[0], 0):min(28, center[0] + 28), max(center[1], 0):min(28, 28 + center[1])] = \
    #         roi[max(0, -center[0]):min(28, 28 - center[0]), max(0, -center[1]):min(28, 28 - center[1])]
    #     cv.imshow('', (res * 255).astype('uint8'))
    #
    #     rois[i] = res[None, ...]

    rois = torch.Tensor(rois)
    rois = rois.to(device)
    scores = []
    for i in tqdm(range(0, rois.shape[0], 32)):
        scores.append(model(rois[i:min(i + 32, rois.shape[0])]))
    scores = torch.concat(scores, dim = 0).cpu()
    scores = F.softmax(scores, dim = 1)

    prob, num = torch.max(scores, dim = 1)
    prob[num == 7] = torch.sqrt(prob[num == 7])
    prob = prob ** torch.Tensor(roiweights)
    selected = prob > thres

    prob = prob[selected]
    locsT = torch.Tensor(locs)[selected]
    nmsed = nms(locsT, prob, 0.1)

    num = num[selected][nmsed]
    locs = locsT[nmsed]

    ious = box_iou(locs, locs)
    f = torch.ones(len(num), dtype = torch.bool)
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            if num[i] == num[j] and ious[i][j] > 0.05:
                f[j] = False
    locs = locs[f]
    num = num[f]

    if verbose:
        rois = rois[selected][nmsed][f].cpu().numpy()
        for r in rois:
            cv.imshow('', (r[0] * 255).astype('uint8'))
            cv.waitKey(0)
    print(num)
    locs = locs.numpy().astype('int32')
    num = num.numpy()

    us = UnionSet(locs.shape[0])
    if combine:
        for i in range(locs.shape[0]):
            for j in range(i + 1, locs.shape[0]):
                if abs((locs[i][0] + locs[i][2]) / 2 - (locs[j][0] + locs[j][2]) / 2) < (locs[i][2] - locs[i][0] + locs[j][2] - locs[j][0]) / 2 * 1.2 \
                    and abs((locs[i][1] + locs[i][3]) / 2 - (locs[j][1] + locs[j][3]) / 2) < (locs[i][3] - locs[i][1] + locs[j][3] - locs[j][1]) / 2 * 1.5:
                    us.union(i, j)

    united = [[] for _ in range(locs.shape[0])]
    for i in range(locs.shape[0]):
        united[us.find(i)].append((locs[i], num[i]))
    finalLocNum = []
    for u in united:
        if len(u) == 0:
            continue
        u = sorted(u, key = lambda x: x[0][1])
        bd = [0, 0, 0, 0]
        bd[0] = u[0][0][0]
        bd[1] = u[0][0][1]
        bd[2] = u[0][0][2]
        bd[3] = u[-1][0][3]
        unum = 0
        for x in u:
            unum = unum * 10 + x[1]
            bd[0] = min(bd[0], x[0][0])
            bd[2] = max(bd[2], x[0][2])
        finalLocNum.append((np.array(bd), unum))

    # for loc in locs:
    #     cv.rectangle(resImg, (loc[1], loc[0]), (loc[3], loc[2]), color = (255, 0, 0))
    for loc, n in finalLocNum:
        cv.rectangle(resImg, (loc[1], loc[0]), (loc[3], loc[2]), color = (255, 0, 0))
        cv.putText(resImg, str(n), (loc[1], loc[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        print(n)
    return resImg, finalLocNum