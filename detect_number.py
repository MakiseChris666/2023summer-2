import cv2 as cv
import numpy as np
import torch
from torch import nn
import detect.core as det
from mnist import models

# img = cv.imread('./data/numbers.jpg')
img = cv.imread('./data/175806.png')
# img = cv.GaussianBlur(img, (5, 5), 0)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('', img)
cv.waitKey(0)
model = models.getModel('vgg19')
model.load_state_dict(torch.load('./ckpt/vgg19'))
model.eval()
model = model.cuda()
# model[-1].fc = nn.Sequential()

resImg, _ = det.detectFromRois2(img, model, needPreproccess = False, windows = np.array([[60, 30], [40, 30]]), weights = [1, 2], combine = True)
cv.imshow('', resImg)
cv.waitKey(0)