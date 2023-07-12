import cv2 as cv
import numpy as np
import torch
from torch import nn
import detect.core as det
from mnist import models

img = cv.imread('./data/numbers.jpg')
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

resImg = det.detectFromRois2(img, model, needPreproccess = False, windows = np.array([[50, 25], [40, 20]]))
cv.imshow('', resImg)
cv.waitKey(0)