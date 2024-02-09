import cv2 as cv
import numpy as np
import torch
from torch import nn

import detect.core as det
from mnist import models
import utils

low = np.array([156, 43, 46])
high = np.array([180, 255, 255])

img = cv.imread('./data/dim red.png')
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask = cv.inRange(img, low, high)
print(mask.shape)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


cv.imshow('', mask)
cv.waitKey(0)

model = models.getModel('vgg19')
model.load_state_dict(torch.load('./ckpt/vgg19'))
model.eval()
model = model.cuda()

resImg, _ = det.detectFromRois2(mask, model, needPreproccess = False, windows = np.array([[50, 20]]))
cv.imshow('', resImg)
cv.waitKey(0)