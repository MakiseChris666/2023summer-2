import cv2 as cv
from mnist import models
import torch
import torch.nn.functional as F

model = models.getModel('resnet50')
model.load_state_dict(torch.load('./ckpt/resnet50-reg1e-3-onecycle'))
model.eval()
model[-1].fc = torch.nn.Sequential()

t1 = torch.load('./data/templates/1.pth')
print(t1.shape)
t1len = torch.sqrt(torch.sum(t1 * t1))

img = torch.zeros((1, 1, 28, 28))
# img = cv.imread('./data/1.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img = img.reshape((1, 1, 28, 28)) / 255
# img = torch.Tensor(img)

y = model(img).reshape(-1)
print(y.shape)
ylen = torch.sqrt(torch.sum(y * y))

sim = torch.sum(t1 * y) / (ylen * t1len)
print(sim)
