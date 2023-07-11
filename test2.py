import cv2 as cv
import torch
import random
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms


def randomTranslate(tensors):
    ret = []
    for tensor in tensors:
        tensor = tensor.squeeze()
        a = torch.where(tensor > 0)
        bd = [torch.min(a[0]), torch.min(a[1]), tensor.shape[0] - torch.max(a[0]), tensor.shape[1] - torch.max(a[1])]
        xt = random.randint(-bd[0], bd[2])
        yt = random.randint(-bd[1], bd[3])
        res = torch.zeros_like(tensor)
        res[max(xt, 0):min(28, xt + 28), max(yt, 0):min(28, 28 + yt)] = \
            tensor[max(0, -xt):min(28, 28 - xt), max(0, -yt):min(28, 28 - yt)]
        res = res[None, None, ...]
        ret.append(res)
    return torch.concat(ret, dim = 0)

trainData = MNIST('./data/mnist', download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(20), transforms.Lambda(randomTranslate)]))
trainLoader = DataLoader(trainData, batch_size = 8, shuffle = True)

for x, label in trainLoader:
    for xx in x:
        xx = xx.squeeze()
        cv.imshow('', (xx.numpy() * 255).astype('uint8'))
        cv.waitKey(0)