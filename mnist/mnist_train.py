from torchvision.datasets.mnist import MNIST
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from mnist import models
import random

def train(model, opt, criterion, modelname, epochs = 10, scheduler = None, schedulerMode = 'epoch', augmentation = True):

    model = model.cuda()
    criterion = criterion.cuda()

    def randomTranslate(tensor):
        tensor = tensor.squeeze()
        a = torch.where(tensor > 0)
        bd = [torch.min(a[0]), torch.min(a[1]), tensor.shape[0] - torch.max(a[0]),
              tensor.shape[1] - torch.max(a[1])]
        xt = random.randint(-bd[0], bd[2])
        yt = random.randint(-bd[1], bd[3])
        res = torch.zeros_like(tensor)
        res[max(xt, 0):min(28, xt + 28), max(yt, 0):min(28, 28 + yt)] = \
            tensor[max(0, -xt):min(28, 28 - xt), max(0, -yt):min(28, 28 - yt)]
        return res[None, ...]

    if augmentation:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(20),
            transforms.Lambda(randomTranslate)
        ])
    else:
        trans = transforms.ToTensor()

    trainData = MNIST('./data/mnist', download = True, transform = trans)
    trainLoader = DataLoader(trainData, batch_size = 32, shuffle = True, pin_memory = True)
    valData = MNIST('./data/mnist', download = True, train = False, transform = trans)
    valLoader = DataLoader(valData, batch_size = 32, shuffle = True, pin_memory = True)

    for epoch in range(epochs):
        lossSum = 0
        process = tqdm(enumerate(trainLoader), desc = f'Epoch {epoch}', total = len(trainLoader), ncols = 120)
        for i, (x, label) in process:
            x = x.cuda()
            label = label.cuda()
            y = model(x)
            label = F.one_hot(label, num_classes = 10).float()
            loss = criterion(y, label)
            lossSum += loss.item()
            process.set_postfix_str('loss = %.6f' % (lossSum / (i + 1)))

            opt.zero_grad()
            loss.backward()
            opt.step()

            if scheduler is not None and schedulerMode == 'iteration':
                scheduler.step()

        valProcess = tqdm(enumerate(valLoader), desc = f'Epoch {epoch} val', total = len(valLoader), ncols = 120)
        tp = 0
        psum = 0

        with torch.no_grad():
            for i, (x, label) in valProcess:
                x = x.cuda()
                label = label.cuda()
                y = model(x)
                ynum = torch.argmax(y, dim = 1)
                tp += torch.sum(ynum == label)
                psum += x.shape[0]
                valProcess.set_postfix_str('Acc = %.3f' % (tp / psum))

        if scheduler is not None and schedulerMode == 'epoch':
            scheduler.step()

    torch.save(model.state_dict(), './ckpt/' + modelname)