import torch
from mnist import models
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

model = models.getModel('resnet50')
model.load_state_dict(torch.load('./ckpt/resnet50-reg1e-3-onecycle'))
model.eval()

trainData = MNIST('./data/mnist', download = True, transform = transforms.ToTensor())
trainLoader = DataLoader(trainData, batch_size = 1, shuffle = False, pin_memory = True)

ftmaps = [None] * 5

def getHook(idx):
    def hook(module, input, output):
        ftmaps[idx] = output.detach().cpu()
    return hook

conv1 = model.get_submodule('1.conv1')
conv1.register_forward_hook(getHook(0))
layer1 = model.get_submodule('1.layer1')
layer1.register_forward_hook(getHook(1))

for x, label in trainLoader:
    print(x.shape)
    y = model(x)

    ftmap = ftmaps[0][0]
    ftmap = torch.sqrt(torch.sum(ftmap * ftmap, dim = 0))

    print(label)

    plt.figure()
    plt.imshow(ftmap.numpy())
    plt.show()