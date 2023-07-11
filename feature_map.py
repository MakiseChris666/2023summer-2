import torch
from mnist import models
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

model = models.getModel('vgg19')
model.load_state_dict(torch.load('./ckpt/vgg19'))
model.eval()
print(model)

trainData = MNIST('./data/mnist', download = True, transform = transforms.ToTensor())
trainLoader = DataLoader(trainData, batch_size = 1, shuffle = False, pin_memory = True)

ftmaps = [None] * 5

def getHook(idx):
    def hook(module, input, output):
        ftmaps[idx] = output.detach().cpu()
    return hook

# conv1 = model.get_submodule('1.conv1')
# conv1.register_forward_hook(getHook(0))
# layer1 = model.get_submodule('1.layer1')
# layer1.register_forward_hook(getHook(1))
avgpool = model.get_submodule('1.avgpool')
avgpool.register_forward_hook(getHook(0))

avgVecs = [torch.zeros(2048) for _ in range(10)]
for v in avgVecs:
    v[...] = 1 / math.sqrt(2048)
momentum = 0.95

for x, label in tqdm(trainLoader, total = len(trainLoader), ncols = 120):
    # print(x.shape)
    y = model(x)

    # ftmap = ftmaps[0][0]
    # ftmap = torch.sqrt(torch.sum(ftmap * ftmap, dim = 0))
    # print(ftmaps[0][0].shape)
    ftvec = ftmaps[0][0].squeeze()
    idx = label.item()
    avgVecs[idx] = avgVecs[idx] * momentum + ftvec * (1 - momentum)

    # print(label)

    # plt.figure()
    # plt.imshow(ftmap.numpy())
    # plt.show()

for i, avg in enumerate(avgVecs):
    torch.save(avg, './data/templates/' + str(i) + '.pth')