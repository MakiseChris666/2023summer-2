from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from mnist import models
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score

def val(modelname, ckpt):

    model = models.getModel(modelname)
    model.load_state_dict(torch.load(ckpt))
    model = model.cuda()

    valData = MNIST('./data/mnist', download = True, train = False, transform = transforms.ToTensor())
    valLoader = DataLoader(valData, batch_size = 1, shuffle = False, pin_memory = True)

    valProcess = tqdm(enumerate(valLoader), desc = f'Val', total = len(valLoader), ncols = 120)
    yPred = []

    with torch.no_grad():
        for i, (x, label) in valProcess:
            x = x.cuda()
            y = model(x)
            ynum = torch.argmax(y, dim = 1)
            yPred.append(ynum)

    yPred = torch.concat(yPred)
    yPred = yPred.cpu().numpy()
    yTrue = valData.targets.numpy()

    acc = accuracy_score(yTrue, yPred)
    prec = precision_score(yTrue, yPred, average = 'weighted')
    recall = recall_score(yTrue, yPred, average = 'weighted')
    print('Overall Acc.: %.4f' % acc)
    print('Overall Prec.: %.4f' % prec)
    print('Overall Recall: %.4f' % recall)
    print()

    print('Prec. for each class: ', *['%.4f' % i for i in precision_score(yTrue, yPred, average = None)])
    print('Recall for each class: ', *['%.4f' % i for i in recall_score(yTrue, yPred, average = None)])