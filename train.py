from mnist import mnist_train, models
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
import torch

model = models.getModel('vgg19')
opt = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()
# scheduler = OneCycleLR(opt, 3e-4, epochs = 10, steps_per_epoch = 1875, div_factor = 3, final_div_factor = 10)
scheduler = ExponentialLR(opt, 0.8)
mnist_train.train(model, opt, criterion, modelname = 'vgg19')