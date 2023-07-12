from mnist import models
import torch
import torch.nn.functional as F

model = models.getModel('resnet50')
model.load_state_dict(torch.load('./ckpt/resnet50-aug-30-exp'))
model.eval()

x = torch.zeros((1, 1, 28, 28))
y = model(x)

print(F.softmax(y))