import torch
from torch import nn
from torchvision.models import resnet18, vgg19
from torch import onnx

backbone = vgg19
model = nn.Sequential(
    nn.UpsamplingBilinear2d(scale_factor = 8),
    nn.Conv2d(1, 3, 1),
    backbone(num_classes = 10)
)

onnx.export(model, torch.zeros([1, 1, 28, 28]), './models/vgg19.pth')
