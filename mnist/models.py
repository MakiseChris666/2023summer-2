from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19
from torch import nn

_MODELS = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'vgg19': vgg19,
    'vgg16': vgg16
}

def getModel(name):
    backbone = _MODELS[name]

    model = nn.Sequential(
        nn.Conv2d(1, 3, 1),
        backbone(num_classes = 10)
    )

    if name.startswith('vgg'):
        model = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 8),
            model
        )

    return model