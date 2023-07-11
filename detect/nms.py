import numpy as np
import torch
from torchvision.ops.boxes import nms as tnms

def nms(boxes, labels, probs):
    boxes = torch.Tensor()
