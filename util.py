import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init(m):
    print("m: ", m)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        r = 1/math.sqrt(m.in_features)
        m.weight.data.uniform_(-r, r)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def surf_plot(data):
    """Plot 2d proj of data."""
    return
