"""
In this file we provide data with given order and estimate properties of the manifold
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


def manifold_est(x, d):
    """
    Return geometric quantities of the manifold.

    :param x: input data, need to be represented in a given order. If x comes from a d-dimensional manifold,
        then x is a two part tensor, where the first part is the intrinsic coordinate and the second is the
         extrinsic coordinate.
    :param d: manifold dimension
    :return: geometric quantities including: curvature, separation distance, winding number.
    """
    curvature = 1
    return curvature
