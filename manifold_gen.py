# In this file we define various manifold generation scheme where we can control the complexity of generated manifolds

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def generate_sphere(N, n_0, d):
  Base_kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
  x_ker = torch.zeros((N, N))
  for id_N in range(N):
    x_ker[id_N, ,id_N-1] = kappa
    x_ker[id_N, ,id_N+1] = kappa
  x = []
  gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
  return x

def generate_clean(N, n_0, d, curvature, sphere_restriction=False):
    x_base = generate_sphere(N, n_0, d)
    K = matrix()
    return x

def noise_addition(x, std, sphere_restriction=False):
    z = std*torch.randn(x.shape)
    return x+z

