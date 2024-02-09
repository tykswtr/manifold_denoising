# In this document we introduce various network structure

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


#Define network
class FC(nn.Module):

    def __init__(self, n_0, n, L, activation=F.relu):
        super(FC, self).__init__()
        self.input_dim = n_0
        self.width = n
        self.L = L
        self.fc_layer = nn.ModuleList()
        self.fc_layer.append(nn.Linear(n_0, n, bias=False))
        self.alphas = []
        self.activation = activation
        for l in range(L-1):
          self.fc_layer.append(nn.Linear(n, n, bias=False))
          # self.alphas.append(nn.zeros)
        self.fc_layer.append(nn.Linear(n, 1, bias=False))

    def forward(self, x):
        for l in range(self.L):
          x = self.activation(self.fc_layer[l](x))
        x = self.fc_layer[L](x)
        return x

def init_weights(m):
  if isinstance(m, nn.Linear):
          torch.nn.init.normal_(m.weight.data, 0, np.sqrt(2/m.weight.data.shape[0]))