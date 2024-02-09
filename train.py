import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.transforms as transforms
from torch.autograd import Variable
from logger import Logger
import numpy as np
import io
import argparse
import sys
import os
import math
import pdb
from collections import OrderedDict
import operator
import time
# custom
from networks import FC, init_weights

from manifold_gen import generate_clean, noise_addition


import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.datasets as datasets

from util import AverageMeter, weights_init


import matplotlib.pyplot as plt

torch.manual_seed(180417)

parser = argparse.ArgumentParser()
parser.add_argument('--lr-epochs', type=str, default='(np.power(3, np.arange(1, 9)) + 1)/ 2')
parser.add_argument('--lrf', type=float, default=np.power(10, -1/7))
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lrgeo', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--slope', type=float, default=0.1)
parser.add_argument('--tb-number', type=int, default=0)
parser.add_argument('--tb-path', type=str, default='tb-mnist-vae')
parser.add_argument('--epochs', type=int, default=4000)
parser.add_argument('--opt', type=str, default='adam')

parser.add_argument('--chk', type=str, default='chks-mnist-vae')
parser.add_argument('--gpu', type=str, default='0')
# parser.add_argument('--n-layers', type=int, default=3)
parser.add_argument('--batch-size', default=20, type=int)
parser.add_argument('--test-batch-size', default=20, type=int)
parser.add_argument('--val', default=False, action='store_true')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--w-recon', default=0, type=float)
parser.add_argument('--ratio', default=1, type=float, help='ratio of computing neurons in next layer')
parser.add_argument('--s-thresh', default=1e-10, type=float)
parser.add_argument('--learnable-z', default=False, action='store_true')
parser.add_argument('--s-init-mag', default=0.1, type=float)
parser.add_argument('--ortho', default=False, action='store_true')
parser.add_argument('--mmd-std', default=-1, type=float)

parser.add_argument('--n-sample', type=int, default=20)

args = parser.parse_args()
print(args)
sys.stdout.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger = Logger(os.path.join(args.tb_path, str(args.tb_number)))
chk_path = os.path.join(args.chk, str(args.tb_number))
if not os.path.exists(chk_path):
    os.makedirs(chk_path)

def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    return Variable(x).to(device)

def train(net, x, criterion=nn.MSELoss(), lr=1e-5, max_epoch=1000):
  net.train()
  optimizer = torch.optim.SGD(net.parameters(), lr=lr)
  losses = []
  y_init = net(x)
  for epoch in range(max_epoch):
      optimizer.zero_grad()
      y_hat = net(x)
      loss = criterion(y_hat, y)
      losses.append(loss)
      if epoch %100 == 0:
        torch.save(net.state_dict(), 'net_itr_'+str(epoch)+'.pt')
        print("iter: {}, loss: {}".format(epoch, loss))
      loss.backward()
      optimizer.step()
      if criterion(y_hat, y) < 1e-6:
        break
  return losses


#Generate Noisy Data
n_0 = 20
N = 400
d = 1
curvature = 3

x = generate_clean(N, n_0, d, curvature)

std = 0.1

x = noise_addition(x, std)

ns = [ t*500 for t in range(1, 3) ];
# ns = [500]
L = 10
repeat = 1

for idn, n in enumerate(ns):
  for r in range(repeat):
    thresh = 1.0/20/np.sqrt(n);
    criterion = nn.MSELoss()

    net = FC(n_0, n, L).to(device)
    net.apply(init_weights)
    torch.save(net.state_dict(), 'net_itr_0.pt')
    print("width: {} , repeat: {}".format(n, r))
    losses = train(net, x)

log_losses = [];
for loss in losses:
  log_losses.append(torch.log(loss).detach().numpy())
print(log_losses)
plt.plot(log_losses)