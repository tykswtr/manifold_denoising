import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.transforms as transforms
from torch.autograd import Variable
# from logger import Logger
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

# from manifold_gen import generate_clean, noise_addition
from manifold_gen import generate_clean, add_noise

import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.datasets as datasets

from util import AverageMeter, weights_init


import matplotlib.pyplot as plt

torch.manual_seed(180417)

# parser = argparse.ArgumentParser()
# parser.add_argument('--lr-epochs', type=str, default='(np.power(3, np.arange(1, 9)) + 1)/ 2')
# parser.add_argument('--lrf', type=float, default=np.power(10, -1/7))
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--lrgeo', type=float, default=1e-3)
# parser.add_argument('--wd', type=float, default=0)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--slope', type=float, default=0.1)
# parser.add_argument('--tb-number', type=int, default=0)
# parser.add_argument('--tb-path', type=str, default='tb-mnist-vae')
# parser.add_argument('--epochs', type=int, default=4000)
# parser.add_argument('--opt', type=str, default='adam')
#
# parser.add_argument('--chk', type=str, default='chks-mnist-vae')
# parser.add_argument('--gpu', type=str, default='0')
# # parser.add_argument('--n-layers', type=int, default=3)
# parser.add_argument('--batch-size', default=20, type=int)
# parser.add_argument('--test-batch-size', default=20, type=int)
# parser.add_argument('--val', default=False, action='store_true')
# parser.add_argument('--resume', default='', type=str)
# parser.add_argument('--w-recon', default=0, type=float)
# parser.add_argument('--ratio', default=1, type=float, help='ratio of computing neurons in next layer')
# parser.add_argument('--s-thresh', default=1e-10, type=float)
# parser.add_argument('--learnable-z', default=False, action='store_true')
# parser.add_argument('--s-init-mag', default=0.1, type=float)
# parser.add_argument('--ortho', default=False, action='store_true')
# parser.add_argument('--mmd-std', default=-1, type=float)
#
# parser.add_argument('--n-sample', type=int, default=20)

# args = parser.parse_args()
# print(args)
# sys.stdout.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# # logger = Logger(os.path.join(args.tb_path, str(args.tb_number)))
# chk_path = os.path.join(args.chk, str(args.tb_number))
# if not os.path.exists(chk_path):
#     os.makedirs(chk_path)

def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    return Variable(x).to(device)


# Hypothetical parameters for data generation
n_0 = 3
n_points = 1000  # Number of points to generate
d = 2  # Dimensionality of the manifold
sigma = 1.0  # Standard deviation for the Gaussian kernel
glue_num = 0  # Number of points to glue together
base_type = 'rectangle'  # Base manifold type
# base_type = 'torus'  # Base manifold type
# base_type = 'sphere'  # Base manifold type
# base_type = 'ball'  # Base manifold type

method = 'Ker'  # Kernel method for generating data
# grid_type = 'random'  # Random distribution of points on the manifold
grid_type = 'uniform'
sphere_restriction = False  # Not restricting points to a sphere

# Generating the data
data = generate_clean(n_0, d, n_points, sigma, glue_num, base_type, method, grid_type)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'data' is your 3D tensor or array of shape (n_points, 3)
data_np = data.numpy()  # Convert PyTorch tensor to NumPy array if necessary

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_np[0, :], data_np[1, :], data_np[2, :])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# Note: This call assumes that the necessary functions are implemented and correctly integrated.

#Generate Noisy Data
# n_0: ambient dimension
# N: number of data point
# d: manifold dimension
# curvature: rough curvature estimate

n_0 = 3
N = 100
d = 2

curvature = [1, 0.5]

width = 0.3
sigma = 100

glue_num = 3

# We generate grid of equal space.
N_d = int(np.ceil(N**(d**-1)))
N_shape = np.repeat(N_d, d)

# method = "Fourier"
method = "Ker"
# ker_choice = "Laplace"
ker_choice = "RBF"
# x = generate_clean(N, n_0, d, curvature)
# x = generate_clean_Fourier(N_shape, n_0, d, curvature)
x, glued_pair = generate_clean(N_shape, n_0, d, curvature, sigma, glue_num, width, method, ker_choice)
print(x.shape)

std = 0.

x = add_noise(x, std)
#
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# n_coord = 10
# nx, ny = n_coord, n_coord
# x = np.linspace(-1, 1, nx)
# y = np.linspace(-1, 1, ny)
# xv, yv = np.meshgrid(x, y)

# The following are for visualization purpose:

points = [np.linspace(-1, 1, n) for n in N_shape]
# Generate the grid using numpy's meshgrid function
# The * operator unpacks the points list into arguments for meshgrid
grids = np.meshgrid(*points, indexing='ij')
# Reshape the grids to have all coordinates for a point grouped together
grid_vectors = np.stack(grids, axis=-1).reshape(-1, len(N_shape))


# We first plot one dimension of my manifold
X = (x[0, :]).reshape((grids[0]).shape)
Y = (x[1, :]).reshape((grids[0]).shape)
Z = (x[2, :]).reshape((grids[0]).shape)

# Contour Plot of the function
plt.figure(figsize=(8, 6))
contour = plt.contourf(grids[0], grids[1], Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('FunctYion Visualization with Grid Overlay')

# Overlay the original grid points

plt.scatter(grid_vectors[:, 0], grid_vectors[:, 1], color='red', s=10, label='Grid Points')

colors = plt.cm.viridis(np.linspace(0, 1, len(glued_pair[:, 0, 0])))
# plt.scatter(glued_pair[:, 0, 0], glued_pair[:, 0, 1], c=colors, s=20, label='Glued Points')
# plt.scatter(glued_pair[:, 1, 0], glued_pair[:, 1, 1], c=colors, s=20, label='Glued Points')
for i in range(len(glued_pair[:, 0, 0])):
    plt.scatter(glued_pair[i, 0, 0], glued_pair[i, 0, 1], color=colors[i], label=f'Point {i}')
    plt.scatter(glued_pair[i, 1, 0], glued_pair[i, 1, 1], color=colors[i], label=f'Point {i}')
    plt.text(glued_pair[i, 0, 0], glued_pair[i, 0, 1], f' {i}', color='black', ha='right', va='bottom')
    plt.text(glued_pair[i, 1, 0], glued_pair[i, 1, 1], f' {i}', color='black', ha='right', va='bottom')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

plt.savefig("func_view_"+str(curvature)+".png")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z)
plt.show()
plt.savefig("manifold_"+str(curvature)+".png")

# surf = ax.plot_surface(xv, yv, x[2, :, :])
# plt.show()

# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()

