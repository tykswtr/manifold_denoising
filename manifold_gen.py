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

import matplotlib.pyplot as plt


# def generate_sphere(N, n_0, d):
#   Base_kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
#   x_ker = torch.zeros((N, N))
#   for id_N in range(N):
#     x_ker[id_N, id_N-1] = kappa
#     x_ker[id_N, id_N+1] = kappa
#   x = []
#   gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
#   return x


# def generate_clean(N, n_0, d, curvature, sphere_restriction=False):
#     """
#     Generate data from a smooth manifold
#
#     :param N: number of generated data point
#     :param n_0: ambient dimension, dimension of the data
#     :param d: intrinsic dimension, dimension of the manifold
#     :param curvature: "rough" estimate of the manifold curvature
#     :param sphere_restriction:
#     :return:
#     """
#     # x_base = generate_sphere(N, n_0, d)
#     x = np.zeros((N, N))
#     # K = matrix()
#     return x


def noise_addition(x, std, sphere_restriction=False):
    """Add independent Gaussian noise to data."""
    z = std * np.random.randn(*x.shape)
    return x+z


def generate_clean(N_shape, n_0, d, curvature, sigma, method, kernel_choice, sphere_restriction=False):

    if method == "Fourier":
        gen_func = generate_Fourier_func
    elif method == "Ker":
        gen_func = generate_Ker_func
    else:
        gen_func = "Fourier"

    # define intrinsic coordinates

    grid_vectors = generate_grid(N_shape)

    return gen_func(n_0, curvature, sigma, N_shape, grid_vectors, kernel_choice)


def generate_grid(N_shape):
    # Generate points for each dimension
    points = [np.linspace(-1, 1, n) for n in N_shape]
    # Generate the grid using numpy's meshgrid function
    # The * operator unpacks the points list into arguments for meshgrid
    grids = np.meshgrid(*points, indexing='ij')
    # Reshape the grids to have all coordinates for a point grouped together
    grid_vectors = np.stack(grids, axis=-1).reshape(-1, len(N_shape))
    return grid_vectors

# # Generate Function value through given kernel
# def generate_clean_ker(N, n_0, d, curvature, sphere_restriction=False):
#     n_coord = 10
#     nx, ny = n_coord
#     x = np.linspace(-1, 1, nx)
#     y = np.linspace(-1, 1, ny)
#     xv, yv = np.meshgrid(x, y)
#     data = np.zeros(())
#     for coor_id in np.range(n_0):
#         data[:, coor_id] = 0
#     return data


# Generate Fourier function with degree K
def generate_Fourier_func(n_0, K, N_shaped, z_in):
    data = np.zeros((n_0, *N_shaped))

    for coor_id in np.arange(n_0):
        data[coor_id, :, :] = generate_single_Fourier_func(K, N_shaped, z_in)
    return data


def generate_single_Fourier_func(K, N_shaped, z_in):
    z = np.zeros(N_shaped)
    for k1 in np.arange(np.floor(np.sqrt(K)).astype(int) + 1):
        for k2 in np.arange(np.floor(np.sqrt(K - k1 ** 2)).astype(int) + 1):
            fourier_coeff = np.random.randn(1)[0]
            for i in np.arange(N_shaped[0]):
                for j in np.arange(N_shaped[1]):
                    z[i, j] += fourier_coeff * np.cos(k1 * np.pi * z_in[0][i] + k2 * np.pi * z_in[1][j])
    return z


# Generate Kernel function with smoothness parameter K
def generate_Ker_func(n_0, K, sigma, N_shape, grid_vectors, kernel_choice):
    data = np.zeros((n_0, *N_shape))

    # Define kernel
    thresh = 2 * (1./np.max(N_shape))**2

    kernel_matrix = compute_kernel_matrix(grid_vectors, thresh, kernel_choice, sigma=sigma, spiky_value=K)

    # optional: display the kernel matrix
    plt.imshow(kernel_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar()  # Show a color bar to indicate the scale
    plt.title("Gaussian Kernel Matrix Visualization")
    plt.xlabel("Index of Point")
    plt.ylabel("Index of Point")
    plt.show()
    # folder_path = "/path/to/your/folder"
    # full_path = f"{folder_path}/{filename}"
    plt.savefig("kernel_"+str(K)+".png")

    data = np.random.multivariate_normal(np.zeros(len(grid_vectors)), kernel_matrix, n_0)
    return data


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-euclidean_distance(x, y) ** 2 / (2 * sigma ** 2))


def l1_distance(point_i, point_j):
    return np.sum(np.abs(point_i - point_j))


def laplace_kernel(point_i, point_j, sigma=1.0):
    return np.exp(-l1_distance(point_i, point_j) / sigma)


def is_adjacent(point_i, point_j, thresh):
    # Compute the absolute differences in each dimension
    diff = np.sum((point_i - point_j) ** 2)
    # Two points are adjacent if exactly one dimension differs by 1 and the rest are 0
    return diff <= thresh


def compute_adjacency_matrix(grid_vectors, thresh):
    num_points = len(grid_vectors)
    adjacency_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        adjacency_matrix[i, i] = 1
        for j in range(i + 1, num_points):  # No need to check j < i because the matrix is symmetric
            # if is_adjacent(grid_vectors[i], grid_vectors[j]):
            if np.sum((grid_vectors[i] - grid_vectors[j]) ** 2) <= thresh:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Symmetrically assign since adjacency is bidirectional
    return adjacency_matrix


def compute_kernel_matrix(grid_vectors, thresh, kernel_choice, sigma=1.0, spiky_value=1.0):
    n_points = len(grid_vectors)
    kernel_matrix = np.zeros((n_points, n_points))
    adjacency = compute_adjacency_matrix(grid_vectors, thresh)

    for i in range(n_points):
        for j in range(n_points):
            if kernel_choice=="Laplace":
                kernel_matrix[i, j] = laplace_kernel(grid_vectors[i], grid_vectors[j], sigma)
            else:
                kernel_matrix[i, j] = gaussian_kernel(grid_vectors[i], grid_vectors[j], sigma)
            if adjacency[i, j] == 1:
                kernel_matrix[i, j] += spiky_value  # Spiky kernel value for adjacent points

    return kernel_matrix

def generate_single_Ker_func(K, N_shaped, z_in):
    return




