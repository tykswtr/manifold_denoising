import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt


def generate_clean(n_0, d, n_points, sigma, glue_num, base_type, method, grid_type='uniform', sphere_restriction=False):
    """
    Generates clean data points on a manifold with specified properties.

    Parameters:
    - n_0 (int): Dimension of the ambient space.
    - d (int): Dimension of the manifold.
    - n_points (int): Number of data points to generate.
    - sigma (float): Gaussian Kernel parameter.
    - glue_num (int): Number of random points to "glue" together in the manifold.
    - base_type (str): The base shape of the manifold ('circle', 'sphere', 'torus', etc.).
    - method (str): Method to use for generating the manifold ('Ker' for Kernel).
    - kernel_choice (str): Choice of kernel function ('Laplace' or 'Gaussian').
    - grid_type (str, optional): Type of grid to generate ('uniform' or 'random'). Defaults to 'uniform'.
    - sphere_restriction (bool, optional): Restricts the points to a sphere if True. Defaults to False.
    """
    if method == "Ker":
        gen_func = generate_Ker_func
    else:
        raise NotImplementedError("Currently, only 'Ker' method is implemented.")

    # define intrinsic coordinates
    grid_vectors = generate_grid(base_type, d, n_points, grid_type)

    # generate random pairs of points
    glued_pairs = generate_random_point_pairs(base_type, d, glue_num, grid_type)

    return gen_func(n_0, d, n_points, sigma, glued_pairs, grid_vectors, base_type)


# def generate_grid(N_shape):
#     # Generate points for each dimension
#     points = [np.linspace(-1, 1, n) for n in N_shape]
#     # Generate the grid using numpy's meshgrid function
#     # The * operator unpacks the points list into arguments for meshgrid
#     grids = np.meshgrid(*points, indexing='ij')
#     # Reshape the grids to have all coordinates for a point grouped together
#     grid_vectors = np.stack(grids, axis=-1).reshape(-1, len(N_shape))
#     return grid_vectors


def generate_grid(base_type, d, n_points, grid_type='uniform'):
    """
    Generates a grid for a d-dimensional manifold based on the specified base type.

    Parameters:
    - base_type (str): Type of the base manifold ('rectangle', 'circle', 'sphere', 'torus').
    - dims (list of float): Dimensions of the base manifold. For circles, spheres, and tori, this defines their radii.
    - n_points (int): Number of points to generate along each axis for 'rectangle'. Total number of points for others.
    - device (str): The device to run computations on ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: Generated points on the manifold.
    """

    if base_type == 'rectangle':
        grid_points = generate_rectangle(n_points, d, grid_type=grid_type)
    elif base_type == 'ball':
        grid_points = generate_ball(n_points, d, grid_type=grid_type)
    elif base_type == 'sphere':
        grid_points = generate_sphere(n_points, d, grid_type=grid_type)
    elif base_type == 'torus':
        grid_points = generate_torus(n_points, d, grid_type=grid_type)
    else:
        raise ValueError(f"Unsupported base type: {base_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return grid_points.to(device)


def generate_rectangle(n_points, d, grid_type='uniform'):
    # if isinstance(dims, torch.Tensor) and dims.dim() == 0:
    #     np.repeat(1, dims)
    # d = len(dims)
    if grid_type == 'uniform':
        n_points_per_dim = int(torch.ceil(n_points ** (1 / d)))
        # Generate a linear space grid for each dimension
        grids = [torch.linspace(-1, 1, n_points_per_dim) for i in range(d)]
        # Cartesian product to form a grid
        mesh = torch.meshgrid(grids)
        grid_points = torch.stack(mesh, dim=-1).reshape(-1, d)
    elif grid_type == 'random':
        # Generate random points within the specified bounds for each dimension
        grid_points = torch.cat([torch.rand(n_points, 1) * 2 - 1 for i in range(d)], dim=1)
    else:
        raise ValueError("Unsupported grid type: {}".format(grid_type))

    return grid_points


def generate_ball(n_points, d=2, grid_type='uniform'):
    radius = 1.0
    if grid_type == 'uniform':
        volume_ratio = (torch.pi ** (d / 2)) / (math.gamma(d / 2 + 1))  # Volume ratio of ball to cube in d dimensions
        cube_edge_length = 2 * radius  # The edge length of the cube
        cube_volume = cube_edge_length ** d  # Volume of the cube
        ball_volume = volume_ratio * cube_volume  # Volume of the ball
        estimated_total_points = int(n_points / (ball_volume / cube_volume))

        # Generate a uniform grid in the cube
        n_per_dim = int(round(estimated_total_points ** (1 / d)))  # Points per dimension
        grid = torch.linspace(-radius, radius, steps=n_per_dim)
        meshes = torch.meshgrid(*([grid] * d))  # Create a meshgrid for d dimensions
        cube_points = torch.stack(meshes, dim=-1).reshape(-1, d)  # Flatten the grid

        # Filter points to keep only those within the ball's radius
        distances = torch.norm(cube_points, p=2, dim=1)
        ball_points = cube_points[distances <= radius]
    elif grid_type == 'random':
        # Generate random points within the specified bounds for each dimension
        ball_points = torch.rand((n_points, d)) * 2 * radius - radius
        norms = torch.norm(ball_points, p=2, dim=1, keepdim=True)
        ball_points = ball_points / norms * (torch.rand(n_points, 1) ** (1 / d) * radius)
    else:
        raise ValueError("Unsupported grid type: {}".format(grid_type))

    return ball_points


def generate_sphere(n_points, d=2, grid_type='uniform'):
    """Generates points on the surface of a d-dimensional sphere (n-sphere)."""
    radius = 1.0
    if grid_type == 'random':
        # Generate points in d-dimensional space
        points = torch.randn(n_points, d)
        # Normalize points to lie on the surface of the sphere
        norms = torch.norm(points, p=2, dim=1, keepdim=True)
        sphere_points = (points / norms) * radius
    else:
        raise ValueError("Unsupported grid type: {}".format(grid_type))

    return sphere_points


def generate_torus(n_points, d=2, grid_type='uniform'):
    """Generates evenly spaced points on the surface of a d-dimensional torus (n-torus)."""
    if grid_type == 'uniform':
        angle_grids = [torch.linspace(0, 2 * torch.pi, n_points) for i in range(d)]
        mesh = torch.meshgrid(angle_grids)
        torus_points = torch.stack(mesh, dim=-1).reshape(-1, d)
    elif grid_type == 'random':
        torus_points = torch.rand(n_points, d) * 2 * torch.pi
    else:
        raise ValueError("Unsupported grid type: {}".format(grid_type))

    return torus_points


# Generate Kernel function with smoothness parameter K
def generate_Ker_func(n_0, d, n_points, sigma, glued_pairs, grid_vectors, base_type, verbose=True):

    # Define kernel
    thresh = 0.1 * torch.sqrt(d)

    kernel_matrix = compute_kernel_matrix(grid_vectors, glued_pairs, base_type, thresh, sigma=sigma)

    if verbose:
        # optional: display the kernel matrix
        plt.imshow(kernel_matrix.numpy(), interpolation='nearest', cmap='viridis')
        plt.colorbar()  # Show a color bar to indicate the scale
        plt.title("Gaussian Kernel Matrix Visualization")
        plt.show()
        # folder_path = "/path/to/your/folder"
        # full_path = f"{folder_path}/{filename}"
        plt.savefig("kernel_matrix"+".png")

    mvn = MultivariateNormal(torch.zeros(n_points), covariance_matrix=kernel_matrix)
    data = mvn.sample(sample_shape=(n_0,))  # Generate n_0 samples

    return data


# def generate_random_point_pairs(glue_num, d):
#     # Generate random numbers within [-1, 1] with the shape (glue_num, 2, d)
#     # This creates glue_num pairs, each pair consisting of two points in d-dimensional space
#     random_points = np.random.uniform(low=-1, high=1, size=(glue_num, 2, d))
#     return random_points
def generate_random_point_pairs(base_type, d, glue_num):
    random_points = generate_grid(base_type, d, 2*glue_num, grid_type='random')
    # Reshape the points to make pairs
    glued_pairs = random_points.view(glue_num, 2, d)
    return glued_pairs


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-euclidean_distance(x, y) ** 2 / (2 * sigma ** 2))


def l1_distance(point_i, point_j):
    return np.sum(np.abs(point_i - point_j))


def laplace_kernel(point_i, point_j, sigma=1.0):
    return np.exp(-l1_distance(point_i, point_j) / sigma)


def angle_distance(phi_1, phi_2):
    """Computes the shortest distance between two angles, considering the periodicity."""
    d = np.pi - np.abs(np.pi - np.abs(phi_1 - phi_2) % (2 * np.pi))
    return d


# def compute_distances_vectorized(X, Y, metric='euclidean'):
#     """
#     Computes the pairwise distances between two sets of vectors, vectorized.
#
#     Parameters:
#     - X: ndarray of shape (n_samples_X, n_features), the first set of vectors.
#     - Y: ndarray of shape (n_samples_Y, n_features), the second set of vectors.
#     - metric: str, the distance metric to use. Currently, only 'euclidean' is implemented.
#
#     Returns:
#     - distances: ndarray of shape (n_samples_X, n_samples_Y), the distances between each pair of vectors from X and Y.
#     """
#     if metric == 'euclidean':
#         # Efficient vectorized computation of the Euclidean distance
#         # Expand dims to broadcast shapes correctly
#         X_square = np.sum(X**2, axis=1).reshape(-1, 1)
#         Y_square = np.sum(Y**2, axis=1).reshape(1, -1)
#         cross_term = np.dot(X, Y.T)
#         distances = np.sqrt(X_square + Y_square - 2 * cross_term)
#     else:
#         raise ValueError(f"Unsupported metric: {metric}")
#     return distances


def compute_base_distances(X, Y, metric='euclidean'):
    """
    Computes the pairwise distances between two sets of vectors using PyTorch with GPU support.

    Parameters:
    - X: Tensor of shape (n_samples_X, n_features), the first set of vectors.
    - Y: Tensor of shape (n_samples_Y, n_features), the second set of vectors.

    Returns:
    - distances: Tensor of shape (n_samples_X, n_samples_Y), the distances between each pair of vectors from X and Y.
    """
    if metric == 'euclidean':
        distances = torch.cdist(X, Y, p=2)
    elif metric == 'laplacian':
        distances = torch.cdist(X, Y, p=1)
    elif metric == 'sphere':
        cos_angles = torch.mm(X, Y.t())
        cos_angles = torch.clamp(cos_angles, -1, 1)  # Numerical stability
        distances = torch.acos(cos_angles)
    elif metric == 'torus':
        # Ensure angles are within [0, 2*pi]
        X = X % (2 * torch.pi)
        Y = Y % (2 * torch.pi)
        # Compute differences in angles, considering periodic boundary conditions
        diff = torch.abs(X.unsqueeze(1) - Y)  # Shape: (n_samples_X, n_samples_Y, d)
        diff = torch.min(diff, 2 * torch.pi - diff)  # Adjust differences to be within [0, pi]
        distances = torch.norm(diff, p=2, dim=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return distances


def compute_adjusted_distances(grid_vectors, glued_pairs, metric, thresh):
    """
    Computes adjusted distances considering glued pairs as shortcuts.

    Parameters:
    - grid_vectors: Tensor of shape (n_points, n_dims), representing point locations.
    - glued_pairs: List of tuples, each containing indices (into grid_vectors) of glued points.
    - thresh: float, the threshold distance to consider a point "close" to a glued point.

    Returns:
    - adjusted_distances: Tensor, adjusted distances considering glued pairs.
    """
    n_points = grid_vectors.shape[0]
    adjusted_distances = torch.full((n_points, n_points), float('inf'), device=grid_vectors.device)

    for point_a, point_b in glued_pairs:

        # Find points close to point_a and point_b
        distances_to_a = compute_base_distances(grid_vectors, point_a, metric)
        distances_to_b = compute_base_distances(grid_vectors, point_b, metric)
        a_indices = torch.nonzero(distances_to_a <= thresh, as_tuple=True)[0]
        b_indices = torch.nonzero(distances_to_b <= thresh, as_tuple=True)[0]

        # Compute adjusted distances for points close to point_a and point_b
        for i in a_indices:
            for j in b_indices:
                # direct_distance = torch.norm(grid_vectors[i] - point_a + point_b - grid_vectors[j])
                direct_distance = compute_base_distances(grid_vectors[i] - point_a, grid_vectors[j] - point_b, metric)
                adjusted_distances[i, j] = min(adjusted_distances[i, j], direct_distance)
                adjusted_distances[j, i] = adjusted_distances[i, j]  # Ensure symmetry

    return adjusted_distances


def compute_distances(grid_vectors, glued_pairs, base_type, thresh):
    distance = compute_base_distances(grid_vectors, grid_vectors, base_type)
    adjusted_distances = compute_adjusted_distances(grid_vectors, glued_pairs, base_type, thresh)
    return min(distance, adjusted_distances)


def compute_kernel_matrix(grid_vectors, glued_pairs, base_type, thresh, sigma=1.0):
    """
    Computes the kernel matrix using a specified kernel function and a vectorized distance metric.

    Parameters:
    - grid_vectors: ndarray, grid vectors for which to compute the kernel matrix.
    - glued_pairs: ndarray, pairs of points that are "glued" together.
    - thresh: float, threshold value for determining glued points.
    - kernel_choice: str, the kernel function to use ('Laplace', 'Gaussian', etc.).
    - sigma: float, parameter for the kernel function.
    - spike: list or None, spike values to adjust the kernel matrix for glued points.
    - distance_metric: str, the metric to use for distance calculations.

    Returns:
    - ndarray: The computed kernel matrix.
    """
    distance = compute_distances(grid_vectors, glued_pairs, base_type, thresh)
    return torch.exp(-distance ** 2 / (2 * sigma ** 2))

    # distances = compute_distances(grid_vectors, grid_vectors, metric=distance_metric)

    # for i in range(n_points):
    #     for j in range(n_points):
    #         if kernel_choice=="Laplace":
    #             kernel_matrix[i, j] = spike[0] * laplace_kernel(grid_vectors[i], grid_vectors[j], sigma)
    #         else:
    #             kernel_matrix[i, j] = spike[0] * gaussian_kernel(grid_vectors[i], grid_vectors[j], sigma)

    # for i in range(n_points):
    #     for g_i in range(len(glued_pairs)):
    #         if euclidean_distance(grid_vectors[i] - glued_pairs[g_i, 0, :]) <= thresh:
    #             kernel_matrix[i, i] = kernel_matrix[i, i] + spike[1]
    #             for j in range(n_points):
    #                 if euclidean_distance(grid_vectors[j] - glued_pairs[g_i, 1, :]) <= thresh:
    #                     kernel_matrix[i, j] = kernel_matrix[i, j] + spike[1] * gaussian_kernel(grid_vectors[i] - glued_pairs[g_i, 0, :],
    #                                                                       grid_vectors[j] - glued_pairs[g_i, 1, :], sigma)
    #                     kernel_matrix[j, i] = kernel_matrix[j, i] + spike[1] * gaussian_kernel(grid_vectors[i] - glued_pairs[g_i, 0, :],
    #                                                                       grid_vectors[j] - glued_pairs[g_i, 1, :], sigma)
    # for i in range(n_points):
    #     for g_i in range(len(glued_pairs)):
    #         if np.sqrt(np.sum((grid_vectors[i] - glued_pairs[g_i, 1, :]) ** 2)) <= thresh:
    #             kernel_matrixatrix[i, i] = kernel_matrix[i, i] + spike[1]
    #
    #
    # return kernel_matrix


def split_data(data, train_ratio=0.8):
    """Splits data into training and testing sets."""
    n_train = int(len(data) * train_ratio)
    np.random.shuffle(data)
    train_set, test_set = data[:n_train], data[n_train:]
    return train_set, test_set


def add_noise(data, noise_std_dev, sphere_restriction=False):
    """Adds Gaussian noise to the data."""
    noise = noise_std_dev * np.random.randn(*data.shape)
    return data + noise


def save_datasets(train_inputs, train_labels, test_inputs, test_labels, prefix="manifold_dataset"):
    """Saves the datasets as NumPy arrays or PyTorch tensors."""
    # Convert to PyTorch tensors and save (for PyTorch; for NumPy, use np.save)
    datasets = {
        "train_inputs": torch.from_numpy(train_inputs),
        "train_labels": torch.from_numpy(train_labels),
        "test_inputs": torch.from_numpy(test_inputs),
        "test_labels": torch.from_numpy(test_labels)
    }
    for name, tensor in datasets.items():
        torch.save(tensor, f"{prefix}_{name}.pt")
    print(f"Datasets saved with prefix {prefix}.")


def save_manifold_data(data, noise_std_dev, train_ratio=0.8, prefix="manifold_dataset"):
    """
    Splits the manifold data into training and testing sets, adds noise,
    and saves the datasets.

    Parameters:
    - data: ndarray, the clean manifold data.
    - noise_std_dev: float, standard deviation for the noise to be added.
    - train_ratio: float, proportion of data to use for training.
    - prefix: str, prefix for saving the datasets.
    """
    # Split the data into training and testing sets
    train_data, test_data = split_data(data, train_ratio)

    # Add noise to create inputs
    train_inputs = add_noise(train_data, noise_std_dev)
    test_inputs = add_noise(test_data, noise_std_dev)

    # Save the datasets
    save_datasets(train_inputs, train_data, test_inputs, test_data, prefix)

    return


class ManifoldDataset(Dataset):
    def __init__(self, prefix, train=True):
        """
        Custom dataset for loading manifold data using only a prefix.

        Parameters:
        - prefix: str, the prefix used to save the datasets.
        - train: bool, flag to determine whether to load training data or testing data.
        """
        self.prefix = prefix
        self.train = train

        inputs_filename = f"{prefix}_train_inputs.pt" if train else f"{prefix}_test_inputs.pt"
        labels_filename = f"{prefix}_train_labels.pt" if train else f"{prefix}_test_labels.pt"

        self.inputs = torch.load(inputs_filename)
        self.labels = torch.load(labels_filename)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# Example usage
# train_dataset = ManifoldDataset("manifold_dataset")
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# test_dataset = ManifoldDataset("manifold_dataset", train=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
