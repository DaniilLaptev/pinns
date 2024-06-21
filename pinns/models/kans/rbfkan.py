
# COPIED FROM https://github.com/Sid2690/RBF-KAN/tree/main

import torch
import torch.nn as nn
import torch.nn.init as init

class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        init.xavier_uniform_(self.weights)

    def gaussian_rbf(self, distances):
        return torch.exp(-self.alpha * distances ** 2)
    
    def cdist(self, a, b):
        """
        Compute the pairwise distances between two sets of points.
        
        Parameters:
        a (np.ndarray): A 2D array of shape (m, d) representing m points in d-dimensional space.
        b (np.ndarray): A 2D array of shape (n, d) representing n points in d-dimensional space.
        
        Returns:
        np.ndarray: A 2D array of shape (m, n) where the element at (i, j) is the distance between a[i] and b[j].
        """
        # Ensure the inputs are numpy arrays
        
        # Compute the squared differences and sum them along the last axis
        dists = torch.sqrt(torch.sum((a.unsqueeze(1)[:, :, :] - b.unsqueeze(0)[:, :, :]) ** 2, axis=-1))
        
        return dists

    def forward(self, x):
        distances = self.cdist(x, self.centers)
        basis_values = self.gaussian_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output
    
class RBFKAN(nn.Module):
    def __init__(self, layers, **parameters):
        super(RBFKAN, self).__init__()
        self.layers = nn.ModuleList([
            RBFKANLayer(
                in_dim, out_dim, **parameters
            ) for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x