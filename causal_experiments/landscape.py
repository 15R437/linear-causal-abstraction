import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import numpy as np
import torch

def visualize_loss_landscape(loss_fn, A_low, A_high_gt, T_gt, S_gt, X_test, n_points=20, range_val=1.0):
    """
    Visualizes the loss landscape of a given loss function around the ground truth parameters.
    
    Parameters:
        loss_fn (function): The loss function to visualize. Should take A_low, A_high, T, S, X as input.
        A_low (np.ndarray): The low-level adjacency matrix.
        A_high_gt (np.ndarray): The ground truth high-level adjacency matrix.
        T_gt (np.ndarray): The ground truth abstraction matrix.
        S_gt (np.ndarray): The ground truth structural equations.
        X_test (np.ndarray): Test data for evaluating the loss function.
        n_points (int): Number of points to sample in each direction for visualization.
        range_val (float): Range of values to perturb the parameters for visualization.
    """
    # Create a grid of points around the ground truth parameters
    A_high_values = np.linspace(A_high_gt - range_val, A_high_gt + range_val, n_points)
    T_values = np.linspace(T_gt - range_val, T_gt + range_val, n_points)
    
    loss_values = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            A_high_perturbed = A_high_values[i]
            T_perturbed = T_values[j]
            loss_values[i, j] = loss_fn(A_low=A_low, A_high=A_high_perturbed, T=T_perturbed, S=S_gt, X=X_test)
    
    # Visualize the loss landscape
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.contourf(T_values.flatten(), A_high_values.flatten(), loss_values.T, levels=50)
    plt.colorbar(label='Loss Value')
    plt.xlabel('T Perturbation')
    plt.ylabel('A_high Perturbation')
    plt.title('Loss Landscape Visualization')
    plt.show()