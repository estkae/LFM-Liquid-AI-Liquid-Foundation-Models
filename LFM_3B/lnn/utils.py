"""
Utility functions for Liquid Neural Networks.
"""

import numpy as np
from typing import Tuple


def initialize_weights(
    input_dim: int, 
    reservoir_dim: int, 
    output_dim: int, 
    spectral_radius: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize weights for the Liquid Neural Network.
    
    Args:
        input_dim: Dimension of input features
        reservoir_dim: Dimension of the reservoir (hidden layer)
        output_dim: Dimension of output
        spectral_radius: Desired spectral radius for reservoir weights
        
    Returns:
        Tuple of (reservoir_weights, input_weights, output_weights)
    """
    # Initialize reservoir weights randomly
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    
    # Scale reservoir weights to achieve desired spectral radius
    eigenvalues = np.linalg.eigvals(reservoir_weights)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    reservoir_weights *= spectral_radius / max_eigenvalue
    
    # Initialize input-to-reservoir weights randomly
    input_weights = np.random.randn(reservoir_dim, input_dim)
    
    # Initialize output weights to zero
    output_weights = np.zeros((output_dim, reservoir_dim))
    
    return reservoir_weights, input_weights, output_weights


def train_lnn(
    input_data: np.ndarray,
    labels: np.ndarray,
    reservoir_weights: np.ndarray,
    input_weights: np.ndarray,
    output_weights: np.ndarray,
    leak_rate: float = 0.1,
    num_epochs: int = 10,
    verbose: bool = True
) -> np.ndarray:
    """
    Train the Liquid Neural Network.
    
    Args:
        input_data: Training input data of shape (num_samples, input_dim)
        labels: Training labels of shape (num_samples, output_dim)
        reservoir_weights: Reservoir weight matrix
        input_weights: Input-to-reservoir weight matrix
        output_weights: Output weight matrix (will be updated)
        leak_rate: Leak rate for reservoir dynamics
        num_epochs: Number of training epochs
        verbose: Whether to print training progress
        
    Returns:
        Updated output weights
    """
    num_samples = input_data.shape[0]
    reservoir_dim = reservoir_weights.shape[0]
    reservoir_states = np.zeros((num_samples, reservoir_dim))
    
    for epoch in range(num_epochs):
        # Forward pass through reservoir
        for i in range(num_samples):
            # Update reservoir state with leak rate
            if i > 0:
                reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
            
            # Add new input contribution
            input_contribution = np.dot(input_weights, input_data[i, :])
            recurrent_contribution = np.dot(reservoir_weights, reservoir_states[i, :])
            reservoir_states[i, :] += leak_rate * np.tanh(input_contribution + recurrent_contribution)
        
        # Train output weights using ridge regression
        # Add small regularization for numerical stability
        regularization = 1e-6
        reservoir_states_T = reservoir_states.T
        output_weights = np.dot(
            labels.T,
            np.linalg.inv(np.dot(reservoir_states_T, reservoir_states) + regularization * np.eye(reservoir_dim))
        ).dot(reservoir_states_T)
        
        if verbose:
            # Compute training accuracy
            train_predictions = np.dot(reservoir_states, output_weights.T)
            train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(labels, axis=1))
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}")
    
    return output_weights


def predict_lnn(
    input_data: np.ndarray,
    reservoir_weights: np.ndarray,
    input_weights: np.ndarray,
    output_weights: np.ndarray,
    leak_rate: float = 0.1
) -> np.ndarray:
    """
    Make predictions using the trained Liquid Neural Network.
    
    Args:
        input_data: Test input data of shape (num_samples, input_dim)
        reservoir_weights: Reservoir weight matrix
        input_weights: Input-to-reservoir weight matrix
        output_weights: Trained output weight matrix
        leak_rate: Leak rate for reservoir dynamics
        
    Returns:
        Predictions of shape (num_samples, output_dim)
    """
    num_samples = input_data.shape[0]
    reservoir_dim = reservoir_weights.shape[0]
    reservoir_states = np.zeros((num_samples, reservoir_dim))
    
    # Forward pass through reservoir
    for i in range(num_samples):
        # Update reservoir state with leak rate
        if i > 0:
            reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
        
        # Add new input contribution
        input_contribution = np.dot(input_weights, input_data[i, :])
        recurrent_contribution = np.dot(reservoir_weights, reservoir_states[i, :])
        reservoir_states[i, :] += leak_rate * np.tanh(input_contribution + recurrent_contribution)
    
    # Compute predictions using output weights
    predictions = np.dot(reservoir_states, output_weights.T)
    
    return predictions