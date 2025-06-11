"""
Liquid Neural Network model implementation.
"""

import numpy as np
from typing import Optional, Tuple
from .utils import initialize_weights, train_lnn, predict_lnn


class LiquidNeuralNetwork:
    """
    Liquid Neural Network implementation with reservoir computing.
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 1000,
        output_dim: int = 10,
        leak_rate: float = 0.1,
        spectral_radius: float = 0.9,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Liquid Neural Network.
        
        Args:
            input_dim: Dimension of input features
            reservoir_dim: Dimension of the reservoir (hidden layer)
            output_dim: Dimension of output
            leak_rate: Leak rate for reservoir dynamics (0 < leak_rate <= 1)
            spectral_radius: Desired spectral radius for reservoir weights
            random_state: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights
        self.reservoir_weights, self.input_weights, self.output_weights = initialize_weights(
            input_dim, reservoir_dim, output_dim, spectral_radius
        )
        
        self.is_trained = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_epochs: int = 10,
        verbose: bool = True
    ) -> 'LiquidNeuralNetwork':
        """
        Train the Liquid Neural Network.
        
        Args:
            X: Training input data of shape (num_samples, input_dim)
            y: Training labels of shape (num_samples, output_dim) or (num_samples,)
            num_epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        # Convert labels to one-hot encoding if needed
        if y.ndim == 1:
            y_onehot = np.zeros((len(y), self.output_dim))
            y_onehot[np.arange(len(y)), y] = 1
            y = y_onehot
        
        # Train the model
        self.output_weights = train_lnn(
            X, y,
            self.reservoir_weights,
            self.input_weights,
            self.output_weights,
            self.leak_rate,
            num_epochs,
            verbose
        )
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data of shape (num_samples, input_dim)
            
        Returns:
            Predictions of shape (num_samples, output_dim)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return predict_lnn(
            X,
            self.reservoir_weights,
            self.input_weights,
            self.output_weights,
            self.leak_rate
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data of shape (num_samples, input_dim)
            
        Returns:
            Class probabilities of shape (num_samples, output_dim)
        """
        predictions = self.predict(X)
        # Apply softmax to get probabilities
        exp_predictions = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        return exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data of shape (num_samples, input_dim)
            
        Returns:
            Class labels of shape (num_samples,)
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test input data of shape (num_samples, input_dim)
            y: True labels of shape (num_samples, output_dim) or (num_samples,)
            
        Returns:
            Accuracy score between 0 and 1
        """
        y_pred = self.predict_classes(X)
        
        # Handle one-hot encoded labels
        if y.ndim == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y
        
        return np.mean(y_pred == y_true)
    
    def get_reservoir_states(self, X: np.ndarray) -> np.ndarray:
        """
        Get reservoir states for given input data.
        
        Args:
            X: Input data of shape (num_samples, input_dim)
            
        Returns:
            Reservoir states of shape (num_samples, reservoir_dim)
        """
        num_samples = X.shape[0]
        reservoir_states = np.zeros((num_samples, self.reservoir_dim))
        
        for i in range(num_samples):
            if i > 0:
                reservoir_states[i, :] = (1 - self.leak_rate) * reservoir_states[i - 1, :]
            
            input_contribution = np.dot(self.input_weights, X[i, :])
            recurrent_contribution = np.dot(self.reservoir_weights, reservoir_states[i, :])
            reservoir_states[i, :] += self.leak_rate * np.tanh(input_contribution + recurrent_contribution)
        
        return reservoir_states