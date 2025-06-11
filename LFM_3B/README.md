# LFM_3B - Liquid Neural Network Implementation

This directory contains an implementation of Liquid Neural Networks (LNN) based on reservoir computing principles, designed for the LFM_3B model.

## Directory Structure

```
LFM_3B/
├── lnn/
│   ├── __init__.py      # Package initialization
│   ├── model.py         # LiquidNeuralNetwork class implementation
│   └── utils.py         # Utility functions for weight initialization, training, and prediction
└── train_mnist_lnn.py   # Example training script using MNIST dataset
```

## Features

- **Liquid Neural Network (LNN)** implementation with reservoir computing
- Efficient training using ridge regression for output weights
- Support for classification tasks
- Hyperparameter tuning capabilities
- MNIST dataset example with multiple experiments

## Installation

```bash
pip install numpy tensorflow
```

## Quick Start

```python
from lnn import LiquidNeuralNetwork
import numpy as np

# Create model
model = LiquidNeuralNetwork(
    input_dim=784,
    reservoir_dim=1000,
    output_dim=10,
    leak_rate=0.1,
    spectral_radius=0.9
)

# Train model
model.fit(X_train, y_train, num_epochs=10)

# Make predictions
predictions = model.predict_classes(X_test)
accuracy = model.score(X_test, y_test)
```

## Key Components

### LiquidNeuralNetwork Class

The main model class with methods:
- `fit()`: Train the model
- `predict()`: Get raw predictions
- `predict_proba()`: Get probability predictions
- `predict_classes()`: Get class labels
- `score()`: Calculate accuracy

### Hyperparameters

- `input_dim`: Input feature dimension
- `reservoir_dim`: Number of reservoir neurons (hidden layer)
- `output_dim`: Number of output classes
- `leak_rate`: Controls the "memory" of the reservoir (0 < leak_rate ≤ 1)
- `spectral_radius`: Controls the dynamics of the reservoir
- `num_epochs`: Number of training iterations

## Example Usage

Run the MNIST training example:

```bash
cd LFM_3B
python train_mnist_lnn.py
```

This will:
1. Train a baseline model on MNIST
2. Run experiments with different hyperparameter configurations
3. Compare performance across different settings

## Performance Notes

The implementation includes several optimizations:
- Ridge regression for stable output weight training
- Efficient matrix operations using NumPy
- Regularization for numerical stability

## Improving Performance

To improve accuracy beyond the baseline ~35%:

1. **Increase reservoir size**: Try 2000-5000 neurons
2. **Tune leak rate**: Experiment with values between 0.05-0.5
3. **Adjust spectral radius**: Try values between 0.8-1.5
4. **Add input scaling**: Normalize or standardize input features
5. **Use ensemble methods**: Combine multiple LNN models
6. **Add regularization**: Adjust the regularization parameter in training