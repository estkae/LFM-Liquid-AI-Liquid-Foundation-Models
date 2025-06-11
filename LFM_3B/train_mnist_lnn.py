"""
Example script for training Liquid Neural Network on MNIST dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from lnn import LiquidNeuralNetwork
import time


def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    
    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    x_train = x_train.reshape((60000, 784)) / 255.0
    x_test = x_test.reshape((10000, 784)) / 255.0
    
    # Convert labels to one-hot encoding
    y_train_onehot = keras.utils.to_categorical(y_train, 10)
    y_test_onehot = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot)


def train_lnn_model(hyperparams=None):
    """Train LNN model with given hyperparameters."""
    
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'input_dim': 784,
            'reservoir_dim': 1000,
            'output_dim': 10,
            'leak_rate': 0.1,
            'spectral_radius': 0.9,
            'num_epochs': 10,
            'random_state': 42
        }
    
    # Load data
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_and_preprocess_mnist()
    
    print("\nInitializing Liquid Neural Network...")
    print(f"Hyperparameters: {hyperparams}")
    
    # Create and train model
    model = LiquidNeuralNetwork(
        input_dim=hyperparams['input_dim'],
        reservoir_dim=hyperparams['reservoir_dim'],
        output_dim=hyperparams['output_dim'],
        leak_rate=hyperparams['leak_rate'],
        spectral_radius=hyperparams['spectral_radius'],
        random_state=hyperparams['random_state']
    )
    
    print("\nTraining model...")
    start_time = time.time()
    
    model.fit(x_train, y_train_onehot, num_epochs=hyperparams['num_epochs'], verbose=True)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = model.score(x_test, y_test_onehot)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make some predictions
    print("\nMaking predictions on first 10 test samples...")
    predictions = model.predict_classes(x_test[:10])
    print(f"Predictions: {predictions}")
    print(f"True labels: {y_test[:10]}")
    
    return model, test_accuracy


def experiment_with_hyperparameters():
    """Run experiments with different hyperparameter configurations."""
    
    print("="*60)
    print("Running experiments with different hyperparameters")
    print("="*60)
    
    # Experiment 1: Baseline
    print("\nExperiment 1: Baseline configuration")
    baseline_params = {
        'input_dim': 784,
        'reservoir_dim': 1000,
        'output_dim': 10,
        'leak_rate': 0.1,
        'spectral_radius': 0.9,
        'num_epochs': 10,
        'random_state': 42
    }
    model1, acc1 = train_lnn_model(baseline_params)
    
    # Experiment 2: Larger reservoir
    print("\n" + "="*60)
    print("Experiment 2: Larger reservoir (2000 neurons)")
    larger_reservoir_params = baseline_params.copy()
    larger_reservoir_params['reservoir_dim'] = 2000
    model2, acc2 = train_lnn_model(larger_reservoir_params)
    
    # Experiment 3: Different leak rate
    print("\n" + "="*60)
    print("Experiment 3: Higher leak rate (0.3)")
    higher_leak_params = baseline_params.copy()
    higher_leak_params['leak_rate'] = 0.3
    model3, acc3 = train_lnn_model(higher_leak_params)
    
    # Experiment 4: Different spectral radius
    print("\n" + "="*60)
    print("Experiment 4: Higher spectral radius (1.2)")
    higher_spectral_params = baseline_params.copy()
    higher_spectral_params['spectral_radius'] = 1.2
    model4, acc4 = train_lnn_model(higher_spectral_params)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Baseline (1000 neurons, leak=0.1, spectral=0.9): {acc1:.4f}")
    print(f"Larger reservoir (2000 neurons): {acc2:.4f}")
    print(f"Higher leak rate (0.3): {acc3:.4f}")
    print(f"Higher spectral radius (1.2): {acc4:.4f}")
    
    return {
        'baseline': (model1, acc1),
        'larger_reservoir': (model2, acc2),
        'higher_leak': (model3, acc3),
        'higher_spectral': (model4, acc4)
    }


if __name__ == "__main__":
    # Run single training
    print("Training single LNN model on MNIST...")
    model, accuracy = train_lnn_model()
    
    # Run hyperparameter experiments
    print("\n" + "="*60)
    print("Do you want to run hyperparameter experiments? (This will take longer)")
    # For automated script, we'll just run the experiments
    experiment_results = experiment_with_hyperparameters()