"""
Example training script for UA-LNN on MNIST dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from ua_lnn.config import UALNNConfig
from ua_lnn.model import UpdateAdaptiveLNN
from ua_lnn.utils import UALNNTrainer, OnlineAdaptationTrainer


def load_mnist_data(batch_size: int = 32):
    """Load MNIST dataset"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def create_sequential_mnist_data(batch_size: int = 32):
    """Create sequential MNIST data for temporal modeling"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    
    class SequentialMNIST(torch.utils.data.Dataset):
        def __init__(self, dataset, seq_len=7):
            self.dataset = dataset
            self.seq_len = seq_len
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            # Convert image to sequence (7x7 patches of 4x4 each)
            seq = img.unfold(1, 4, 4).unfold(2, 4, 4)
            seq = seq.reshape(self.seq_len, self.seq_len, -1)
            seq = seq.reshape(self.seq_len * self.seq_len, -1)
            return seq, label
    
    train_seq = SequentialMNIST(
        datasets.MNIST('./data', train=True, download=True, transform=transform)
    )
    test_seq = SequentialMNIST(
        datasets.MNIST('./data', train=False, transform=transform)
    )
    
    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train UA-LNN on MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--sequential', action='store_true', help='Use sequential MNIST')
    parser.add_argument('--adapt-online', action='store_true', help='Test online adaptation')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--save-path', type=str, default='ua_lnn_mnist.pt', 
                       help='Path to save model')
    
    args = parser.parse_args()
    
    # Configuration
    config = UALNNConfig(
        input_dim=16 if args.sequential else 784,  # 4x4 patches or full image
        hidden_dim=args.hidden_dim,
        output_dim=10,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        leak_rate=0.95,
        spectral_radius=0.9,
        sparsity=0.2,
        adaptation_rate=0.05,
        update_threshold=0.2,
        uncertainty_threshold=0.8,
        adaptation_method="hybrid"
    )
    
    # Load data
    if args.sequential:
        print("Loading sequential MNIST data...")
        train_loader, test_loader = create_sequential_mnist_data(args.batch_size)
    else:
        print("Loading standard MNIST data...")
        train_loader, test_loader = load_mnist_data(args.batch_size)
    
    # Create model
    model = UpdateAdaptiveLNN(config)
    print(f"Created UA-LNN model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = UALNNTrainer(model, config, use_wandb=args.wandb)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader, 
        test_loader, 
        epochs=args.epochs,
        save_path=args.save_path
    )
    
    # Final evaluation
    print("\nFinal evaluation:")
    final_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {final_metrics['loss']:.4f}")
    print(f"Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Average Uncertainty: {final_metrics['avg_uncertainty']:.4f}")
    print(f"Average Confidence: {final_metrics['avg_confidence']:.4f}")
    
    # Test online adaptation if requested
    if args.adapt_online:
        print("\nTesting online adaptation...")
        adaptation_trainer = OnlineAdaptationTrainer(model, config)
        
        # Test on a few batches
        test_iter = iter(test_loader)
        for i in range(5):
            x, y = next(test_iter)
            x, y = x.to(trainer.device), y.to(trainer.device)
            
            adapt_info = adaptation_trainer.adapt_step(x, y)
            print(f"Batch {i+1}: Loss before: {adapt_info['loss_before']:.4f}, "
                  f"after: {adapt_info['loss_after']:.4f}, "
                  f"improvement: {adapt_info['improvement']:.4f}")
    
    # Demonstrate uncertainty estimation
    print("\nDemonstrating uncertainty estimation...")
    model.eval()
    x_sample, y_sample = next(iter(test_loader))
    x_sample = x_sample[:5].to(trainer.device)  # Take 5 samples
    
    with torch.no_grad():
        results = model.predict_with_uncertainty(x_sample, n_samples=50)
        predictions = results['prediction'].argmax(dim=-1)
        uncertainty = results['epistemic_uncertainty']
        confidence = results['confidence']
        
        print("\nSample predictions with uncertainty:")
        for i in range(5):
            print(f"Sample {i+1}: Prediction={predictions[i].item()}, "
                  f"Confidence={confidence[i].item():.3f}, "
                  f"Uncertainty={uncertainty[i].item():.3f}")


if __name__ == '__main__':
    main()