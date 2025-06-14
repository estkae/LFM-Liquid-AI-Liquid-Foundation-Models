"""
Core components for Update Adaptive Liquid Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class AdaptiveNeuron(nn.Module):
    """
    Adaptive neuron with update-based adaptation mechanism
    """
    
    def __init__(self, input_dim: int, config: 'UALNNConfig'):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        
        # Neuron parameters
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Adaptive parameters
        self.adaptation_rate = nn.Parameter(torch.tensor(config.adaptation_rate))
        self.threshold = nn.Parameter(torch.tensor(config.update_threshold))
        
        # Memory for adaptation
        self.register_buffer('memory', torch.zeros(config.memory_window, input_dim))
        self.register_buffer('memory_ptr', torch.tensor(0))
        
        # Liquid dynamics
        self.leak_rate = config.leak_rate
        self.register_buffer('state', torch.zeros(1))
        
    def forward(self, x: torch.Tensor, adapt: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive updates
        
        Args:
            x: Input tensor [batch_size, input_dim]
            adapt: Whether to perform adaptation
            
        Returns:
            output: Neuron output [batch_size, 1]
            info: Dictionary with adaptation info
        """
        batch_size = x.size(0)
        
        # Compute weighted input
        weighted_input = torch.matmul(x, self.weight) + self.bias
        
        # Update state with leak
        self.state = self.leak_rate * self.state + (1 - self.leak_rate) * weighted_input.mean()
        
        # Compute activation
        activation = torch.tanh(weighted_input + self.state)
        
        # Adaptation mechanism
        adaptation_delta = torch.zeros_like(self.weight)
        if adapt and self.training:
            # Store in memory
            ptr = int(self.memory_ptr.item())
            self.memory[ptr] = x.mean(dim=0)
            self.memory_ptr = (self.memory_ptr + 1) % self.config.memory_window
            
            # Compute adaptation based on memory
            if ptr > 10:  # Need some history
                memory_mean = self.memory[:ptr].mean(dim=0)
                memory_std = self.memory[:ptr].std(dim=0) + 1e-6
                
                # Compute update signal
                update_signal = (x.mean(dim=0) - memory_mean) / memory_std
                
                # Apply threshold
                significant_change = torch.abs(update_signal) > self.threshold
                
                # Adapt weights
                if self.config.adaptation_method == "gradient":
                    adaptation_delta = self.adaptation_rate * update_signal * significant_change.float()
                elif self.config.adaptation_method == "hebbian":
                    adaptation_delta = self.adaptation_rate * torch.outer(
                        activation.mean(dim=0), x.mean(dim=0)
                    ).squeeze() * significant_change.float()
                else:  # hybrid
                    grad_term = self.adaptation_rate * update_signal
                    hebb_term = self.adaptation_rate * torch.outer(
                        activation.mean(dim=0), x.mean(dim=0)
                    ).squeeze()
                    adaptation_delta = (0.5 * grad_term + 0.5 * hebb_term) * significant_change.float()
                
                # Update weights
                self.weight.data += adaptation_delta
        
        info = {
            'state': self.state.item(),
            'adaptation_norm': adaptation_delta.norm().item(),
            'activation_mean': activation.mean().item()
        }
        
        return activation, info


class UncertaintyModule(nn.Module):
    """
    Module for estimating epistemic and aleatoric uncertainty
    """
    
    def __init__(self, hidden_dim: int, config: 'UALNNConfig'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Uncertainty estimation networks
        self.epistemic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.aleatoric_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, features: torch.Tensor, 
                dropout_samples: int = 10) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate uncertainty from features
        
        Args:
            features: Hidden features [batch_size, hidden_dim]
            dropout_samples: Number of MC dropout samples
            
        Returns:
            total_uncertainty: Combined uncertainty score [batch_size, 1]
            uncertainty_info: Detailed uncertainty information
        """
        batch_size = features.size(0)
        
        # Epistemic uncertainty via MC dropout
        epistemic_samples = []
        self.train()  # Enable dropout
        for _ in range(dropout_samples):
            epistemic_samples.append(self.epistemic_net(features))
        
        epistemic_samples = torch.stack(epistemic_samples, dim=0)
        epistemic_mean = epistemic_samples.mean(dim=0)
        epistemic_std = epistemic_samples.std(dim=0)
        
        # Aleatoric uncertainty
        self.eval()  # Disable dropout for aleatoric
        aleatoric = self.aleatoric_net(features)
        
        # Combined uncertainty
        epistemic_uncertainty = epistemic_std / (epistemic_mean + 1e-6)
        total_uncertainty = (
            self.config.epistemic_weight * epistemic_uncertainty +
            self.config.aleatoric_weight * aleatoric
        )
        
        # Normalize to [0, 1]
        total_uncertainty = torch.sigmoid(total_uncertainty)
        
        uncertainty_info = {
            'epistemic_mean': epistemic_mean.mean().item(),
            'epistemic_std': epistemic_std.mean().item(),
            'aleatoric': aleatoric.mean().item(),
            'total': total_uncertainty.mean().item()
        }
        
        return total_uncertainty, uncertainty_info


class LiquidReservoir(nn.Module):
    """
    Liquid reservoir with adaptive dynamics
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, config: 'UALNNConfig'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Input weights
        self.W_in = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        
        # Reservoir weights with sparsity
        W_res = torch.randn(hidden_dim, hidden_dim)
        
        # Apply sparsity
        mask = torch.rand(hidden_dim, hidden_dim) < config.sparsity
        W_res = W_res * mask.float()
        
        # Normalize spectral radius
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        W_res = W_res * (config.spectral_radius / eigenvalues.max())
        
        self.W_res = nn.Parameter(W_res)
        
        # Adaptive neurons
        self.adaptive_neurons = nn.ModuleList([
            AdaptiveNeuron(hidden_dim, config) 
            for _ in range(hidden_dim // 4)  # 25% adaptive neurons
        ])
        
        # State
        self.register_buffer('state', torch.zeros(1, hidden_dim))
        
    def forward(self, x: torch.Tensor, 
                reset_state: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through liquid reservoir
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            reset_state: Whether to reset reservoir state
            
        Returns:
            states: Reservoir states [batch_size, seq_len, hidden_dim]
            info: Reservoir information
        """
        batch_size, seq_len, _ = x.size()
        
        if reset_state or self.state.size(0) != batch_size:
            self.state = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        states = []
        adaptive_info = []
        
        for t in range(seq_len):
            # Input contribution
            input_contrib = torch.matmul(x[:, t], self.W_in)
            
            # Reservoir dynamics
            res_contrib = torch.matmul(self.state, self.W_res.T)
            
            # Update state with leak
            self.state = (self.config.leak_rate * self.state + 
                         (1 - self.config.leak_rate) * torch.tanh(input_contrib + res_contrib))
            
            # Apply adaptive neurons to subset of state
            if len(self.adaptive_neurons) > 0:
                adaptive_indices = torch.linspace(
                    0, self.hidden_dim-1, len(self.adaptive_neurons)
                ).long()
                
                for i, neuron in enumerate(self.adaptive_neurons):
                    idx = adaptive_indices[i]
                    adapted_output, info = neuron(self.state)
                    self.state[:, idx] = adapted_output.squeeze()
                    adaptive_info.append(info)
            
            states.append(self.state.clone())
        
        states = torch.stack(states, dim=1)
        
        reservoir_info = {
            'final_state_norm': self.state.norm(dim=1).mean().item(),
            'states_mean': states.mean().item(),
            'states_std': states.std().item()
        }
        
        if adaptive_info:
            reservoir_info['adaptive_norm'] = np.mean([
                info['adaptation_norm'] for info in adaptive_info
            ])
        
        return states, reservoir_info