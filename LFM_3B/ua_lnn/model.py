"""
Update Adaptive Liquid Neural Network (UA-LNN) Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from .config import UALNNConfig
from .components import AdaptiveNeuron, UncertaintyModule, LiquidReservoir


class UpdateAdaptiveLNN(nn.Module):
    """
    Main UA-LNN model with update-based adaptation and uncertainty awareness
    """
    
    def __init__(self, config: UALNNConfig):
        super().__init__()
        self.config = config
        config.validate()
        
        # Multi-layer liquid reservoirs
        self.reservoirs = nn.ModuleList()
        dims = [config.input_dim] + [config.hidden_dim] * config.num_layers
        
        for i in range(config.num_layers):
            self.reservoirs.append(
                LiquidReservoir(dims[i], dims[i+1], config)
            )
        
        # Uncertainty estimation
        self.uncertainty_module = UncertaintyModule(config.hidden_dim, config)
        
        # Output layers
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Task-specific heads (for medical applications)
        if config.medical_mode:
            self.medical_heads = nn.ModuleDict({
                'diagnostic': nn.Linear(config.hidden_dim, config.output_dim),
                'treatment': nn.Linear(config.hidden_dim, config.output_dim),
                'risk_assessment': nn.Linear(config.hidden_dim, 1)
            })
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Evidence accumulator for medical mode
        if config.medical_mode and config.require_evidence:
            self.evidence_extractor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
            )
    
    def forward(self, 
                x: torch.Tensor,
                task: Optional[str] = None,
                return_uncertainty: bool = True,
                return_evidence: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through UA-LNN
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            task: Specific task for medical mode
            return_uncertainty: Whether to compute uncertainty
            return_evidence: Whether to extract evidence (medical mode)
            
        Returns:
            Dictionary containing:
                - output: Model predictions
                - uncertainty: Uncertainty scores (if requested)
                - evidence: Evidence vectors (if requested)
                - confidence: Confidence scores
                - reservoir_states: Hidden states from reservoirs
        """
        # Handle both sequential and non-sequential inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        batch_size, seq_len, _ = x.size()
        
        # Process through liquid reservoirs
        current_input = x
        reservoir_states = []
        reservoir_info = []
        
        for i, reservoir in enumerate(self.reservoirs):
            states, info = reservoir(current_input, reset_state=(i == 0))
            reservoir_states.append(states)
            reservoir_info.append(info)
            current_input = states
        
        # Get final hidden representation
        hidden = current_input[:, -1, :]  # [batch_size, hidden_dim]
        hidden = self.dropout(hidden)
        
        # Compute uncertainty if requested
        uncertainty = None
        uncertainty_info = {}
        if return_uncertainty:
            uncertainty, uncertainty_info = self.uncertainty_module(hidden)
        
        # Generate output
        if self.config.medical_mode and task in self.medical_heads:
            output = self.medical_heads[task](hidden)
        else:
            output = self.output_projection(hidden)
        
        # Compute confidence
        if output.size(-1) > 1:  # Classification
            confidence = F.softmax(output, dim=-1).max(dim=-1)[0]
        else:  # Regression
            confidence = 1.0 - (uncertainty.squeeze() if uncertainty is not None 
                              else torch.zeros_like(output).squeeze())
        
        # Extract evidence if requested (medical mode)
        evidence = None
        if (self.config.medical_mode and 
            self.config.require_evidence and 
            return_evidence):
            evidence = self.evidence_extractor(hidden)
        
        # Safety check for medical mode
        if self.config.medical_mode:
            safety_mask = confidence >= self.config.safety_threshold
            if not safety_mask.all():
                # Mark low-confidence predictions
                output = output.masked_fill(~safety_mask.unsqueeze(-1), float('-inf'))
        
        # Prepare results
        results = {
            'output': output,
            'confidence': confidence,
            'reservoir_states': reservoir_states,
            'hidden_features': hidden
        }
        
        if uncertainty is not None:
            results['uncertainty'] = uncertainty
            results['uncertainty_info'] = uncertainty_info
        
        if evidence is not None:
            results['evidence'] = evidence
        
        # Add diagnostic information
        results['diagnostics'] = {
            'reservoir_info': reservoir_info,
            'adaptation_active': any(
                info.get('adaptive_norm', 0) > 0 
                for info in reservoir_info
            )
        }
        
        return results
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor,
                                n_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty quantification
        
        Args:
            x: Input tensor
            n_samples: Number of forward passes for uncertainty
            
        Returns:
            Dictionary with predictions and uncertainty measures
        """
        self.train()  # Enable dropout for MC sampling
        
        outputs = []
        confidences = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                result = self.forward(x, return_uncertainty=True)
                outputs.append(result['output'])
                confidences.append(result['confidence'])
        
        outputs = torch.stack(outputs, dim=0)
        confidences = torch.stack(confidences, dim=0)
        
        # Compute statistics
        mean_output = outputs.mean(dim=0)
        std_output = outputs.std(dim=0)
        mean_confidence = confidences.mean(dim=0)
        
        # For classification, compute entropy
        if outputs.size(-1) > 1:
            mean_probs = F.softmax(mean_output, dim=-1)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        else:
            entropy = std_output.squeeze()
        
        self.eval()
        
        return {
            'prediction': mean_output,
            'std': std_output,
            'confidence': mean_confidence,
            'entropy': entropy,
            'epistemic_uncertainty': std_output.mean(dim=-1),
            'samples': outputs
        }
    
    def adapt_online(self, x: torch.Tensor, y: torch.Tensor, 
                    learning_rate: Optional[float] = None):
        """
        Perform online adaptation based on new data
        
        Args:
            x: Input data
            y: Target labels
            learning_rate: Learning rate for adaptation
        """
        if learning_rate is None:
            learning_rate = self.config.adaptation_rate
        
        # Forward pass
        result = self.forward(x)
        output = result['output']
        
        # Compute loss
        if output.size(-1) > 1:  # Classification
            loss = F.cross_entropy(output, y)
        else:  # Regression
            loss = F.mse_loss(output.squeeze(), y)
        
        # Backward pass for adaptation
        loss.backward()
        
        # Update only adaptive components
        with torch.no_grad():
            for reservoir in self.reservoirs:
                for neuron in reservoir.adaptive_neurons:
                    if neuron.weight.grad is not None:
                        neuron.weight.data -= learning_rate * neuron.weight.grad
                        neuron.weight.grad.zero_()
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance scores
        
        Args:
            x: Input tensor
            
        Returns:
            Feature importance scores [input_dim]
        """
        x.requires_grad_(True)
        
        result = self.forward(x)
        output = result['output']
        
        # Compute gradients
        if output.size(-1) > 1:
            target = output.max(dim=-1)[1]
            loss = F.cross_entropy(output, target)
        else:
            loss = output.mean()
        
        grads = torch.autograd.grad(loss, x, create_graph=False)[0]
        
        # Aggregate over batch and sequence
        importance = grads.abs().mean(dim=(0, 1))
        
        return importance / importance.sum()