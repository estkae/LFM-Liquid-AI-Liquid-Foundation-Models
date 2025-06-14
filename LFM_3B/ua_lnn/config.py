"""
Configuration for Update Adaptive Liquid Neural Network (UA-LNN)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class UALNNConfig:
    """Configuration for UA-LNN model"""
    
    # Basic architecture parameters
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    num_layers: int = 3
    
    # Liquid dynamics parameters
    leak_rate: float = 0.9
    spectral_radius: float = 0.95
    sparsity: float = 0.1
    
    # Update adaptation parameters
    adaptation_rate: float = 0.1
    update_threshold: float = 0.3
    memory_window: int = 100
    
    # Uncertainty parameters
    uncertainty_threshold: float = 0.85
    confidence_scaling: bool = True
    epistemic_weight: float = 0.5
    aleatoric_weight: float = 0.5
    
    # Medical application parameters
    medical_mode: bool = False
    medical_specialties: List[str] = field(default_factory=lambda: [
        "diagnostic", "treatment", "risk_assessment"
    ])
    safety_threshold: float = 0.95
    require_evidence: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.1
    
    # Regularization
    l2_reg: float = 0.0001
    gradient_clip: float = 1.0
    
    # Adaptive neuron parameters
    neuron_types: List[str] = field(default_factory=lambda: [
        "standard", "adaptive", "gated"
    ])
    adaptation_method: str = "gradient"  # gradient, hebbian, or hybrid
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'UALNNConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.leak_rate <= 1, "Leak rate must be in (0, 1]"
        assert 0 < self.spectral_radius < 1, "Spectral radius must be in (0, 1)"
        assert 0 <= self.sparsity <= 1, "Sparsity must be in [0, 1]"
        assert 0 < self.adaptation_rate < 1, "Adaptation rate must be in (0, 1)"
        assert 0 <= self.uncertainty_threshold <= 1, "Uncertainty threshold must be in [0, 1]"
        
        if self.medical_mode:
            assert self.safety_threshold >= 0.9, "Medical safety threshold must be >= 0.9"
            assert self.require_evidence, "Medical mode requires evidence tracking"