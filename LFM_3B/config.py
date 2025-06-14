from dataclasses import dataclass
from typing import Optional, Literal
import torch


@dataclass
class LFM3BConfig:
    """Configuration for LFM-3B model combining transformer architecture with liquid neural networks"""
    
    # Model architecture
    model_type: str = "lfm_3b"
    hidden_size: int = 3072
    num_hidden_layers: int = 20
    num_attention_heads: int = 24
    num_key_value_heads: Optional[int] = None
    head_dim: int = 128
    intermediate_size: int = 8192
    
    # Liquid Neural Network parameters
    use_liquid_layers: bool = True
    liquid_num_reservoirs: int = 3
    liquid_reservoir_size: int = 512
    liquid_leak_rate: float = 0.3
    liquid_spectral_radius: float = 0.99
    liquid_input_scaling: float = 0.1
    liquid_sparse_connectivity: float = 0.1
    liquid_update_interval: int = 4  # Update liquid state every N layers
    
    # Mixture of Experts
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    
    # Model dimensions
    vocab_size: int = 128256
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # Attention config
    attention_bias: bool = False
    attention_dropout: float = 0.0
    
    # Regularization
    hidden_dropout: float = 0.0
    
    # Normalization
    rms_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5
    use_rms_norm: bool = True
    
    # Initialization
    initializer_range: float = 0.02
    
    # Other
    use_cache: bool = True
    tie_word_embeddings: bool = False
    torch_dtype: Optional[str] = "float32"
    
    # Medical mode (inherited from UA-LNN)
    medical_mode: bool = False
    medical_safety_threshold: float = 0.8
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
            
        # Validate head dimensions
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        # Update head_dim if needed
        self.head_dim = self.hidden_size // self.num_attention_heads
    
    @property
    def total_params(self) -> int:
        """Calculate total parameters"""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size
        
        # Attention parameters (Q, K, V, O projections)
        attn_params = 4 * self.hidden_size * self.hidden_size * self.num_hidden_layers
        
        # MoE FFN parameters
        expert_params = 2 * self.hidden_size * self.intermediate_size * self.num_experts * self.num_hidden_layers
        
        # Router parameters
        router_params = self.hidden_size * self.num_experts * self.num_hidden_layers
        
        # Liquid layer parameters (if used)
        liquid_params = 0
        if self.use_liquid_layers:
            # Input projection to reservoirs
            liquid_params += self.hidden_size * self.liquid_reservoir_size * self.liquid_num_reservoirs
            # Output projection from reservoirs
            liquid_params += self.liquid_reservoir_size * self.liquid_num_reservoirs * self.hidden_size
            # Multiply by number of liquid layers
            liquid_params *= (self.num_hidden_layers // self.liquid_update_interval)
        
        # Layer norm parameters
        ln_params = 2 * self.hidden_size * self.num_hidden_layers
        
        # Output head
        output_params = self.hidden_size * self.vocab_size if not self.tie_word_embeddings else 0
        
        total = embed_params + attn_params + expert_params + router_params + liquid_params + ln_params + output_params
        return total
    
    @property
    def total_params_in_billions(self) -> float:
        """Return total parameters in billions"""
        return self.total_params / 1e9