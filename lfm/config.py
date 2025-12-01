from dataclasses import dataclass
from typing import Optional


@dataclass
class LFMConfig:
    """Configuration class for Liquid Foundation Models"""
    
    model_name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    num_experts: int = 8
    num_experts_per_token: int = 2
    vocab_size: int = 128256
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    use_bias: bool = False
    tie_word_embeddings: bool = False
    
    # RPT (Reinforcement Preference Training) parameters
    rpt_enabled: bool = False
    rpt_learning_rate: float = 1e-6
    rpt_temperature: float = 0.8
    rpt_num_samples: int = 8
    rpt_entropy_threshold: float = 2.0
    rpt_dynamic_sampling_start: int = 500
    rpt_kl_penalty: float = 0.0
    
    @property
    def total_params(self) -> int:
        """Calculate total parameters in billions"""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_dim
        
        # Transformer layer parameters
        # Self-attention: Q, K, V projections + output projection
        attn_params = 4 * self.hidden_dim * self.hidden_dim * self.num_layers
        
        # MoE FFN parameters
        # Each expert has: hidden_dim -> intermediate_dim -> hidden_dim
        expert_params = 2 * self.hidden_dim * self.intermediate_dim * self.num_experts * self.num_layers
        
        # Router parameters
        router_params = self.hidden_dim * self.num_experts * self.num_layers
        
        # Layer norm parameters
        ln_params = 2 * self.hidden_dim * self.num_layers
        
        # Output head
        output_params = self.hidden_dim * self.vocab_size if not self.tie_word_embeddings else 0
        
        total = embed_params + attn_params + expert_params + router_params + ln_params + output_params
        return total


# Pre-defined configurations
LFM_CONFIGS = {
    "LFM-1B": LFMConfig(
        model_name="LFM-1B",
        hidden_dim=2048,
        num_layers=16,
        num_heads=16,
        head_dim=128,
        intermediate_dim=5632,
        num_experts=8,
        num_experts_per_token=2,
    ),
    "LFM-3B": LFMConfig(
        model_name="LFM-3B",
        hidden_dim=3072,
        num_layers=20,
        num_heads=24,
        head_dim=128,
        intermediate_dim=8192,
        num_experts=8,
        num_experts_per_token=2,
    ),
    "LFM-7B": LFMConfig(
        model_name="LFM-7B",
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        intermediate_dim=11008,
        num_experts=8,
        num_experts_per_token=2,
    ),
    "LFM-40B": LFMConfig(
        model_name="LFM-40B",
        hidden_dim=6656,
        num_layers=48,
        num_heads=52,
        head_dim=128,
        intermediate_dim=17920,
        num_experts=16,
        num_experts_per_token=2,
    ),
}


def get_config(model_name: str) -> LFMConfig:
    """Get configuration for a specific model size"""
    if model_name not in LFM_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LFM_CONFIGS.keys())}")
    return LFM_CONFIGS[model_name]