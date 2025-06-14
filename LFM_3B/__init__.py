"""
LFM-3B: Liquid Foundation Model with 3 Billion Parameters

This model combines transformer architecture with liquid neural networks
for enhanced dynamic processing and adaptation capabilities.
"""

from .config import LFM3BConfig
from .model import (
    LFM3BModel,
    LFM3BForCausalLM,
    LFM3BOutput,
    LFM3BAttention,
    LFM3BDecoderLayer,
    MixtureOfExperts,
    LiquidLayer,
)
from .utils import (
    save_model,
    load_model,
    count_parameters,
    get_model_size_mb,
    print_model_summary,
    create_attention_mask,
    get_linear_schedule_with_warmup,
)

__version__ = "0.1.0"

__all__ = [
    "LFM3BConfig",
    "LFM3BModel", 
    "LFM3BForCausalLM",
    "LFM3BOutput",
    "LFM3BAttention",
    "LFM3BDecoderLayer",
    "MixtureOfExperts",
    "LiquidLayer",
    "save_model",
    "load_model",
    "count_parameters",
    "get_model_size_mb",
    "print_model_summary",
    "create_attention_mask",
    "get_linear_schedule_with_warmup",
]