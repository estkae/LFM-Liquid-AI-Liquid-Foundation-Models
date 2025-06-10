#!/usr/bin/env python3
"""Test script to load LFM model correctly"""

from lfm.model import LFModel
from lfm.config import LFMConfig, get_config

# Option 1: Use pre-defined configuration
print("Option 1: Using pre-defined LFM-3B config")
config = get_config("LFM-3B")
model = LFModel(config)
print(f'Model loaded: {model.get_num_params():,} parameters')
print(f'Config: {config.model_name}')
print()

# Option 2: Create custom configuration
print("Option 2: Creating custom config")
custom_config = LFMConfig(
    model_name="LFM-3B-Custom",
    hidden_dim=3072,
    num_layers=32,
    num_heads=24,
    head_dim=128,
    intermediate_dim=8192,
    num_experts=8,
    num_experts_per_token=2
)
custom_model = LFModel(custom_config)
print(f'Custom model loaded: {custom_model.get_num_params():,} parameters')

# Option 3: Modify existing config
print("\nOption 3: Modifying existing config for medical use")
medical_config = get_config("LFM-3B")
medical_config.num_experts = 12  # 12 medical experts
medical_config.model_name = "LFM-3B-Medical"
medical_model = LFModel(medical_config)
print(f'Medical model loaded: {medical_model.get_num_params():,} parameters')