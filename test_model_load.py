#!/usr/bin/env python3
"""Test script to load LFM model correctly"""

from lfm.model_v2 import LFMModel
from lfm.config import LFMConfig, get_config

# Option 1: Use pre-defined configuration
print("Option 1: Using pre-defined LFM-3B config")
config = get_config("LFM-3B")
model = LFMModel(config)
param_count = sum(p.numel() for p in model.parameters())
print(f'Model loaded: {param_count:,} parameters')
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
custom_model = LFMModel(custom_config)
custom_param_count = sum(p.numel() for p in custom_model.parameters())
print(f'Custom model loaded: {custom_param_count:,} parameters')

# Option 3: Modify existing config
print("\nOption 3: Modifying existing config for medical use")
medical_config = get_config("LFM-3B")
medical_config.num_experts = 12  # 12 medical experts
medical_config.model_name = "LFM-3B-Medical"
medical_model = LFMModel(medical_config)
medical_param_count = sum(p.numel() for p in medical_model.parameters())
print(f'Medical model loaded: {medical_param_count:,} parameters')

# Show the difference in parameters
print(f"\nParameter difference with medical experts: {medical_param_count - param_count:,}")