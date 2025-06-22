#!/usr/bin/env python3
"""Test script to verify MedicalHealthConfig loading works properly"""

import json
import sys
import os
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.medical_health_base import MedicalHealthConfig

def test_config_loading():
    """Test that we can create and load MedicalHealthConfig properly"""
    
    # Create a config
    config = MedicalHealthConfig(
        model_name="test-medical",
        hidden_dim=1024,
        num_layers=12,
        num_heads=16,
        head_dim=64,
        intermediate_dim=2048,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=2048,
        vocab_size=50257
    )
    
    print("Created config successfully")
    print(f"Config has {len(config.__dict__)} attributes")
    
    # Get valid fields
    valid_fields = {f.name for f in fields(MedicalHealthConfig)}
    print(f"\nValid MedicalHealthConfig fields ({len(valid_fields)}):")
    for field in sorted(valid_fields):
        print(f"  - {field}")
    
    # Show extra attributes added in __post_init__
    extra_attrs = set(config.__dict__.keys()) - valid_fields
    print(f"\nExtra attributes added in __post_init__ ({len(extra_attrs)}):")
    for attr in sorted(extra_attrs):
        print(f"  - {attr}: {getattr(config, attr)}")
    
    # Simulate saving
    config_dict = config.__dict__
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    
    print(f"\nFiltered config has {len(filtered_config)} fields")
    
    # Test loading
    try:
        loaded_config = MedicalHealthConfig(**filtered_config)
        print("\n✅ Successfully loaded config from filtered dict!")
        print(f"Loaded config has {len(loaded_config.__dict__)} attributes")
    except Exception as e:
        print(f"\n❌ Failed to load config: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)