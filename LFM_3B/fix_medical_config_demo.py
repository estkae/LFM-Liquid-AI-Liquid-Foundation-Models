#!/usr/bin/env python3
"""Demonstrate the fix for MedicalHealthConfig loading issue"""

import json
import tempfile
import os
import sys
from pathlib import Path
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.medical_health_base import MedicalHealthConfig, create_medical_health_model

def demonstrate_issue_and_fix():
    """Show the problem and the solution"""
    
    print("=== MedicalHealthConfig Loading Fix Demo ===\n")
    
    # Create a model with config
    print("1. Creating Medical Health Model...")
    model = create_medical_health_model(config_name="tiny")
    config = model.config
    
    # Show the issue
    print(f"\n2. Config has {len(config.__dict__)} total attributes")
    valid_fields = {f.name for f in fields(MedicalHealthConfig)}
    print(f"   But MedicalHealthConfig only accepts {len(valid_fields)} fields in __init__")
    
    # Show extra fields
    extra_fields = set(config.__dict__.keys()) - valid_fields
    print(f"\n3. Extra fields added in __post_init__: {extra_fields}")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        save_path.mkdir()
        
        # Simulate the problematic save (what create_medical_health_model.py does)
        print("\n4. Simulating problematic save...")
        config_dict_full = config.__dict__.copy()
        with open(save_path / "medical_config_bad.json", 'w') as f:
            json.dump(config_dict_full, f, indent=2)
        print(f"   Saved {len(config_dict_full)} fields to medical_config_bad.json")
        
        # Try to load it (this would fail)
        print("\n5. Trying to load with all fields (THIS WOULD FAIL):")
        try:
            bad_config = MedicalHealthConfig(**config_dict_full)
            print("   ❌ This shouldn't work!")
        except TypeError as e:
            print(f"   ✅ Expected error: {e}")
        
        # Show the fix
        print("\n6. Applying the fix (filtering to valid fields)...")
        config_dict_filtered = {k: v for k, v in config_dict_full.items() if k in valid_fields}
        with open(save_path / "medical_config_good.json", 'w') as f:
            json.dump(config_dict_filtered, f, indent=2)
        print(f"   Saved {len(config_dict_filtered)} fields to medical_config_good.json")
        
        # Load with fix
        print("\n7. Loading with filtered fields:")
        try:
            good_config = MedicalHealthConfig(**config_dict_filtered)
            print("   ✅ Success! Config loaded properly")
            print(f"   Loaded config has {len(good_config.__dict__)} attributes (after __post_init__)")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
        
        # Show what train_german_medical.py now does
        print("\n8. The fix in train_german_medical.py:")
        print("   - When loading: filters config_dict to only valid dataclass fields")
        print("   - When saving: uses save_medical_model() which only saves valid fields")
        print("   - This ensures compatibility when loading saved models")

if __name__ == "__main__":
    demonstrate_issue_and_fix()