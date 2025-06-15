#!/usr/bin/env python3
"""
Script to check if a model has MoE (Mixture of Experts)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.utils import load_model
from LFM_3B.model import LFM3BForCausalLM


def check_moe_in_model(model_path: str):
    """Check if model has MoE layers"""
    
    print(f"🔍 Checking model: {model_path}")
    print("=" * 50)
    
    # Load model
    model = load_model(LFM3BForCausalLM, model_path)
    
    # Check for MoE layers
    moe_layers = []
    expert_count = 0
    
    for name, module in model.named_modules():
        # Check for MoE indicators
        if "moe" in name.lower() or "mixture" in name.lower():
            moe_layers.append(name)
            print(f"✅ Found MoE layer: {name}")
            
        if "expert" in name.lower():
            expert_count += 1
            
        # Check module type
        module_type = type(module).__name__
        if "MixtureOfExperts" in module_type or "MoE" in module_type:
            print(f"✅ MoE Module: {name} ({module_type})")
    
    print(f"\n📊 Summary:")
    print(f"   MoE Layers found: {len(moe_layers)}")
    print(f"   Expert modules: {expert_count}")
    
    # Check config
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'num_experts'):
            print(f"   Config num_experts: {config.num_experts}")
        if hasattr(config, 'num_experts_per_token'):
            print(f"   Experts per token: {config.num_experts_per_token}")
    
    # Detailed layer inspection
    print(f"\n🔎 Detailed Layer Structure:")
    for i, (name, module) in enumerate(model.named_modules()):
        if i > 20:  # Limit output
            print("   ... (truncated)")
            break
        indent = "   " * name.count(".")
        print(f"{indent}{name}: {type(module).__name__}")
    
    return len(moe_layers) > 0


def compare_models(original_path: str, trained_path: str):
    """Compare original and trained models"""
    
    print("\n🆚 Model Comparison")
    print("=" * 50)
    
    print("\n1️⃣ Original Model:")
    has_moe_original = check_moe_in_model(original_path)
    
    print("\n2️⃣ Trained Model:")
    has_moe_trained = check_moe_in_model(trained_path)
    
    print("\n📝 Comparison Result:")
    print(f"   Original has MoE: {'✅ Yes' if has_moe_original else '❌ No'}")
    print(f"   Trained has MoE: {'✅ Yes' if has_moe_trained else '❌ No'}")
    
    if has_moe_original and has_moe_trained:
        print("\n✅ Both models have MoE! Training preserved the architecture.")
    elif not has_moe_original and not has_moe_trained:
        print("\n❌ Neither model has MoE. You need the Medical Health MoE model!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check if model has MoE")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model")
    parser.add_argument("--compare-with", type=str,
                        help="Path to second model for comparison")
    
    args = parser.parse_args()
    
    if args.compare_with:
        compare_models(args.model_path, args.compare_with)
    else:
        has_moe = check_moe_in_model(args.model_path)
        
        print(f"\n🎯 Result: Model {'HAS' if has_moe else 'DOES NOT HAVE'} MoE")
        
        if not has_moe:
            print("\n💡 To get a Medical MoE model, use:")
            print("   cd ../lfm")
            print("   python3 create_medical_health_model.py --size small --save-path ./medical_moe")


if __name__ == "__main__":
    main()