#!/usr/bin/env python3
"""
Test script for Municipal MoE Model
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig, create_municipal_moe_model


def test_moe_creation():
    """Test MoE model creation"""
    print("🧪 Testing MoE model creation...")
    
    config = MunicipalMoEConfig(
        vocab_size=1000,  # Small for testing
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_experts=4,
        num_experts_per_tok=2
    )
    
    model = MunicipalMoEModel(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model(input_ids=input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    print("✅ Model creation and forward pass successful!")
    return model


def test_expert_routing():
    """Test that different experts are activated"""
    print("\n🧪 Testing expert routing...")
    
    config = MunicipalMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,  # Must divide hidden_size evenly
        num_experts=8,
        num_experts_per_tok=2
    )
    
    model = MunicipalMoEModel(config)
    
    # Create input that should activate different experts
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Hook to capture expert routing
    expert_selections = []
    
    def hook_fn(module, input, output):
        if hasattr(module, 'router'):
            _, indices = module.router(input[0])
            expert_selections.append(indices)
    
    # Register hook on MoE layer
    for layer in model.layers:
        if hasattr(layer, 'router'):
            layer.register_forward_hook(hook_fn)
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    # Check that different experts were selected
    if expert_selections:
        all_experts = torch.cat([sel.flatten() for sel in expert_selections])
        unique_experts = torch.unique(all_experts)
        print(f"✅ {len(unique_experts)} different experts activated out of {config.num_experts}")
        print(f"   Expert IDs: {unique_experts.tolist()}")
    
    return model


def test_save_load():
    """Test model save and load"""
    print("\n🧪 Testing model save/load...")
    
    # Create and save model
    model, config = create_municipal_moe_model("./test_municipal_moe")
    
    # Create test input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Get output from original model
    model.eval()
    with torch.no_grad():
        original_output = model(input_ids=input_ids)["logits"]
    
    # Load model
    loaded_model = MunicipalMoEModel.from_pretrained("./test_municipal_moe")
    loaded_model.eval()
    
    # Get output from loaded model
    with torch.no_grad():
        loaded_output = loaded_model(input_ids=input_ids)["logits"]
    
    # Compare outputs
    assert torch.allclose(original_output, loaded_output, atol=1e-5)
    print("✅ Model save/load successful!")
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_municipal_moe")
    
    return loaded_model


def test_training_step():
    """Test a single training step"""
    print("\n🧪 Testing training step...")
    
    config = MunicipalMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2
    )
    
    model = MunicipalMoEModel(config)
    model.train()
    
    # Create dummy batch
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    
    # Check loss
    assert loss is not None
    assert loss.item() > 0
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grads = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break
    
    assert has_grads, "No gradients computed"
    
    print(f"✅ Training step successful! Loss: {loss.item():.4f}")
    
    return model


def main():
    print("🚀 Starting Municipal MoE Model Tests\n")
    
    # Run all tests
    test_moe_creation()
    test_expert_routing()
    test_save_load()
    test_training_step()
    
    print("\n✅ All tests passed!")
    print("\n📝 Next steps:")
    print("1. Create base model: python train_german_municipal.py --create-base-model")
    print("2. Create sample data: python train_german_municipal.py --create-sample-data municipal_data.jsonl")
    print("3. Train model: python train_german_municipal.py --model-path ./municipal_moe_base --data-file municipal_data.jsonl")


if __name__ == "__main__":
    main()