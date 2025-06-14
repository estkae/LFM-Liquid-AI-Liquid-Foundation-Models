"""
Example usage of LFM-3B model
"""

import torch
from config import LFM3BConfig
from model import LFM3BForCausalLM


def main():
    # Initialize configuration
    config = LFM3BConfig()
    
    print(f"Model Configuration:")
    print(f"- Hidden size: {config.hidden_size}")
    print(f"- Number of layers: {config.num_hidden_layers}")
    print(f"- Number of attention heads: {config.num_attention_heads}")
    print(f"- Number of experts: {config.num_experts}")
    print(f"- Liquid reservoirs: {config.liquid_num_reservoirs}")
    print(f"- Total parameters: {config.total_params_in_billions:.2f}B")
    print()
    
    # Initialize model
    model = LFM3BForCausalLM(config)
    model.eval()
    
    # Example input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_router_logits=True,
        )
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Number of hidden states: {len(outputs.hidden_states)}")
    print(f"Number of liquid states: {len(outputs.liquid_states)}")
    
    # Count liquid layers
    liquid_layers = sum(1 for state in outputs.liquid_states if state is not None)
    print(f"Active liquid layers: {liquid_layers}")
    
    # Generation example
    print("\nGeneration example:")
    prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Example token IDs
    
    generated = model.generate(
        input_ids=prompt,
        max_length=20,
        temperature=0.8,
        do_sample=True,
    )
    
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    # Medical mode example
    print("\nMedical mode configuration:")
    medical_config = LFM3BConfig(medical_mode=True, medical_safety_threshold=0.85)
    print(f"- Medical mode: {medical_config.medical_mode}")
    print(f"- Safety threshold: {medical_config.medical_safety_threshold}")


if __name__ == "__main__":
    main()