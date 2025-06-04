import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lfm.model_v2 import create_lfm_model
from lfm.config import get_config
from loguru import logger


def main():
    # Create LFM-7B model
    logger.info("Creating LFM-7B model...")
    model = create_lfm_model("LFM-7B")
    
    # Print model configuration
    config = get_config("LFM-7B")
    logger.info(f"Model Configuration:")
    logger.info(f"  - Hidden dimension: {config.hidden_dim}")
    logger.info(f"  - Number of layers: {config.num_layers}")
    logger.info(f"  - Number of attention heads: {config.num_heads}")
    logger.info(f"  - Number of experts: {config.num_experts}")
    logger.info(f"  - Intermediate dimension: {config.intermediate_dim}")
    logger.info(f"  - Vocabulary size: {config.vocab_size}")
    logger.info(f"  - Max sequence length: {config.max_position_embeddings}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")
    model = model.to(device)
    
    # Test forward pass
    logger.info("\nTesting forward pass...")
    batch_size = 2
    seq_length = 128
    
    # Generate random input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Forward pass
    with torch.no_grad():
        loss, logits = model(input_ids)
    
    logger.info(f"Forward pass successful!")
    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"Expected shape: [{batch_size}, {seq_length}, {config.vocab_size}]")
    
    # Test generation (simple greedy decoding)
    logger.info("\nTesting text generation...")
    model.eval()
    
    # Start with a prompt token
    prompt = torch.tensor([[1]], device=device)  # Token ID 1 as start token
    max_length = 50
    
    generated = prompt
    with torch.no_grad():
        for _ in range(max_length - 1):
            # Get logits for the last position
            _, logits = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Greedy selection
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if we hit an end token (assuming 2 is end token)
            if next_token.item() == 2:
                break
    
    logger.info(f"Generated sequence length: {generated.shape[1]}")
    logger.info(f"Generated token IDs: {generated[0].tolist()}")
    
    # Memory usage
    if torch.cuda.is_available():
        logger.info(f"\nGPU Memory Usage:")
        logger.info(f"  - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"  - Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()