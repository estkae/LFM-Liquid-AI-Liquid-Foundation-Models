"""
Load and use saved LFM model for text generation
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from lfm.config import LFMConfig
from lfm.model_v2 import LFM
from loguru import logger
import time


def load_model(model_path, device='cuda'):
    """
    Load saved model from disk
    
    Args:
        model_path: Path to saved model file (.pth)
        device: Device to load model on
    
    Returns:
        model: Loaded LFM model
        config: Model configuration
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate config
    config_dict = checkpoint['config']
    config = LFMConfig(**config_dict)
    
    # Create model
    model = LFM(config).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    return model, config


def generate_text(model, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    """
    Generate text using the model
    
    Args:
        model: LFM model
        input_ids: Input token IDs [batch_size, seq_length]
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p (nucleus) sampling
    
    Returns:
        generated_ids: Generated token IDs
    """
    model.eval()
    device = next(model.parameters()).device
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions
            _, logits = model(generated)
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop if EOS token is generated (assuming EOS token ID is 2)
            if (next_token == 2).any():
                break
    
    return generated


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = "./saved_models/lfm_7b.pth"
    
    # Check if saved model exists
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}. Please run save_model.py first.")
        return
    
    model, config = load_model(model_path, device)
    
    # Example: Generate text
    logger.info("Generating text...")
    
    # Create sample input (you would normally use a tokenizer here)
    batch_size = 1
    prompt_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_length), device=device)
    
    # Generate
    start_time = time.time()
    generated_ids = generate_text(
        model, 
        input_ids, 
        max_length=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    
    generation_time = time.time() - start_time
    num_tokens = generated_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = num_tokens / generation_time
    
    logger.info(f"Generated {num_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    logger.info(f"Generated shape: {generated_ids.shape}")
    
    # In practice, you would decode the generated IDs back to text using a tokenizer
    # Example: text = tokenizer.decode(generated_ids[0])
    
    # Batch inference example
    logger.info("\nBatch inference example...")
    batch_size = 4
    input_ids_batch = torch.randint(0, config.vocab_size, (batch_size, prompt_length), device=device)
    
    with torch.no_grad():
        loss, logits = model(input_ids_batch)
        logger.info(f"Batch inference - Loss: {loss.item():.4f}, Logits shape: {logits.shape}")
    
    # Memory usage
    if torch.cuda.is_available():
        logger.info(f"\nGPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()