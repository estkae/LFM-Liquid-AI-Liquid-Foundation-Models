"""
Simple example of saving and loading LFM models
"""
import torch
from lfm.config import LFMConfig
from lfm.model_v2 import LFM


def main():
    # 1. Create a model
    config = LFMConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )
    
    model = LFM(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # 2. Save the model
    save_path = "./my_lfm_model"
    model.save_pretrained(save_path)
    
    # 3. Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = LFM.from_pretrained(save_path, device=device)
    print(f"Model loaded from {save_path}")
    
    # 4. Use the model
    batch_size = 1
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    with torch.no_grad():
        loss, logits = loaded_model(input_ids)
        print(f"Output shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()