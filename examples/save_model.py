"""
Save LFM model to disk for local usage
"""
import torch
import os
from pathlib import Path
from lfm.config import LFMConfig
from lfm.model_v2 import LFM
from loguru import logger


def save_model(model, save_dir, model_name="lfm_model"):
    """
    Save model weights and configuration
    
    Args:
        model: LFM model instance
        save_dir: Directory to save the model
        model_name: Name for the saved model files
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = save_path / f"{model_name}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save configuration separately for easy loading
    config_path = save_path / f"{model_name}_config.pth"
    torch.save(model.config.__dict__, config_path)
    logger.info(f"Config saved to {config_path}")
    
    # Save in HuggingFace format (optional)
    hf_path = save_path / f"{model_name}_hf"
    hf_path.mkdir(exist_ok=True)
    
    # Save weights in HF format
    torch.save(model.state_dict(), hf_path / "pytorch_model.bin")
    
    # Save config in JSON format
    import json
    with open(hf_path / "config.json", 'w') as f:
        json.dump(model.config.__dict__, f, indent=2)
    logger.info(f"Model saved in HuggingFace format to {hf_path}")
    
    return save_path


def main():
    # Initialize configuration
    config = LFMConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        num_experts=8,
        topk_experts=2,
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LFM(config).to(device)
    
    # Optional: Load trained weights if available
    # checkpoint_path = "path/to/checkpoint.pth"
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Save model
    save_dir = "./saved_models"
    save_path = save_model(model, save_dir, "lfm_7b")
    
    # Verify save
    saved_files = list(save_path.glob("*"))
    logger.info(f"Saved files: {[f.name for f in saved_files]}")
    
    # Test loading
    checkpoint = torch.load(save_path / "lfm_7b.pth", map_location=device)
    logger.info(f"Successfully loaded checkpoint with keys: {checkpoint.keys()}")


if __name__ == "__main__":
    main()