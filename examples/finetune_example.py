"""
Simple example of fine-tuning LFM model
"""
import sys
sys.path.append('..')

import torch
from pathlib import Path
from loguru import logger

# Import from parent directory
from finetune_lfm import FinetuneConfig, TextDataset, Trainer
from lfm.config import LFMConfig
from lfm.model_v2 import LFM


def main():
    # 1. Create or load a model
    logger.info("Setting up model...")
    
    # Option A: Create new small model for testing
    config = LFMConfig(
        vocab_size=10000,      # Smaller vocab for testing
        hidden_size=512,       # Smaller model
        num_hidden_layers=6,   # Fewer layers
        num_attention_heads=8,
        intermediate_size=2048,
        num_experts=4,         # Fewer experts
        topk_experts=2,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LFM(config).to(device)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Created model with {param_count:.2f}M parameters")
    
    # Option B: Load existing model (uncomment to use)
    # model = LFM.from_pretrained("./saved_models/lfm_7b", device=device)
    
    # 2. Prepare data
    logger.info("Preparing training data...")
    
    # First, create sample data if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    if not (data_dir / "train.jsonl").exists():
        # Create sample data
        import subprocess
        subprocess.run([
            sys.executable, 
            "../examples/prepare_data.py", 
            "--mode", "sample",
            "--output", "./data/train.jsonl",
            "--num_samples", "500"
        ])
    
    # 3. Configure fine-tuning
    finetune_config = FinetuneConfig(
        model_path="",  # Not used since we created model directly
        data_path="./data/train.jsonl",
        output_dir="./finetuned_model_example",
        
        # Small values for quick testing
        learning_rate=5e-5,
        batch_size=2,
        gradient_accumulation_steps=2,
        num_epochs=2,
        max_length=256,
        
        # Logging
        log_steps=5,
        eval_steps=20,
        save_steps=50,
        
        # Options
        use_mixed_precision=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Disabled for small model
        use_wandb=False
    )
    
    # 4. Create dataset and trainer
    dataset = TextDataset(
        finetune_config.data_path, 
        finetune_config.max_length,
        config.vocab_size
    )
    
    # Simple train/eval split
    train_size = int(len(dataset) * 0.9)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    from torch.utils.data import DataLoader
    from finetune_lfm import collate_fn
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=finetune_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=finetune_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 5. Create trainer and start fine-tuning
    trainer = Trainer(model, finetune_config, device)
    
    logger.info("Starting fine-tuning...")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    logger.info(f"Batch size: {finetune_config.batch_size}")
    logger.info(f"Learning rate: {finetune_config.learning_rate}")
    
    # Train!
    trainer.train(train_dataloader, eval_dataloader)
    
    # 6. Test the fine-tuned model
    logger.info("\nTesting fine-tuned model...")
    model.eval()
    
    # Generate some text
    with torch.no_grad():
        test_input = torch.randint(0, config.vocab_size, (1, 10), device=device)
        loss, logits = model(test_input)
        logger.info(f"Test output shape: {logits.shape}")
        logger.info(f"Test loss: {loss.item():.4f}")
    
    logger.info("\nFine-tuning completed! Model saved to: " + finetune_config.output_dir)


if __name__ == "__main__":
    main()