"""
Fine-tune LFM model on custom dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
from loguru import logger
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
import wandb
from lfm.config import LFMConfig
from lfm.model_v2 import LFM


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning"""
    model_path: str = "./saved_models/lfm_7b"
    data_path: str = "./data/train.jsonl"
    output_dir: str = "./finetuned_model"
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data parameters
    max_length: int = 512
    train_split: float = 0.9
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Logging
    log_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    use_wandb: bool = False


class TextDataset(Dataset):
    """Simple text dataset for fine-tuning"""
    
    def __init__(self, data_path: str, max_length: int = 512, vocab_size: int = 32000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.data = []
        
        # Load data from JSONL file
        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.data.append(item)
            logger.info(f"Loaded {len(self.data)} examples from {data_path}")
        else:
            # Create dummy data for demonstration
            logger.warning(f"Data file not found at {data_path}. Creating dummy data...")
            self.data = [
                {"text": "This is a sample text for fine-tuning."},
                {"text": "The model will learn from this data."},
                {"text": "Fine-tuning helps adapt the model to specific tasks."},
            ] * 100
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # In practice, you would use a proper tokenizer here
        # For now, we'll create random tokens as placeholder
        text = self.data[idx].get("text", "")
        
        # Simulate tokenization (replace with real tokenizer)
        input_ids = torch.randint(0, self.vocab_size, (self.max_length,))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        
        # Labels are same as input_ids for language modeling
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


class Trainer:
    """Trainer for fine-tuning LFM model"""
    
    def __init__(self, model: LFM, config: FinetuneConfig, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Gradient checkpointing
        if config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project="lfm-finetune", config=config.__dict__)
            wandb.watch(self.model, log="all")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                loss, _ = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"]
                )
        else:
            loss, _ = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"]
            )
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', enabled=self.config.use_mixed_precision):
                    loss, _ = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"]
                    )
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = np.exp(avg_loss)
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity
        }
    
    def save_checkpoint(self, step: int, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        torch.save({
            "step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "metrics": metrics
        }, checkpoint_dir / "training_state.pth")
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        global_step = 0
        best_eval_loss = float('inf')
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        logger.info(f"Total training steps: {total_steps}")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            optimizer_step = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.config.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    scheduler.step()
                    self.optimizer.zero_grad()
                    optimizer_step += 1
                
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Logging
                if global_step % self.config.log_steps == 0:
                    metrics = {
                        "train_loss": loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step
                    }
                    
                    if self.config.use_wandb:
                        wandb.log(metrics)
                    
                    logger.info(f"Step {global_step}: loss={loss:.4f}, lr={metrics['learning_rate']:.2e}")
                
                # Evaluation
                if eval_dataloader and global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    logger.info(f"Evaluation at step {global_step}: {eval_metrics}")
                    
                    if self.config.use_wandb:
                        wandb.log(eval_metrics)
                    
                    # Save best model
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        self.save_checkpoint(global_step, eval_metrics)
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        self.model.save_pretrained(self.config.output_dir)
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LFM model")
    parser.add_argument("--model_path", type=str, default="./saved_models/lfm_7b", help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, default="./data/train.jsonl", help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Create config
    config = FinetuneConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if Path(config.model_path).exists():
        logger.info(f"Loading pretrained model from {config.model_path}")
        model = LFM.from_pretrained(config.model_path, device=device)
    else:
        logger.warning("Pretrained model not found. Creating new model...")
        model_config = LFMConfig(
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
        model = LFM(model_config).to(device)
    
    # Get model config for dataset
    model_config = model.config
    
    # Create dataset
    dataset = TextDataset(config.data_path, config.max_length, model_config.vocab_size)
    
    # Split into train/eval
    train_size = int(len(dataset) * config.train_split)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Start training
    trainer.train(train_dataloader, eval_dataloader)
    
    # Clean up
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()