import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
import argparse
from tqdm import tqdm
import wandb
from typing import Optional, Dict

from lfm.model_v2 import create_lfm_model
from lfm.config import get_config
from lfm.rpt_training import RPTConfig, create_rpt_trainer


class MathReasoningDataset(Dataset):
    """Dataset for mathematical reasoning tasks (similar to OmniMATH)"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_length: int = 8000,
        split: str = 'train'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data (placeholder - replace with actual data loading)
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str):
        """Load mathematical reasoning problems"""
        # This is a placeholder - implement actual data loading
        # Expected format: list of dicts with 'question' and 'answer' keys
        logger.info(f"Loading {self.split} data from {data_path}")
        
        # For demonstration, create dummy data
        dummy_data = []
        for i in range(1000 if self.split == 'train' else 100):
            dummy_data.append({
                'question': f"What is {i} + {i+1}?",
                'answer': f"The answer is {2*i + 1}."
            })
        
        return dummy_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize question and answer
        question = item['question']
        answer = item['answer']
        
        # Create prompt
        prompt = f"Question: {question}\nAnswer: "
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        # Combine tokens
        input_ids = torch.cat([
            prompt_tokens['input_ids'],
            answer_tokens['input_ids']
        ], dim=1).squeeze(0)
        
        # Create labels (only for answer part)
        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_tokens['input_ids'][0]):] = input_ids[len(prompt_tokens['input_ids'][0]):]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }


def train_epoch_rpt(
    rpt_trainer,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
    wandb_log: bool = False
):
    """Train for one epoch using RPT"""
    model = rpt_trainer.model
    model.train()
    
    epoch_metrics = {
        'total_loss': 0,
        'lm_loss': 0,
        'pg_loss': 0,
        'avg_reward': 0,
        'max_reward': 0,
        'num_batches': 0
    }
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # RPT training step
        metrics = rpt_trainer.train_step(batch, optimizer)
        
        # Update learning rate
        scheduler.step()
        
        # Accumulate metrics
        for key in ['total_loss', 'lm_loss', 'pg_loss', 'avg_reward', 'max_reward']:
            if key in metrics:
                epoch_metrics[key] += metrics.get(key.replace('total_', ''), 0)
        epoch_metrics['num_batches'] += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{metrics.get('loss', 0):.4f}",
            'reward': f"{metrics.get('avg_reward', 0):.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log to wandb
        if wandb_log and batch_idx % 10 == 0:
            wandb.log({
                'train/loss': metrics.get('loss', 0),
                'train/lm_loss': metrics.get('lm_loss', 0),
                'train/pg_loss': metrics.get('pg_loss', 0),
                'train/avg_reward': metrics.get('avg_reward', 0),
                'train/max_reward': metrics.get('max_reward', 0),
                'train/lr': scheduler.get_last_lr()[0],
                'train/step': rpt_trainer.global_step
            })
    
    # Average epoch metrics
    avg_metrics = {
        k: v / epoch_metrics['num_batches'] 
        for k, v in epoch_metrics.items() 
        if k != 'num_batches'
    }
    
    return avg_metrics


def evaluate_rpt(
    rpt_trainer,
    dataloader,
    num_eval_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model with RPT metrics"""
    return rpt_trainer.evaluate(dataloader, num_eval_samples)


def main():
    parser = argparse.ArgumentParser(description="Train LFM with RPT")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='LFM-3B',
                        choices=['LFM-1B', 'LFM-3B', 'LFM-7B', 'LFM-40B'],
                        help='Model size to train')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to pre-trained base model (e.g., Deepseek-R1-Distill)')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (e.g., OmniMATH dataset)')
    
    # RPT hyperparameters
    parser.add_argument('--rpt_lr', type=float, default=1e-6,
                        help='Learning rate for RPT')
    parser.add_argument('--rpt_temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--rpt_batch_size', type=int, default=256,
                        help='Batch size for RPT')
    parser.add_argument('--rpt_num_samples', type=int, default=8,
                        help='Number of response samples per question')
    parser.add_argument('--rpt_max_seq_length', type=int, default=8000,
                        help='Maximum sequence length')
    parser.add_argument('--rpt_training_steps', type=int, default=1000,
                        help='Total training steps')
    parser.add_argument('--rpt_dynamic_sampling_start', type=int, default=500,
                        help='Step to enable dynamic sampling')
    parser.add_argument('--rpt_entropy_threshold', type=float, default=2.0,
                        help='Entropy threshold for token filtering')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')
    parser.add_argument('--save_dir', type=str, default='checkpoints/rpt',
                        help='Save directory')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluate every N steps')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='lfm-rpt',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model}-rpt-{args.rpt_temperature}",
            config=vars(args)
        )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.base_model:
        # Load from pre-trained base model
        logger.info(f"Loading base model from {args.base_model}")
        # Implement loading logic based on your base model format
        model = create_lfm_model(args.model)
    else:
        model = create_lfm_model(args.model)
    
    model = model.to(device)
    
    # Get config
    config = get_config(args.model)
    
    # Create RPT configuration
    rpt_config = RPTConfig(
        learning_rate=args.rpt_lr,
        temperature=args.rpt_temperature,
        batch_size=args.rpt_batch_size,
        num_samples=args.rpt_num_samples,
        max_seq_length=args.rpt_max_seq_length,
        training_steps=args.rpt_training_steps,
        dynamic_sampling_start=args.rpt_dynamic_sampling_start,
        entropy_threshold=args.rpt_entropy_threshold,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Create tokenizer (placeholder - use actual tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MathReasoningDataset(
        args.data_path,
        tokenizer,
        max_length=args.rpt_max_seq_length,
        split='train'
    )
    val_dataset = MathReasoningDataset(
        args.data_path,
        tokenizer,
        max_length=args.rpt_max_seq_length,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.rpt_batch_size // args.gradient_accumulation_steps,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.rpt_batch_size // args.gradient_accumulation_steps,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create RPT trainer
    logger.info("Creating RPT trainer...")
    rpt_trainer = create_rpt_trainer(
        model=model,
        config=rpt_config,
        tokenizer=tokenizer,
        proxy_model=None  # Could use a smaller model for entropy calculation
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.rpt_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.rpt_lr * 0.1
    )
    
    # Training loop
    logger.info("Starting RPT training...")
    best_eval_reward = -float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch_rpt(
            rpt_trainer,
            train_loader,
            optimizer,
            scheduler,
            epoch + 1,
            wandb_log=args.use_wandb
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Metrics: {train_metrics}")
        
        # Evaluate
        if (epoch + 1) % args.eval_steps == 0 or epoch == args.epochs - 1:
            eval_metrics = evaluate_rpt(rpt_trainer, val_loader)
            logger.info(f"Epoch {epoch + 1} - Eval Metrics: {eval_metrics}")
            
            if args.use_wandb:
                wandb.log({
                    f'eval/{k}': v for k, v in eval_metrics.items()
                })
            
            # Save best checkpoint
            if eval_metrics['eval_avg_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['eval_avg_reward']
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'rpt_config': rpt_config,
                    'model_config': config,
                }
                checkpoint_path = os.path.join(args.save_dir, f'{args.model}_rpt_best.pt')
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'rpt_config': rpt_config,
            'model_config': config,
        }
        checkpoint_path = os.path.join(args.save_dir, f'{args.model}_rpt_latest.pt')
        torch.save(checkpoint, checkpoint_path)
    
    logger.info("RPT training completed!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()