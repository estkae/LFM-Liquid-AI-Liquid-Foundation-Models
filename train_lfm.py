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

from lfm.model_v2 import create_lfm_model
from lfm.config import get_config


class DummyTextDataset(Dataset):
    """Dummy dataset for testing - replace with your actual dataset"""
    
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random token sequences for testing
        tokens = torch.randint(1, self.vocab_size, (self.seq_length + 1,))
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        loss, _ = model(input_ids=input_ids, labels=labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _ = model(input_ids=input_ids, labels=labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train LFM model")
    parser.add_argument('--model', type=str, default='LFM-7B', 
                        choices=['LFM-1B', 'LFM-3B', 'LFM-7B', 'LFM-40B'],
                        help='Model size to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                        help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = create_lfm_model(args.model)
    model = model.to(device)
    
    # Get config
    config = get_config(args.model)
    
    # Create dummy dataset (replace with your actual dataset)
    logger.info("Creating datasets...")
    train_dataset = DummyTextDataset(config.vocab_size, args.seq_length, num_samples=10000)
    val_dataset = DummyTextDataset(config.vocab_size, args.seq_length, num_samples=1000)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }
            checkpoint_path = os.path.join(args.save_dir, f'{args.model}_best.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config,
        }
        checkpoint_path = os.path.join(args.save_dir, f'{args.model}_latest.pt')
        torch.save(checkpoint, checkpoint_path)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()