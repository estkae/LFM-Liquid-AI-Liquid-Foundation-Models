import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LFM_3B'))

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
from pathlib import Path

# Import Municipal MoE model
from LFM_3B.municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig

# Import RPT components
from lfm.rpt_training import RPTConfig, create_rpt_trainer

# Import data handling
from create_math_dataset import MathProblemGenerator


class MunicipalMathDataset(Dataset):
    """Dataset that combines municipal knowledge with mathematical reasoning"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_length: int = 8000,
        split: str = 'train',
        include_municipal_context: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.include_municipal_context = include_municipal_context
        
        # Load mathematical problems
        self.math_data = self._load_math_data(data_path)
        
        # Municipal context templates
        self.municipal_contexts = [
            "Als Mitarbeiter des Bauamts müssen Sie folgende Berechnung durchführen: ",
            "Für die Stadtkasse ist folgende Kalkulation erforderlich: ",
            "Das Ordnungsamt benötigt die Lösung für: ",
            "Im Rahmen der Verwaltungsarbeit beim Einwohnermeldeamt: ",
            "Für die Planung im Sozialamt berechnen Sie bitte: ",
            "Das Jugendamt fragt nach der Lösung von: ",
            "Zur Gebührenberechnung im Standesamt: ",
            "Für die kommunale Haushaltsplanung: "
        ]
    
    def _load_math_data(self, data_path: str):
        """Load mathematical problems from JSONL file"""
        import json
        
        file_path = os.path.join(data_path, f"{self.split}.jsonl")
        problems = []
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    problems.append(json.loads(line))
        else:
            # Generate problems on the fly if file doesn't exist
            logger.warning(f"Dataset file {file_path} not found. Generating problems on the fly.")
            generator = MathProblemGenerator()
            for _ in range(100):
                problems.append(generator.generate_problem())
                
        return problems
    
    def __len__(self):
        return len(self.math_data)
    
    def __getitem__(self, idx):
        problem = self.math_data[idx]
        
        # Optionally add municipal context
        if self.include_municipal_context and self.split == 'train':
            context = self.municipal_contexts[idx % len(self.municipal_contexts)]
            question = context + problem['question']
        else:
            question = problem['question']
        
        # Create prompt with German instructions
        prompt = f"Frage: {question}\nLösung: "
        answer = problem['solution'] + f"\nAntwort: {problem['answer']}"
        
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


def load_municipal_moe_checkpoint(checkpoint_path: str, config: MunicipalMoEConfig, device: torch.device):
    """Load a pre-trained Municipal MoE checkpoint"""
    model = MunicipalMoEModel(config)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using randomly initialized model.")
    
    return model.to(device)


def train_epoch_municipal_rpt(
    rpt_trainer,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
    wandb_log: bool = False
):
    """Train for one epoch using RPT with Municipal MoE"""
    model = rpt_trainer.model
    model.train()
    
    epoch_metrics = {
        'total_loss': 0,
        'lm_loss': 0,
        'pg_loss': 0,
        'avg_reward': 0,
        'max_reward': 0,
        'expert_usage': {},
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
        
        # Track expert usage if available
        if hasattr(model, 'get_expert_usage'):
            expert_usage = model.get_expert_usage()
            for expert_id, usage in expert_usage.items():
                if expert_id not in epoch_metrics['expert_usage']:
                    epoch_metrics['expert_usage'][expert_id] = 0
                epoch_metrics['expert_usage'][expert_id] += usage
        
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
            log_dict = {
                'train/loss': metrics.get('loss', 0),
                'train/lm_loss': metrics.get('lm_loss', 0),
                'train/pg_loss': metrics.get('pg_loss', 0),
                'train/avg_reward': metrics.get('avg_reward', 0),
                'train/max_reward': metrics.get('max_reward', 0),
                'train/lr': scheduler.get_last_lr()[0],
                'train/step': rpt_trainer.global_step
            }
            
            # Add expert usage to logs
            if hasattr(model, 'get_expert_usage'):
                for expert_id, usage in expert_usage.items():
                    expert_name = model.config.expert_domains.get(expert_id, f"expert_{expert_id}")
                    log_dict[f'train/expert_usage/{expert_name}'] = usage
                    
            wandb.log(log_dict)
    
    # Average epoch metrics
    avg_metrics = {
        k: v / epoch_metrics['num_batches'] 
        for k, v in epoch_metrics.items() 
        if k not in ['num_batches', 'expert_usage']
    }
    
    # Normalize expert usage
    if epoch_metrics['expert_usage']:
        total_usage = sum(epoch_metrics['expert_usage'].values())
        avg_metrics['expert_usage'] = {
            k: v / total_usage for k, v in epoch_metrics['expert_usage'].items()
        }
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Municipal MoE with RPT")
    
    # Model arguments
    parser.add_argument('--base_model_path', type=str, default='municipal_moe_base',
                        help='Path to pre-trained Municipal MoE model')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a specific checkpoint to load')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/math',
                        help='Path to training data')
    parser.add_argument('--include_municipal_context', action='store_true',
                        help='Add municipal context to math problems')
    
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
    parser.add_argument('--save_dir', type=str, default='checkpoints/municipal_rpt',
                        help='Save directory')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluate every N steps')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='municipal-moe-rpt',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"municipal-moe-rpt-{args.rpt_temperature}",
            config=vars(args)
        )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load Municipal MoE configuration
    config_path = os.path.join(args.base_model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = MunicipalMoEConfig(**config_dict)
    else:
        logger.info("Using default Municipal MoE configuration")
        config = MunicipalMoEConfig()
    
    # Load or create model
    if args.checkpoint_path:
        model = load_municipal_moe_checkpoint(args.checkpoint_path, config, device)
    else:
        # Try to load from base model path
        model_path = os.path.join(args.base_model_path, 'pytorch_model.bin')
        if os.path.exists(model_path):
            model = load_municipal_moe_checkpoint(model_path, config, device)
        else:
            logger.info("Creating new Municipal MoE model")
            model = MunicipalMoEModel(config).to(device)
    
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
    
    # Create tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MunicipalMathDataset(
        args.data_path,
        tokenizer,
        max_length=args.rpt_max_seq_length,
        split='train',
        include_municipal_context=args.include_municipal_context
    )
    val_dataset = MunicipalMathDataset(
        args.data_path,
        tokenizer,
        max_length=args.rpt_max_seq_length,
        split='val',
        include_municipal_context=False  # No context for validation
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
    logger.info("Creating RPT trainer for Municipal MoE...")
    rpt_trainer = create_rpt_trainer(
        model=model,
        config=rpt_config,
        tokenizer=tokenizer,
        proxy_model=None
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
    logger.info("Starting Municipal MoE RPT training...")
    best_eval_reward = -float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch_municipal_rpt(
            rpt_trainer,
            train_loader,
            optimizer,
            scheduler,
            epoch + 1,
            wandb_log=args.use_wandb
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Metrics: {train_metrics}")
        
        # Log expert usage
        if 'expert_usage' in train_metrics:
            logger.info("Expert usage distribution:")
            for expert_id, usage in train_metrics['expert_usage'].items():
                expert_name = config.expert_domains.get(int(expert_id), f"expert_{expert_id}")
                logger.info(f"  {expert_name}: {usage:.2%}")
        
        # Evaluate
        if (epoch + 1) % args.eval_steps == 0 or epoch == args.epochs - 1:
            eval_metrics = rpt_trainer.evaluate(val_loader, num_eval_samples=50)
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
                checkpoint_path = os.path.join(args.save_dir, 'municipal_moe_rpt_best.pt')
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
        checkpoint_path = os.path.join(args.save_dir, 'municipal_moe_rpt_latest.pt')
        torch.save(checkpoint, checkpoint_path)
    
    logger.info("Municipal MoE RPT training completed!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()