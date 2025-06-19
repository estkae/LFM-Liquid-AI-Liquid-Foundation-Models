#!/usr/bin/env python3
"""
Improved training script for Municipal MoE Model with better hyperparameters
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig
from LFM_3B.create_municipal_training_data import create_municipal_training_data


class MunicipalDataset(Dataset):
    """Dataset for municipal administration texts"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(data['text'])
        
        # Data augmentation - add variations
        augmented = []
        for text in self.examples:
            augmented.append(text)
            # Add with different formatting
            if "Frage:" in text and "Antwort:" in text:
                augmented.append(text.replace("Frage:", "Q:").replace("Antwort:", "A:"))
            # Add with prompt prefix
            if random.random() < 0.3:
                augmented.append(f"Bürgerfrage: {text}")
        
        self.examples = augmented
        print(f"📚 Loaded {len(self.examples)} training examples (with augmentation)")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize with proper padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels (same as input_ids but with -100 for padding)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def train_model(
    model_path: str,
    data_file: str,
    output_dir: str,
    epochs: int = 5,  # More epochs
    batch_size: int = 2,  # Smaller batch size
    learning_rate: float = 3e-4,  # Higher learning rate
    max_length: int = 256,
    gradient_accumulation_steps: int = 4,
    resume_from_checkpoint: str = None
):
    """Train the Municipal MoE model with improved settings"""
    
    print("🏛️ Starting improved Municipal MoE training...")
    
    # Load model and tokenizer
    if resume_from_checkpoint:
        print(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint_path = Path(output_dir) / resume_from_checkpoint
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints:")
            output_path = Path(output_dir)
            if output_path.exists():
                checkpoints = [d.name for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
                for cp in sorted(checkpoints):
                    print(f"  - {cp}")
            return
        model = MunicipalMoEModel.from_pretrained(checkpoint_path)
        print(f"✅ Loaded model from checkpoint: {checkpoint_path}")
    else:
        print(f"📂 Loading base model from {model_path}")
        model = MunicipalMoEModel.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🖥️ Training on {device}")
    
    # Create dataset and dataloader
    dataset = MunicipalDataset(data_file, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate total steps
    total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
    
    # Setup scheduler with more warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.2 * total_steps),  # 20% warmup
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}",
                'step': global_step
            })
            
            # Save checkpoint every 200 steps
            if global_step % 200 == 0 and global_step > 0:
                checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\n💾 Saved checkpoint to {checkpoint_dir}")
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_dir = Path(output_dir) / "best_model"
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"🏆 New best model saved with loss: {best_loss:.4f}")
    
    # Save final model
    print(f"\n💾 Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "base_model": model_path,
        "training_data": data_file,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "final_loss": avg_loss,
        "best_loss": best_loss
    }
    
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Training completed! Model saved to {output_dir}")
    
    # Test generation with better decoding
    print("\n🧪 Testing trained model with improved generation...")
    test_examples = [
        "Ich möchte meinen Personalausweis verlängern",
        "Wie beantrage ich eine Baugenehmigung",
        "Was kostet eine Geburtsurkunde",
        "Frage: Wo kann ich mich ummelden?"
    ]
    
    model.eval()
    for prompt in test_examples:
        print(f"\n📝 Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate with better parameters
        with torch.no_grad():
            # Get prompt length
            prompt_length = inputs.input_ids.shape[1]
            
            # Generate
            generated = inputs.input_ids
            for _ in range(100):  # Max 100 new tokens
                outputs = model(input_ids=generated)
                logits = outputs['logits']
                
                # Get next token with temperature sampling
                next_token_logits = logits[:, -1, :] / 0.8  # Temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Stop if EOS or period
                if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.encode('.')[0]:
                    generated = torch.cat([generated, next_token], dim=1)
                    break
                
                generated = torch.cat([generated, next_token], dim=1)
        
        # Decode only the generated part
        full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):].strip()
        
        if generated_text:
            print(f"💬 Generated: {generated_text}")
        else:
            print(f"⚠️ No generation (may need more training)")


def main():
    parser = argparse.ArgumentParser(description="Improved Municipal MoE Training")
    parser.add_argument("--model-path", type=str, default="./municipal_moe_base",
                        help="Path to base model")
    parser.add_argument("--data-file", type=str, default="municipal_training_data.jsonl",
                        help="Training data file")
    parser.add_argument("--output-dir", type=str, default="./municipal_moe_trained_v2",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Resume training from checkpoint (e.g., checkpoint-1000)")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Training data not found: {args.data_file}")
        print("First create data with: python train_municipal_moe.py --create-data")
        return
    
    # Train model
    train_model(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()