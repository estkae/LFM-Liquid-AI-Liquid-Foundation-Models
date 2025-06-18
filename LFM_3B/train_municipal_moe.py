#!/usr/bin/env python3
"""
Training script for Municipal MoE Model
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig
from LFM_3B.create_municipal_training_data import create_municipal_training_data


class MunicipalDataset(Dataset):
    """Dataset for municipal administration texts"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(data['text'])
        
        print(f"📚 Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For language modeling, labels are the same as input_ids
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def train_model(
    model_path: str,
    data_file: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 256
):
    """Train the Municipal MoE model"""
    
    print("🏛️ Starting Municipal MoE training...")
    
    # Load model and tokenizer
    print(f"📂 Loading model from {model_path}")
    model = MunicipalMoEModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🖥️ Training on {device}")
    
    # Create dataset and dataloader
    dataset = MunicipalDataset(data_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Save checkpoint every 500 steps
            if global_step % 500 == 0:
                checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                print(f"\n💾 Saved checkpoint to {checkpoint_dir}")
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
    
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
        "learning_rate": learning_rate,
        "max_length": max_length,
        "final_loss": avg_loss
    }
    
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Training completed! Model saved to {output_dir}")
    
    # Test generation
    print("\n🧪 Testing trained model...")
    test_prompts = [
        "Ich möchte meinen Personalausweis verlängern",
        "Wie beantrage ich eine Baugenehmigung",
        "Was kostet eine Geburtsurkunde"
    ]
    
    model.eval()
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            # Simple greedy generation
            generated = inputs.input_ids
            for _ in range(50):
                outputs = model(input_ids=generated)
                logits = outputs['logits']
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"💬 Response: {response}")


def main():
    parser = argparse.ArgumentParser(description="Train Municipal MoE Model")
    parser.add_argument("--model-path", type=str, default="./municipal_moe_base",
                        help="Path to base model")
    parser.add_argument("--data-file", type=str, default="municipal_training_data.jsonl",
                        help="Training data file")
    parser.add_argument("--output-dir", type=str, default="./municipal_moe_trained",
                        help="Output directory for trained model")
    parser.add_argument("--create-data", action="store_true",
                        help="Create training data")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create training data if requested
    if args.create_data:
        create_municipal_training_data(args.data_file)
        print(f"\n✅ Training data created: {args.data_file}")
        print("Now run training with: python train_municipal_moe.py")
        return
    
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
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()