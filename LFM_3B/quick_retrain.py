#!/usr/bin/env python3
"""
Quick retraining with even more aggressive parameters
"""

import subprocess
import sys

def quick_retrain():
    """Retrain with more aggressive settings"""
    
    print("🔥 Quick retraining with aggressive parameters...")
    
    # More aggressive training
    cmd = [
        "python3", "train_municipal_moe_improved.py",
        "--model-path", "./municipal_moe_base",
        "--data-file", "municipal_training_data.jsonl", 
        "--output-dir", "./municipal_moe_retrained",
        "--epochs", "10",  # More epochs
        "--batch-size", "1",  # Smaller batch
        "--learning-rate", "5e-4",  # Higher learning rate
        "--gradient-accumulation-steps", "8",  # Larger effective batch
        "--max-length", "128"  # Shorter sequences for better learning
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Retraining completed!")
        print("\nTest the retrained model with:")
        print('python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_retrained/best_model --prompt "Frage: Wie beantrage ich einen Personalausweis?\\nAntwort:"')
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    quick_retrain()