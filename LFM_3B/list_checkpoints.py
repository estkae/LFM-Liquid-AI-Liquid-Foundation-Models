#!/usr/bin/env python3
"""
List available training checkpoints
"""

import os
from pathlib import Path
import argparse


def list_checkpoints(output_dir: str):
    """List all available checkpoints in the output directory"""
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    print(f"📁 Checking for checkpoints in: {output_dir}")
    print("="*60)
    
    # Find all checkpoint directories
    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoints.append(item.name)
    
    if not checkpoints:
        print("❌ No checkpoints found.")
        print("\nNote: Checkpoints are saved every 200 steps during training.")
        return
    
    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    
    print(f"✅ Found {len(checkpoints)} checkpoints:")
    print()
    
    for i, checkpoint in enumerate(checkpoints, 1):
        checkpoint_path = output_path / checkpoint
        
        # Get size of checkpoint
        total_size = 0
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        
        # Check if this is the best model
        is_best = (output_path / "best_model").exists() and \
                 (output_path / "best_model").resolve() == checkpoint_path.resolve()
        
        best_marker = " 🏆 (BEST)" if is_best else ""
        
        print(f"{i:2d}. {checkpoint}{best_marker}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Path: {checkpoint_path}")
        print()
    
    print("="*60)
    print("💡 To resume training from a checkpoint, use:")
    print(f"   --resume-from-checkpoint {checkpoints[-1]}")
    print()
    print("📋 Latest checkpoint command:")
    print(f"python3 train_municipal_moe_improved.py \\")
    print(f"    --model-path ./municipal_moe_base \\")
    print(f"    --data-file massive_municipal_training_data.jsonl \\")
    print(f"    --output-dir {output_dir} \\")
    print(f"    --resume-from-checkpoint {checkpoints[-1]} \\")
    print(f"    --epochs 3 --batch-size 4 --learning-rate 5e-5")


def main():
    parser = argparse.ArgumentParser(description="List training checkpoints")
    parser.add_argument("--output-dir", type=str, default="./municipal_moe_massive_trained",
                        help="Output directory to check for checkpoints")
    
    args = parser.parse_args()
    list_checkpoints(args.output_dir)


if __name__ == "__main__":
    main()