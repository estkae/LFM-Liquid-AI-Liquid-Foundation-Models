#!/bin/sh
# Speed-optimized training script for Municipal MoE

echo "🚀 SPEED-OPTIMIZED Municipal MoE Training"
echo "========================================"

# Speed optimizations explained:
# - Larger batch size (16) = fewer iterations
# - Gradient accumulation (8) = effective batch 128
# - Mixed precision training = 2x faster
# - Shorter sequences (128) = faster processing
# - DataLoader workers = parallel data loading
# - Gradient checkpointing = larger batches possible

echo "⚡ Training with maximum speed settings..."

# Set environment variables for speed
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

python3 train_municipal_moe_improved.py \
    --model-path ./municipal_moe_base \
    --data-file massive_municipal_training_data.jsonl \
    --output-dir ./municipal_moe_speed_trained \
    --epochs 2 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --gradient-accumulation-steps 8 \
    --max-length 128

echo "✅ Speed-optimized training complete!"