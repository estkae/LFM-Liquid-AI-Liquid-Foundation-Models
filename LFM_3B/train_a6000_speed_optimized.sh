#!/bin/bash
# Speed-optimized training script for A6000 GPU (45 GiB)
# Optimiert für maximale Geschwindigkeit mit 8 CPUs und 45 GiB RAM

echo "🚀 A6000 SPEED-OPTIMIZED Municipal MoE Training"
echo "==============================================="
echo "GPU: NVIDIA A6000 (45 GiB)"
echo "CPU: 8 Cores (45 GiB RAM)"
echo ""

# Optimierungen für A6000:
# - Batch size 16 = maximale GPU-Auslastung
# - Gradient accumulation 2 = effektive batch size 32
# - Mixed precision (fp16) = 2x schneller, 50% weniger Memory
# - Max length 256 = schnellere Verarbeitung
# - 6 DataLoader workers = optimale CPU-Nutzung (8 CPUs - 2 für Hauptprozess)
# - Pin memory = schnellerer GPU Transfer
# - Gradient checkpointing aus = maximale Geschwindigkeit

echo "⚡ Starte Training mit A6000-optimierten Einstellungen..."

# Environment Variables für maximale Performance
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Optional: TF32 für A6000 aktivieren (noch schneller bei leicht reduzierter Präzision)
export NVIDIA_TF32_OVERRIDE=1

python3 train_municipal_moe_improved.py \
    --model-path ./municipal_moe_base \
    --data-file massive_municipal_training_data.jsonl \
    --output-dir ./municipal_moe_a6000_speed \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --gradient-accumulation-steps 2 \
    --max-length 256 \
    --fp16 \
    --dataloader-num-workers 6 \
    --dataloader-pin-memory \
    --logging-steps 50 \
    --save-steps 500 \
    --warmup-ratio 0.05 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --seed 42

echo ""
echo "✅ A6000 Speed-optimized Training abgeschlossen!"
echo "📊 Modell gespeichert in: ./municipal_moe_a6000_speed"