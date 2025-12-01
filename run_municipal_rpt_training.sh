#!/bin/bash

# RPT Training Script for Municipal MoE with Conda Environment

echo "Activating conda environment 'finetune'..."
source /home/$USER/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate finetune

echo "Setting up Municipal MoE base model..."
if [ ! -d "municipal_moe_base" ]; then
    echo "Running setup script for Municipal MoE..."
    python LFM_3B/setup_municipal_moe.py
fi

echo "Checking for math dataset..."
if [ ! -d "data/math" ]; then
    echo "Creating math dataset..."
    python create_math_dataset.py \
        --output_path data/math \
        --num_problems 4428
fi

echo "Starting Municipal MoE training with RPT..."

python train_municipal_moe_rpt.py \
    --base_model_path municipal_moe_base \
    --data_path data/math \
    --include_municipal_context \
    --rpt_lr 1e-6 \
    --rpt_temperature 0.8 \
    --rpt_batch_size 64 \
    --rpt_num_samples 8 \
    --rpt_max_seq_length 2048 \
    --rpt_training_steps 1000 \
    --rpt_dynamic_sampling_start 500 \
    --rpt_entropy_threshold 2.0 \
    --epochs 3 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --save_dir checkpoints/municipal_rpt \
    --eval_steps 100 \
    --use_wandb \
    --wandb_project municipal-moe-rpt-training

echo "Training completed!"
echo "Model saved to checkpoints/municipal_rpt/"