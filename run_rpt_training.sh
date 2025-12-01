#!/bin/bash

# RPT Training Script for LFM with Conda Environment

echo "Activating conda environment 'finetune'..."
source /home/$USER/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate finetune

echo "Starting LFM training with RPT..."

# Create math dataset if not exists
if [ ! -d "data/math" ]; then
    echo "Creating math dataset..."
    python create_math_dataset.py \
        --output_path data/math \
        --num_problems 4428  # Same as OmniMATH
fi

# Run RPT training
python train_lfm_rpt.py \
    --model LFM-3B \
    --data_path data/math \
    --rpt_lr 1e-6 \
    --rpt_temperature 0.8 \
    --rpt_batch_size 256 \
    --rpt_num_samples 8 \
    --rpt_max_seq_length 8000 \
    --rpt_training_steps 1000 \
    --rpt_dynamic_sampling_start 500 \
    --rpt_entropy_threshold 2.0 \
    --epochs 3 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --save_dir checkpoints/rpt \
    --eval_steps 100 \
    --use_wandb \
    --wandb_project lfm-rpt-training

echo "Training completed!"