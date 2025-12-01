# RPT (Reinforcement Preference Training) Integration for LFM

## Overview

This document describes the integration of Reinforcement Preference Training (RPT) into the Liquid Foundation Model (LFM) architecture. RPT is a training methodology that improves model reasoning capabilities through reinforcement learning with prefix-matching rewards.

## Key Features

### 1. Token-Level Filtering
- **Entropy-based filtering**: Focuses computational resources on challenging tokens
- **Proxy model support**: Can use a smaller model to calculate token entropy
- **Configurable threshold**: Adjustable entropy threshold for filtering

### 2. Prefix-Matching Reward System
- **Exact match rewards**: Binary reward (1 for exact match, 0 for mismatch)
- **Partial reward support**: Optional proportional rewards based on matched prefix length
- **Multi-token predictions**: Handles predictions spanning multiple tokens

### 3. Dynamic Sampling
- **Adaptive temperature**: Temperature decreases as training progresses
- **Configurable start**: Dynamic sampling activates after specified steps
- **Multiple response generation**: Generates G=8 responses per question by default

### 4. Integration with LFM Architecture
- **Seamless integration**: Works with all LFM model sizes (1B, 3B, 7B, 40B)
- **Liquid layer compatibility**: Maintains liquid neural network components
- **MoE support**: Compatible with Mixture of Experts architecture

## Usage

### Basic Training Script

```bash
python train_lfm_rpt.py \
    --model LFM-3B \
    --data_path /path/to/math/dataset \
    --rpt_lr 1e-6 \
    --rpt_temperature 0.8 \
    --rpt_batch_size 256 \
    --rpt_num_samples 8 \
    --epochs 3
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rpt_learning_rate` | 1e-6 | Conservative learning rate for stable training |
| `rpt_temperature` | 0.8 | Sampling temperature for exploration/exploitation |
| `rpt_num_samples` | 8 | Number of response samples per question |
| `rpt_entropy_threshold` | 2.0 | Threshold for token-level filtering |
| `rpt_dynamic_sampling_start` | 500 | Step to enable dynamic sampling |
| `rpt_kl_penalty` | 0.0 | KL divergence penalty (0 = no constraint) |

### Programmatic Usage

```python
from lfm.model_v2 import create_lfm_model
from lfm.rpt_training import RPTConfig, create_rpt_trainer

# Create model
model = create_lfm_model("LFM-3B")

# Configure RPT
rpt_config = RPTConfig(
    learning_rate=1e-6,
    temperature=0.8,
    num_samples=8,
    max_seq_length=8000
)

# Create trainer
rpt_trainer = create_rpt_trainer(
    model=model,
    config=rpt_config,
    tokenizer=tokenizer
)

# Training step
metrics = rpt_trainer.train_step(batch, optimizer)
```

## Implementation Details

### File Structure
```
lfm/
├── rpt_training.py      # Core RPT implementation
├── config.py           # Extended with RPT parameters
train_lfm_rpt.py        # Training script with RPT
```

### Key Classes

1. **`RPTConfig`**: Configuration dataclass for all RPT hyperparameters
2. **`TokenLevelFilter`**: Implements entropy-based token filtering
3. **`PrefixMatchingReward`**: Computes rewards for RL training
4. **`DynamicSampling`**: Manages adaptive sampling strategy
5. **`RPTTrainer`**: Main trainer class coordinating all components

### Training Flow

1. **Sample Generation**: Generate multiple responses for each input
2. **Reward Computation**: Calculate prefix-matching rewards
3. **Token Filtering**: Filter high-entropy tokens (if enabled)
4. **Policy Gradient**: Compute REINFORCE loss with rewards
5. **Combined Loss**: Mix policy gradient loss with standard LM loss

## Performance Considerations

- **Memory Usage**: Generating multiple samples increases memory requirements
- **Computational Cost**: Token filtering reduces computation on easy tokens
- **Batch Size**: Large batch sizes (256) recommended for stable training
- **Gradient Accumulation**: Use to achieve effective batch size on limited hardware

## Expected Results

Based on the RPT paper:
- Improved mathematical reasoning capabilities
- Better performance on competition-level problems
- More coherent multi-step reasoning
- Reduced hallucination in logical deductions

## Future Enhancements

1. **Multi-domain Support**: Extend beyond mathematical reasoning
2. **Curriculum Learning**: Progressive difficulty increase
3. **Distributed Training**: Multi-GPU/node support
4. **Online Learning**: Continuous improvement from user feedback