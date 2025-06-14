# LFM-3B: Liquid Foundation Model (3 Billion Parameters)

## Overview

LFM-3B is a 3-billion parameter language model that combines transformer architecture with Liquid Neural Networks (LNN) for enhanced dynamic processing and adaptation capabilities. The model integrates:

- **Transformer Architecture**: Multi-head attention with Rotary Position Embeddings (RoPE)
- **Liquid Neural Networks**: Dynamic reservoir computing integrated at regular intervals
- **Mixture of Experts (MoE)**: Sparse activation with top-k routing for efficiency
- **Medical Mode**: Special safety features for medical applications

## Architecture Details

### Model Configuration
- **Parameters**: ~3 billion
- **Hidden Size**: 3072
- **Layers**: 20
- **Attention Heads**: 24
- **Experts**: 8 (2 active per token)
- **Vocabulary**: 128,256 tokens
- **Context Length**: 8,192 tokens

### Key Components

1. **Liquid Layers**: Integrated every 4 transformer layers
   - 3 parallel reservoirs per layer
   - 512 units per reservoir
   - Sparse connectivity (10%)
   - Leak rate: 0.3

2. **Mixture of Experts**
   - 8 expert networks per layer
   - Top-2 routing
   - Load balancing auxiliary loss

3. **Attention Mechanism**
   - Grouped Query Attention (GQA) support
   - RoPE for position encoding
   - Flash attention compatible

## Installation

```bash
pip install torch numpy
```

## Usage

```python
from LFM_3B import LFM3BConfig, LFM3BForCausalLM
import torch

# Initialize model
config = LFM3BConfig()
model = LFM3BForCausalLM(config)

# Generate text
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Your tokenized input
output = model.generate(
    input_ids=input_ids,
    max_length=100,
    temperature=0.8,
    do_sample=True
)
```

## Medical Mode

For medical applications, enable safety features:

```python
config = LFM3BConfig(
    medical_mode=True,
    medical_safety_threshold=0.85
)
```

## Model Features

- **Dynamic Adaptation**: Liquid layers provide temporal dynamics
- **Efficient Scaling**: MoE enables sparse computation
- **Interpretability**: Evidence extraction from liquid states
- **Safety Features**: Medical mode with uncertainty estimation

## Training

The model supports standard language modeling objectives:
- Next token prediction
- Causal language modeling
- Auxiliary load balancing loss for MoE

## Future Enhancements

- [ ] Integration with tokenizers (HuggingFace compatible)
- [ ] Distributed training support
- [ ] RLHF/DPO fine-tuning
- [ ] Quantization support
- [ ] Model checkpointing utilities