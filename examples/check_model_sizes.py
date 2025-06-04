import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lfm.config import LFM_CONFIGS
from loguru import logger


def estimate_model_memory(config):
    """Estimate memory requirements for a model configuration"""
    total_params = config.total_params
    
    # Memory estimates (in GB)
    param_memory = (total_params * 4) / (1024**3)  # 4 bytes per float32
    param_memory_fp16 = (total_params * 2) / (1024**3)  # 2 bytes per float16
    
    # Rough estimates for activations and gradients
    # This is approximate and depends on batch size and sequence length
    activation_memory_per_token = (config.hidden_dim * config.num_layers * 4) / (1024**3)
    
    return {
        'total_params': total_params,
        'param_memory_fp32': param_memory,
        'param_memory_fp16': param_memory_fp16,
        'activation_memory_per_token': activation_memory_per_token
    }


def main():
    logger.info("LFM Model Size Analysis")
    logger.info("=" * 80)
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        logger.info("No GPU available")
        gpu_memory = 0
    
    logger.info("=" * 80)
    
    # Analyze each model configuration
    for model_name, config in LFM_CONFIGS.items():
        memory_info = estimate_model_memory(config)
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Total Parameters: {memory_info['total_params']:,} ({memory_info['total_params']/1e9:.2f}B)")
        logger.info(f"  Parameter Memory (FP32): {memory_info['param_memory_fp32']:.2f} GB")
        logger.info(f"  Parameter Memory (FP16): {memory_info['param_memory_fp16']:.2f} GB")
        logger.info(f"  Activation Memory per Token: ~{memory_info['activation_memory_per_token']:.4f} GB")
        
        # Estimate batch sizes for 8GB GPU
        if gpu_memory > 0:
            # Conservative estimate: 70% of GPU memory for model + activations
            available_memory = gpu_memory * 0.7
            
            # For inference (no gradients)
            inference_memory = memory_info['param_memory_fp16']  # Using FP16
            remaining_for_activations = available_memory - inference_memory
            
            if remaining_for_activations > 0:
                # Rough estimate: batch_size * seq_length * activation_memory_per_token
                max_tokens_inference = int(remaining_for_activations / memory_info['activation_memory_per_token'])
                logger.info(f"  Estimated max tokens for inference (FP16): ~{max_tokens_inference:,}")
                logger.info(f"  Suggested batch_size x seq_length combinations:")
                for seq_len in [128, 256, 512, 1024]:
                    max_batch = max_tokens_inference // seq_len
                    if max_batch > 0:
                        logger.info(f"    - seq_length={seq_len}: batch_size ≤ {max_batch}")
            else:
                logger.info(f"  ⚠️  Model too large for {gpu_memory:.1f}GB GPU even with FP16")
        
        logger.info(f"  Architecture Details:")
        logger.info(f"    - Hidden Dimension: {config.hidden_dim}")
        logger.info(f"    - Number of Layers: {config.num_layers}")
        logger.info(f"    - Number of Heads: {config.num_heads}")
        logger.info(f"    - Number of Experts: {config.num_experts}")
        logger.info(f"    - Intermediate Dimension: {config.intermediate_dim}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Recommendations for 8GB GPU:")
    logger.info("  - Use LFM-1B for comfortable inference and fine-tuning")
    logger.info("  - LFM-3B possible with small batch sizes and FP16")
    logger.info("  - LFM-7B and LFM-40B require larger GPUs or model parallelism")
    logger.info("  - Use gradient checkpointing for training larger models")
    logger.info("  - Consider quantization (INT8) for even better memory efficiency")


if __name__ == "__main__":
    main()