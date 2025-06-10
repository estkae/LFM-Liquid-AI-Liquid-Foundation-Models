#!/usr/bin/env python3
"""
Medical Model Configuration for LFM
Demonstrates how to configure LFM models for medical use cases
"""

from lfm.config import LFMConfig, get_config
from lfm.model_v2 import LFMModel
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MedicalConfig:
    """Extended configuration for medical-specific features"""
    base_config: LFMConfig
    medical_specialties: List[str] = None
    use_safety_gates: bool = True
    confidence_threshold: float = 0.85
    evidence_extraction: bool = True
    urgency_adaptive_experts: bool = True  # Use more experts for urgent cases
    max_experts_urgent: int = 4  # Up to 4 experts for urgent cases
    
    def __post_init__(self):
        if self.medical_specialties is None:
            self.medical_specialties = [
                'cardiology', 'neurology', 'oncology', 'radiology',
                'pathology', 'pharmacology', 'emergency', 'pediatrics',
                'surgery', 'psychiatry', 'internal_medicine', 'general'
            ]


def create_medical_config(base_model: str = "LFM-3B") -> MedicalConfig:
    """Create a medical configuration based on a base LFM model"""
    
    # Start with base configuration
    base_config = get_config(base_model)
    
    # Modify for medical use
    base_config.num_experts = 12  # 12 medical specialty experts
    base_config.model_name = f"{base_model}-Medical"
    
    # Create medical config wrapper
    medical_config = MedicalConfig(
        base_config=base_config,
        medical_specialties=[
            'cardiology', 'neurology', 'oncology', 'radiology',
            'pathology', 'pharmacology', 'emergency', 'pediatrics',
            'surgery', 'psychiatry', 'internal_medicine', 'general'
        ],
        use_safety_gates=True,
        confidence_threshold=0.85,
        evidence_extraction=True,
        urgency_adaptive_experts=True,
        max_experts_urgent=4
    )
    
    return medical_config


def create_custom_medical_config() -> MedicalConfig:
    """Create a fully custom medical configuration"""
    
    # Manual configuration
    base_config = LFMConfig(
        model_name="LFM-3B-Medical-Custom",
        hidden_dim=3072,
        num_layers=32,
        num_heads=24,
        head_dim=128,
        intermediate_dim=8192,
        num_experts=12,  # 12 medical experts
        num_experts_per_token=2,  # Default 2, can be increased dynamically
        vocab_size=128256,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        dropout=0.0,
        use_bias=False,
        tie_word_embeddings=False
    )
    
    return MedicalConfig(base_config=base_config)


def load_medical_model(config: MedicalConfig) -> LFMModel:
    """Load a medical model with the given configuration"""
    
    # Initialize base model
    model = LFMModel(config.base_config)
    
    # Note: In a real implementation, you would replace the MoE layers
    # with MedicalMoE layers here. For now, this returns the base model
    # with medical-appropriate configuration.
    
    print(f"Loaded medical model: {config.base_config.model_name}")
    print(f"Number of experts: {config.base_config.num_experts}")
    print(f"Medical specialties: {len(config.medical_specialties)}")
    print(f"Safety features: {config.use_safety_gates}")
    
    return model


# Example usage
if __name__ == "__main__":
    # Method 1: Based on existing model
    print("Method 1: Creating medical config based on LFM-3B")
    medical_config_v1 = create_medical_config("LFM-3B")
    print(f"Config: {medical_config_v1.base_config.model_name}")
    print(f"Experts: {medical_config_v1.base_config.num_experts}")
    print(f"Specialties: {medical_config_v1.medical_specialties[:3]}...")
    print()
    
    # Method 2: Custom configuration
    print("Method 2: Creating custom medical config")
    medical_config_v2 = create_custom_medical_config()
    print(f"Config: {medical_config_v2.base_config.model_name}")
    print(f"Hidden dim: {medical_config_v2.base_config.hidden_dim}")
    print(f"Layers: {medical_config_v2.base_config.num_layers}")
    print()
    
    # Load model
    print("Loading medical model...")
    model = load_medical_model(medical_config_v1)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")