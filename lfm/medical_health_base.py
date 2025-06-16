#!/usr/bin/env python3
"""
Medical Health Base Model with Specialized MoE
Combines the existing medical_moe.py with LFM_3B architecture for health applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, field
import json
import sys
import os

# Import from existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfm.config import LFMConfig
from lfm.medical_moe import MedicalConfig, MedicalMoE, MedicalExpert, MedicalRouter

# Add LFM_3B to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LFM_3B'))
from LFM_3B.model import RMSNorm, RotaryPositionalEmbedding, apply_rotary_pos_emb


@dataclass
class MedicalHealthConfig(MedicalConfig):
    """Enhanced Medical Health Configuration"""
    
    # Base model architecture (from LFM-3B)
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    vocab_size: int = 50257
    max_position_embeddings: int = 4096
    
    # Medical Health specific
    health_specialties: List[str] = field(default_factory=lambda: [
        "primary_care",      # Hausarzt
        "emergency",         # Notfall
        "cardiology",        # Herz
        "neurology",         # Neurologie
        "oncology",          # Krebs
        "pediatrics",        # Kinder
        "psychiatry",        # Psyche
        "pharmacy",          # Medikamente
        "radiology",         # Bildgebung
        "surgery",           # Chirurgie
        "internal_medicine", # Innere Medizin
        "preventive_care"    # Prävention
    ])
    
    # Safety and compliance for health
    phi_protection: bool = True          # PHI (Protected Health Information) protection
    medical_disclaimer: bool = True      # Add medical disclaimers
    evidence_based: bool = True          # Require evidence-based responses
    uncertainty_estimation: bool = True  # Estimate uncertainty in responses
    
    # Regulatory compliance
    hipaa_compliant: bool = True         # HIPAA compliance
    gdpr_compliant: bool = True          # GDPR compliance for EU
    medical_device_class: str = "II"     # Medical device classification
    
    # Language support
    multilingual: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ["en", "de", "es", "fr"])
    
    # Risk assessment
    risk_categories: List[str] = field(default_factory=lambda: [
        "low",      # Niedrig
        "moderate", # Mittel
        "high",     # Hoch
        "critical"  # Kritisch
    ])
    
    def __post_init__(self):
        # Set medical specialties from health specialties
        self.medical_specialties = self.health_specialties
        self.num_medical_experts = len(self.health_specialties)
        
        # Compatibility with LFMConfig
        self.dim = self.hidden_size
        self.n_layers = self.num_hidden_layers
        self.n_heads = self.num_attention_heads
        self.n_experts_per_tok = 3  # Use 3 experts for medical decisions
        
        # Liquid Neural Network attributes (for compatibility with print_model_summary)
        self.use_liquid_layers = False
        self.liquid_num_reservoirs = 4
        self.liquid_reservoir_size = 256
        self.liquid_update_interval = 4
        
        # MoE attributes (for compatibility with print_model_summary)
        self.num_experts = self.num_medical_experts
        self.num_experts_per_token = self.n_experts_per_tok
        
        # Initialize parent if it has __post_init__
        if hasattr(super(), '__post_init__'):
            super().__post_init__()


class HealthExpert(MedicalExpert):
    """Enhanced Medical Expert for Health Applications"""
    
    def __init__(self, hidden_dim: int, intermediate_dim: int, specialty: str, config: MedicalHealthConfig):
        super().__init__(hidden_dim, intermediate_dim, specialty)
        self.config = config
        
        # Add health-specific layers
        if specialty == "pharmacy":
            # Drug interaction and dosage calculator
            self.drug_interaction_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # Interaction risk score
            )
        
        if specialty == "emergency":
            # Triage scoring system
            self.triage_scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, len(config.risk_categories)),
                nn.Softmax(dim=-1)
            )
        
        if specialty == "preventive_care":
            # Risk prediction for preventive measures
            self.risk_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 10),  # Common health risks
                nn.Sigmoid()
            )
        
        # PHI detection layer (if enabled)
        if config.phi_protection:
            self.phi_detector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # PHI probability
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced forward pass with health-specific outputs"""
        output, confidence = super().forward(x)
        
        health_outputs = {"confidence": confidence}
        
        # Specialty-specific outputs
        if hasattr(self, 'drug_interaction_layer'):
            health_outputs['drug_interaction_risk'] = self.drug_interaction_layer(output)
        
        if hasattr(self, 'triage_scorer'):
            health_outputs['triage_score'] = self.triage_scorer(output)
        
        if hasattr(self, 'risk_predictor'):
            health_outputs['health_risks'] = self.risk_predictor(output)
        
        if hasattr(self, 'phi_detector'):
            health_outputs['phi_probability'] = self.phi_detector(output)
        
        return output, health_outputs


class HealthAttention(nn.Module):
    """Medical-aware attention mechanism"""
    
    def __init__(self, config: MedicalHealthConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
        )
        
        # Medical context attention bias
        self.medical_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        medical_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add medical bias for health-related tokens
        attn_scores = attn_scores + self.medical_bias
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class HealthMoELayer(nn.Module):
    """Health-specialized MoE Layer"""
    
    def __init__(self, config: MedicalHealthConfig):
        super().__init__()
        self.config = config
        
        # Create health experts
        self.experts = nn.ModuleList([
            HealthExpert(
                config.hidden_size,
                config.intermediate_size,
                specialty,
                config
            )
            for specialty in config.health_specialties
        ])
        
        # Medical router
        self.router = MedicalRouter(
            config.hidden_size,
            len(config.health_specialties),
            config.health_specialties
        )
        
        # Safety and compliance layers
        if config.phi_protection:
            self.phi_filter = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
        if config.uncertainty_estimation:
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route to experts
        expert_logits, specialty_probs, urgency_score = self.router(hidden_states)
        expert_probs = F.softmax(expert_logits, dim=-1)
        
        # Select top experts (3 for medical decisions)
        top_experts = torch.topk(expert_probs, self.config.n_experts_per_tok, dim=-1)
        expert_weights = F.softmax(top_experts.values, dim=-1)
        expert_indices = top_experts.indices
        
        # Process through experts
        expert_outputs = []
        health_metadata = {}
        
        for i, expert in enumerate(self.experts):
            # Check if this expert is selected
            mask = (expert_indices == i).any(dim=-1)
            if mask.any():
                expert_input = hidden_states[mask]
                if expert_input.numel() > 0:
                    expert_out, expert_health = expert(expert_input)
                    expert_outputs.append((i, expert_out, expert_health, mask))
        
        # Combine expert outputs
        output = torch.zeros_like(hidden_states)
        combined_health_outputs = {}
        
        for expert_idx, expert_out, expert_health, mask in expert_outputs:
            # Get weights for this expert
            weight_mask = (expert_indices == expert_idx)
            weights = expert_weights[weight_mask].mean(dim=-1, keepdim=True).unsqueeze(-1)
            
            # Apply weighted output
            output[mask] += weights * expert_out
            
            # Combine health outputs
            for key, value in expert_health.items():
                if key not in combined_health_outputs:
                    combined_health_outputs[key] = []
                combined_health_outputs[key].append(value)
        
        # Safety checks
        aux_outputs = {
            'specialty_distribution': specialty_probs,
            'urgency_score': urgency_score,
            'expert_usage': expert_indices,
        }
        
        # Add health-specific outputs
        for key, values in combined_health_outputs.items():
            if values:
                aux_outputs[key] = torch.cat(values, dim=0).mean()
        
        # PHI protection
        if self.config.phi_protection and hasattr(self, 'phi_filter'):
            phi_score = self.phi_filter(output.mean(dim=1))
            aux_outputs['phi_risk'] = phi_score
            
            # Mask potential PHI
            phi_mask = (phi_score > 0.7).float().unsqueeze(1).unsqueeze(2)
            output = output * (1 - phi_mask * 0.8)  # Reduce PHI content
        
        # Uncertainty estimation
        if self.config.uncertainty_estimation and hasattr(self, 'uncertainty_estimator'):
            uncertainty = self.uncertainty_estimator(output.mean(dim=1))
            aux_outputs['uncertainty'] = uncertainty
        
        return output, aux_outputs


class MedicalHealthTransformerBlock(nn.Module):
    """Transformer block with health-specific MoE"""
    
    def __init__(self, config: MedicalHealthConfig):
        super().__init__()
        self.attention = HealthAttention(config)
        self.moe = HealthMoELayer(config)
        self.ln1 = RMSNorm(config.hidden_size)
        self.ln2 = RMSNorm(config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attn_out
        
        # MoE with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        moe_out, aux_outputs = self.moe(hidden_states, attention_mask)
        hidden_states = residual + moe_out
        
        return hidden_states, aux_outputs


class MedicalHealthBaseModel(nn.Module):
    """Complete Medical Health Base Model"""
    
    def __init__(self, config: MedicalHealthConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MedicalHealthTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.hidden_size)
        
        # Health-specific heads
        self.diagnosis_head = nn.Linear(config.hidden_size, 1000)  # ICD-10 codes
        self.risk_head = nn.Linear(config.hidden_size, len(config.risk_categories))
        self.specialty_head = nn.Linear(config.hidden_size, len(config.health_specialties))
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_medical_outputs: bool = False,
        task: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Forward through layers
        all_aux_outputs = {}
        for layer in self.layers:
            hidden_states, aux_outputs = layer(hidden_states, attention_mask)
            
            # Accumulate auxiliary outputs
            for key, value in aux_outputs.items():
                if key not in all_aux_outputs:
                    all_aux_outputs[key] = []
                all_aux_outputs[key].append(value)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling logits
        lm_logits = self.lm_head(hidden_states)
        
        outputs = {
            'logits': lm_logits,
            'last_hidden_state': hidden_states,
            'auxiliary_outputs': all_aux_outputs
        }
        
        # Medical task outputs
        if return_medical_outputs:
            pooled = hidden_states.mean(dim=1)  # Simple pooling
            
            if task == "diagnosis" or task is None:
                outputs['diagnosis_logits'] = self.diagnosis_head(pooled)
            
            if task == "risk" or task is None:
                outputs['risk_logits'] = self.risk_head(pooled)
            
            if task == "specialty" or task is None:
                outputs['specialty_logits'] = self.specialty_head(pooled)
        
        return outputs


def create_medical_health_model(
    config_name: str = "small",
    **kwargs
) -> MedicalHealthBaseModel:
    """Create Medical Health Model with different sizes"""
    
    configs = {
        "tiny": MedicalHealthConfig(
            model_name="medical-health-tiny",
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            head_dim=64,
            intermediate_dim=1024,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=1024,
            vocab_size=32000,
            **kwargs
        ),
        "small": MedicalHealthConfig(
            model_name="medical-health-small",
            hidden_dim=1024,
            num_layers=12,
            num_heads=16,
            head_dim=64,
            intermediate_dim=2048,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            intermediate_size=2048,
            vocab_size=50257,
            **kwargs
        ),
        "base": MedicalHealthConfig(
            model_name="medical-health-base",
            hidden_dim=2048,
            num_layers=24,
            num_heads=16,
            head_dim=128,
            intermediate_dim=5120,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=5120,
            **kwargs
        ),
        "large": MedicalHealthConfig(
            model_name="medical-health-large",
            hidden_dim=3072,
            num_layers=32,
            num_heads=24,
            head_dim=128,
            intermediate_dim=8192,
            hidden_size=3072,
            num_hidden_layers=32,
            num_attention_heads=24,
            intermediate_size=8192,
            **kwargs
        )
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    model = MedicalHealthBaseModel(config)
    
    print(f"Created Medical Health Model ({config_name})")
    print(f"Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Health Specialties: {len(config.health_specialties)}")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_medical_health_model(
        config_name="small",
        phi_protection=True,
        uncertainty_estimation=True,
        multilingual=True
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    vocab_size = model.config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_medical_outputs=True
        )
    
    print(f"\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    print(f"\nAuxiliary outputs:")
    for key, values in outputs['auxiliary_outputs'].items():
        print(f"  {key}: {len(values)} layers")