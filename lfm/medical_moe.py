"""
Medical-Specialized Mixture of Experts (MoE) Model
Implements domain-specific expert routing for medical tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from lfm.config import LFMConfig
from lfm.custom_moe import CustomExpert, AttentionRouter


@dataclass
class MedicalConfig(LFMConfig):
    """Configuration for medical-specialized model"""
    # Medical domain experts
    num_medical_experts: int = 12
    medical_specialties: List[str] = None
    
    # Safety and compliance
    enable_safety_checks: bool = True
    confidence_threshold: float = 0.85
    require_evidence: bool = True
    
    # Expert specialization
    specialty_expert_mapping: Dict[str, List[int]] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.medical_specialties is None:
            self.medical_specialties = [
                "cardiology",
                "neurology", 
                "oncology",
                "radiology",
                "pathology",
                "pharmacology",
                "emergency",
                "pediatrics",
                "surgery",
                "psychiatry",
                "internal_medicine",
                "general"
            ]
        
        # Map specialties to expert indices
        if self.specialty_expert_mapping is None:
            self.specialty_expert_mapping = {}
            experts_per_specialty = self.num_medical_experts // len(self.medical_specialties)
            for i, specialty in enumerate(self.medical_specialties):
                start_idx = i * experts_per_specialty
                end_idx = start_idx + experts_per_specialty
                self.specialty_expert_mapping[specialty] = list(range(start_idx, end_idx))


class MedicalExpert(nn.Module):
    """Specialized medical expert with domain knowledge"""
    
    def __init__(self, hidden_dim: int, intermediate_dim: int, specialty: str, dropout: float = 0.1):
        super().__init__()
        self.specialty = specialty
        self.hidden_dim = hidden_dim
        
        # Specialty-specific architecture
        if specialty in ["radiology", "pathology"]:
            # Visual-heavy specialties need different architecture
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim * 2, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        elif specialty in ["pharmacology"]:
            # Drug interactions need complex reasoning
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),  # Less dropout for precise calculations
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, intermediate_dim // 2),
                nn.GELU(),
                nn.Linear(intermediate_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else:
            # Standard medical expert
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, intermediate_dim // 2),
                nn.GELU(),
                nn.Linear(intermediate_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        # Confidence estimation layer
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with confidence estimation"""
        output = self.net(x)
        confidence = self.confidence_estimator(output)
        return output, confidence


class MedicalRouter(nn.Module):
    """Router that considers medical context and specialty"""
    
    def __init__(self, hidden_dim: int, num_experts: int, specialties: List[str], num_heads: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.specialties = specialties
        
        # Context understanding layers
        self.context_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Medical concept extraction
        self.concept_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(specialties))
        )
        
        # Expert routing based on medical concepts
        self.expert_router = nn.Sequential(
            nn.Linear(hidden_dim + len(specialties), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Urgency/priority detection
        self.urgency_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            expert_logits: Routing logits for each expert
            specialty_probs: Probability distribution over medical specialties
            urgency_score: Urgency score (0-1)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Self-attention to understand context
        attn_out, _ = self.context_attention(x, x, x)
        
        # Pool over sequence for routing decisions
        pooled = attn_out.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Extract medical concepts
        specialty_logits = self.concept_extractor(pooled)
        specialty_probs = F.softmax(specialty_logits, dim=-1)
        
        # Combine features for expert routing
        routing_features = torch.cat([pooled, specialty_probs], dim=-1)
        expert_logits = self.expert_router(routing_features)
        
        # Detect urgency
        urgency_score = self.urgency_detector(pooled)
        
        # Expand back to sequence length
        expert_logits = expert_logits.unsqueeze(1).expand(-1, seq_len, -1)
        
        return expert_logits, specialty_probs, urgency_score


class MedicalMoE(nn.Module):
    """Medical-specialized Mixture of Experts"""
    
    def __init__(self, config: MedicalConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_medical_experts
        self.num_experts_per_token = config.n_experts_per_tok if hasattr(config, 'n_experts_per_tok') else 3
        self.hidden_dim = config.dim
        self.intermediate_dim = config.intermediate_size if hasattr(config, 'intermediate_size') else config.dim * 4
        
        # Medical router
        self.router = MedicalRouter(
            self.hidden_dim,
            self.num_experts,
            config.medical_specialties
        )
        
        # Create medical experts
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            # Assign specialty to expert
            specialty_idx = i % len(config.medical_specialties)
            specialty = config.medical_specialties[specialty_idx]
            
            expert = MedicalExpert(
                self.hidden_dim,
                self.intermediate_dim,
                specialty=specialty,
                dropout=0.1
            )
            self.experts.append(expert)
        
        # Safety gate for high-risk decisions
        self.safety_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Evidence requirement module
        self.evidence_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # Adaptive expert weighting based on confidence
        self.confidence_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        medical_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with medical safety checks
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            medical_context: Optional medical context information
        
        Returns:
            output: Processed tensor
            aux_outputs: Dictionary with auxiliary outputs (confidence, safety, etc.)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing decisions
        expert_logits, specialty_probs, urgency_score = self.router(hidden_states)
        expert_logits_flat = expert_logits.view(-1, self.num_experts)
        expert_probs = F.softmax(expert_logits_flat, dim=-1)
        
        # For urgent cases, use more experts
        num_experts_to_use = self.num_experts_per_token
        if urgency_score.mean() > 0.7:
            num_experts_to_use = min(self.num_experts_per_token + 2, self.num_experts)
        
        # Select top experts
        expert_weights, expert_indices = torch.topk(
            expert_probs, num_experts_to_use, dim=-1
        )
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Process through experts with confidence tracking
        expert_output = torch.zeros_like(hidden_states_flat)
        total_confidence = torch.zeros(batch_size * seq_len, 1, device=hidden_states.device)
        
        for i in range(num_experts_to_use):
            expert_idx = expert_indices[:, i]
            expert_weight = expert_weights[:, i]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_out, confidence = self.experts[e](expert_input)
                    
                    # Weight by both routing weight and confidence
                    combined_weight = expert_weight[mask].unsqueeze(-1) * confidence
                    expert_output[mask] += combined_weight * expert_out
                    total_confidence[mask] += confidence
        
        # Reshape output
        output = expert_output.view(batch_size, seq_len, hidden_dim)
        
        # Safety checks for high-risk content
        safety_score = self.safety_gate(output.mean(dim=1))
        
        # Apply safety scaling if needed
        if self.config.enable_safety_checks:
            safety_mask = (safety_score < 0.5).float().unsqueeze(-1).unsqueeze(-1)
            output = output * (1 - safety_mask * 0.5)  # Reduce confidence for unsafe content
        
        # Evidence extraction if required
        evidence_features = None
        if self.config.require_evidence:
            evidence_features = self.evidence_extractor(output)
        
        # Prepare auxiliary outputs
        aux_outputs = {
            'specialty_distribution': specialty_probs,
            'urgency_score': urgency_score,
            'average_confidence': total_confidence.view(batch_size, seq_len, 1).mean(),
            'safety_score': safety_score,
            'expert_usage': expert_indices.view(batch_size, seq_len, -1),
            'evidence_features': evidence_features
        }
        
        return output, aux_outputs


class MedicalLFM(nn.Module):
    """Complete Medical Language Foundation Model"""
    
    def __init__(self, base_model, config: MedicalConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Replace MoE layers with medical MoE
        self._replace_moe_layers()
        
        # Add medical-specific heads
        self.add_medical_heads()
    
    def _replace_moe_layers(self):
        """Replace standard MoE layers with medical MoE"""
        for name, module in self.base_model.named_modules():
            if "moe" in name.lower() or "mixture" in name.lower():
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = self.base_model
                
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                
                # Replace with medical MoE
                medical_moe = MedicalMoE(self.config)
                setattr(parent, attr_name, medical_moe)
    
    def add_medical_heads(self):
        """Add task-specific heads for medical tasks"""
        hidden_dim = self.config.dim
        
        # Diagnosis prediction head
        self.diagnosis_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1000)  # ICD-10 codes
        )
        
        # Treatment recommendation head
        self.treatment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 500)  # Common treatments
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
        return_medical_outputs: bool = False
    ):
        """Forward pass with medical task routing"""
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if not return_medical_outputs:
            return outputs
        
        # Extract hidden states
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        
        # Pool for classification tasks
        if attention_mask is not None:
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Apply task-specific heads
        medical_outputs = {}
        if task == "diagnosis" or task is None:
            medical_outputs['diagnosis_logits'] = self.diagnosis_head(pooled)
        
        if task == "treatment" or task is None:
            medical_outputs['treatment_logits'] = self.treatment_head(pooled)
        
        if task == "risk" or task is None:
            medical_outputs['risk_score'] = self.risk_head(pooled)
        
        # Add medical outputs to model outputs
        if hasattr(outputs, '__dict__'):
            for key, value in medical_outputs.items():
                setattr(outputs, key, value)
        else:
            outputs = (outputs[0], medical_outputs)
        
        return outputs


def create_medical_model(base_model_name: str = "liquid/lfm-3b", **kwargs) -> MedicalLFM:
    """Create a medical-specialized LFM model"""
    from transformers import AutoModelForCausalLM
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Create medical config
    config = MedicalConfig(**kwargs)
    
    # Create medical model
    medical_model = MedicalLFM(base_model, config)
    
    return medical_model


# Example usage
if __name__ == "__main__":
    # Create medical model
    model = create_medical_model(
        base_model_name="liquid/lfm-3b",
        num_medical_experts=12,
        enable_safety_checks=True,
        confidence_threshold=0.9
    )
    
    print(f"Created medical model with {model.config.num_medical_experts} experts")
    print(f"Medical specialties: {model.config.medical_specialties}")