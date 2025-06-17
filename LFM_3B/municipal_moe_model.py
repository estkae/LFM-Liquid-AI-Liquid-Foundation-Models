#!/usr/bin/env python3
"""
Municipal MoE (Mixture of Experts) Model for German Municipal Administration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class MunicipalMoEConfig:
    """Configuration for Municipal MoE Model"""
    # Base model parameters
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    
    # MoE specific parameters
    num_experts: int = 8  # Number of expert networks
    num_experts_per_tok: int = 2  # Top-k experts selected per token
    expert_capacity: int = 128  # Max tokens per expert
    
    # Municipal domain experts mapping
    expert_domains: Dict[int, str] = None
    
    def __post_init__(self):
        if self.expert_domains is None:
            # Map expert indices to municipal departments
            self.expert_domains = {
                0: "einwohnermeldeamt",  # Registration office
                1: "bauamt",             # Building department
                2: "ordnungsamt",        # Public order office
                3: "stadtkasse",         # City treasury
                4: "sozialamt",          # Social welfare office
                5: "standesamt",         # Registry office
                6: "jugendamt",          # Youth welfare office
                7: "general"             # General administration
            }
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to directory"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.__dict__.copy()
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """Load configuration from directory"""
        config_path = Path(pretrained_path) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class MunicipalExpertRouter(nn.Module):
    """Router to select which experts to use for each token"""
    
    def __init__(self, config: MunicipalMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_domains = config.expert_domains
        
        # Gating network
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            expert_weights: [batch_size, seq_len, num_experts_per_tok]
            expert_indices: [batch_size, seq_len, num_experts_per_tok]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate router logits
        router_logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize weights with softmax
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        return expert_weights, expert_indices


class MunicipalExpert(nn.Module):
    """Single expert network specialized for municipal tasks"""
    
    def __init__(self, config: MunicipalMoEConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.domain = config.expert_domains.get(expert_id, "general")
        
        # Expert FFN layers
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        
        # Domain-specific layer (optional specialization)
        self.domain_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expert forward pass"""
        # Standard FFN
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        # Domain-specific transformation
        hidden_states = hidden_states + self.domain_projection(hidden_states)
        
        return hidden_states


class MunicipalMoELayer(nn.Module):
    """MoE layer with multiple municipal experts"""
    
    def __init__(self, config: MunicipalMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Create router
        self.router = MunicipalExpertRouter(config)
        
        # Create experts
        self.experts = nn.ModuleList([
            MunicipalExpert(config, expert_id=i) 
            for i in range(config.num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE forward pass
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get expert routing decisions
        expert_weights, expert_indices = self.router(hidden_states)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == i).any(dim=-1)  # [batch, seq_len]
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = hidden_states[expert_mask]
                
                # Run expert
                expert_output = self.experts[i](expert_input)
                
                # Find weights for this expert
                weights_for_expert = expert_weights[expert_indices == i]
                
                # Weighted sum back to output
                output[expert_mask] += expert_output * weights_for_expert.unsqueeze(-1)
        
        return output


class MunicipalMoEModel(nn.Module):
    """Complete Municipal MoE Model"""
    
    def __init__(self, config: MunicipalMoEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            # Attention layer
            self.layers.append(nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True
            ))
            # MoE FFN layer
            self.layers.append(MunicipalMoELayer(config))
            # Layer norm
            self.layers.append(nn.LayerNorm(config.hidden_size))
        
        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = word_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Process through layers
        for i in range(0, len(self.layers), 3):
            # Self-attention
            attn_layer = self.layers[i]
            norm_layer = self.layers[i + 2]
            
            residual = hidden_states
            hidden_states, _ = attn_layer(hidden_states, hidden_states, hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = norm_layer(hidden_states)
            
            # MoE FFN
            moe_layer = self.layers[i + 1]
            residual = hidden_states
            hidden_states = moe_layer(hidden_states)
            hidden_states = residual + hidden_states
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model and config"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        print(f"✅ Municipal MoE model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """Load model from directory"""
        # Load config
        config = MunicipalMoEConfig.from_pretrained(pretrained_path)
        
        # Create model
        model = cls(config)
        
        # Load weights
        weights_path = Path(pretrained_path) / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        return model


def create_municipal_moe_model(save_path: str = "./municipal_moe_base"):
    """Create and save a base Municipal MoE model"""
    
    # Create configuration
    config = MunicipalMoEConfig(
        vocab_size=50257,  # GPT-2 vocab size
        hidden_size=768,
        num_hidden_layers=6,  # Smaller for demo
        num_attention_heads=12,
        intermediate_size=3072,
        num_experts=8,
        num_experts_per_tok=2
    )
    
    # Create model
    model = MunicipalMoEModel(config)
    
    # Initialize weights (Xavier/Glorot initialization)
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    model.apply(_init_weights)
    
    # Save model
    model.save_pretrained(save_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Number of experts: {config.num_experts}")
    print(f"Experts per token: {config.num_experts_per_tok}")
    print(f"\nExpert domains:")
    for idx, domain in config.expert_domains.items():
        print(f"  Expert {idx}: {domain}")
    
    return model, config


if __name__ == "__main__":
    # Create the base model
    model, config = create_municipal_moe_model()
    print("\n✅ Municipal MoE base model created successfully!")