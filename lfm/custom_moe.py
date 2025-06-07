import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from lfm.config import LFMConfig


class CustomExpert(nn.Module):
    """Custom expert with more complex architecture"""
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Linear(intermediate_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class AttentionRouter(nn.Module):
    """Router with attention mechanism for better expert selection"""
    def __init__(self, hidden_dim: int, num_experts: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
    
    def forward(self, x):
        # Use self-attention to capture token relationships
        attn_out, _ = self.attention(x, x, x)
        # Route based on attended features
        return self.router(attn_out)


class LoadBalancedMoE(nn.Module):
    """MoE with load balancing to ensure equal expert usage"""
    def __init__(self, config: LFMConfig, use_custom_experts: bool = True):
        super().__init__()
        self.config = config
        self.num_experts = config.n_experts if hasattr(config, 'n_experts') else 8
        self.num_experts_per_token = config.n_experts_per_tok if hasattr(config, 'n_experts_per_tok') else 2
        self.hidden_dim = config.dim
        self.intermediate_dim = config.intermediate_size if hasattr(config, 'intermediate_size') else config.dim * 4
        
        # Use attention-based router
        self.router = AttentionRouter(self.hidden_dim, self.num_experts)
        
        # Create experts
        if use_custom_experts:
            self.experts = nn.ModuleList([
                CustomExpert(self.hidden_dim, self.intermediate_dim)
                for _ in range(self.num_experts)
            ])
        else:
            # Standard MLP experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.intermediate_dim),
                    nn.SiLU(),
                    nn.Linear(self.intermediate_dim, self.hidden_dim)
                )
                for _ in range(self.num_experts)
            ])
        
        # Adaptive scaling per expert
        self.expert_scale = nn.Parameter(torch.ones(self.num_experts))
        
        # Load balancing loss weight
        self.load_balance_weight = 0.01
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        router_logits_flat = router_logits.view(-1, self.num_experts)
        router_probs = F.softmax(router_logits_flat, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Calculate load balancing loss
        expert_usage = torch.zeros(self.num_experts, device=hidden_states.device)
        expert_usage.scatter_add_(
            0, 
            expert_indices.flatten(), 
            torch.ones_like(expert_indices.flatten(), dtype=torch.float)
        )
        load_balance_loss = (expert_usage.std() / (expert_usage.mean() + 1e-6)) * self.load_balance_weight
        
        # Process tokens through selected experts
        expert_output = torch.zeros_like(hidden_states_flat)
        for i in range(self.num_experts_per_token):
            expert_idx = expert_indices[:, i]
            expert_weight = expert_weights[:, i]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_out = self.experts[e](expert_input)
                    expert_out = expert_out * self.expert_scale[e]
                    expert_output[mask] += expert_weight[mask].unsqueeze(-1) * expert_out
        
        output = expert_output.view(batch_size, seq_len, hidden_dim)
        return output, load_balance_loss


class ConditionalMoE(nn.Module):
    """MoE that can switch between different expert sets based on task/condition"""
    def __init__(self, config: LFMConfig, num_conditions: int = 3):
        super().__init__()
        self.num_conditions = num_conditions
        
        # Different expert sets for different tasks
        self.expert_sets = nn.ModuleList([
            LoadBalancedMoE(config, use_custom_experts=True)
            for _ in range(num_conditions)
        ])
        
        # Condition classifier
        self.condition_classifier = nn.Linear(config.dim, num_conditions)
    
    def forward(self, hidden_states: torch.Tensor, task_id: Optional[int] = None):
        if task_id is not None:
            # Use specific expert set for given task
            return self.expert_sets[task_id](hidden_states)
        else:
            # Automatically determine which expert set to use
            pooled = hidden_states.mean(dim=1)  # [batch, hidden_dim]
            condition_logits = self.condition_classifier(pooled)
            condition_probs = F.softmax(condition_logits, dim=-1)
            
            # Weighted combination of all expert sets
            outputs = []
            total_loss = 0
            for i, expert_set in enumerate(self.expert_sets):
                output, loss = expert_set(hidden_states)
                weight = condition_probs[:, i].unsqueeze(-1).unsqueeze(-1)
                outputs.append(output * weight)
                if loss is not None:
                    total_loss += loss * condition_probs[:, i].mean()
            
            return sum(outputs), total_loss


class HierarchicalMoE(nn.Module):
    """Two-level MoE: coarse routing to expert groups, then fine routing within groups"""
    def __init__(self, config: LFMConfig, num_groups: int = 4):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = config.dim
        
        # Coarse-grained router
        self.coarse_router = nn.Linear(self.hidden_dim, num_groups)
        
        # Create smaller MoE within each group
        group_config = config
        if hasattr(config, 'n_experts'):
            # Reduce experts per group
            group_config.n_experts = config.n_experts // num_groups
        
        self.expert_groups = nn.ModuleList([
            LoadBalancedMoE(group_config, use_custom_experts=True)
            for _ in range(num_groups)
        ])
    
    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.size(0)
        
        # Coarse routing
        pooled = hidden_states.mean(dim=1)
        group_logits = self.coarse_router(pooled)
        group_probs = F.softmax(group_logits, dim=-1)
        
        # Process through each expert group
        outputs = []
        total_loss = 0
        
        for i, expert_group in enumerate(self.expert_groups):
            group_output, group_loss = expert_group(hidden_states)
            weight = group_probs[:, i].unsqueeze(-1).unsqueeze(-1)
            outputs.append(group_output * weight)
            if group_loss is not None:
                total_loss += group_loss
        
        return sum(outputs), total_loss


# Example usage in model
def replace_moe_in_model(model, moe_type="load_balanced"):
    """Replace standard MoE layers with custom implementations"""
    
    moe_classes = {
        "load_balanced": LoadBalancedMoE,
        "conditional": ConditionalMoE,
        "hierarchical": HierarchicalMoE
    }
    
    for name, module in model.named_modules():
        if "moe" in name.lower() or "mixture" in name.lower():
            # Get parent module and attribute name
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model
            
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            
            # Replace with custom MoE
            if hasattr(module, "config"):
                config = module.config
            else:
                # Extract config from model
                config = model.config if hasattr(model, "config") else None
            
            if config and moe_type in moe_classes:
                new_moe = moe_classes[moe_type](config)
                setattr(parent, attr_name, new_moe)
                print(f"Replaced {name} with {moe_type} MoE")
    
    return model