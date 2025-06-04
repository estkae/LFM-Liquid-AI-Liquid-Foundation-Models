import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from loguru import logger

from .config import LFMConfig, get_config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x=None, seq_len=None):
        if seq_len is None:
            if x is None:
                raise ValueError("Either x or seq_len must be provided")
            seq_len = x.shape[1]
        
        # Get device from x if provided, otherwise use inv_freq device
        device = x.device if x is not None else self.inv_freq.device
            
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding to query and key tensors."""
    # cos/sin have shape [1, seq_len, head_dim]
    # position_ids has shape [batch_size, seq_len]
    # We need to gather the positions for each batch
    
    batch_size = position_ids.shape[0]
    seq_len = position_ids.shape[1]
    
    # Gather the cos/sin values for the positions
    # First, expand cos/sin to match batch size
    cos = cos.expand(batch_size, -1, -1)  # [batch_size, seq_len, head_dim]
    sin = sin.expand(batch_size, -1, -1)  # [batch_size, seq_len, head_dim]
    
    # Gather using position_ids
    position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, cos.shape[-1])  # [batch_size, seq_len, head_dim]
    cos = torch.gather(cos, 1, position_ids_expanded)  # [batch_size, seq_len, head_dim]
    sin = torch.gather(sin, 1, position_ids_expanded)  # [batch_size, seq_len, head_dim]
    
    # Add head dimension
    cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class AdaptiveAttention(nn.Module):
    """Multi-head attention with adaptive components"""
    
    def __init__(self, config: LFMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Adaptive component
        self.adapt_gate = nn.Linear(self.hidden_size, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Generate rotary embeddings - pass None for x since we're providing seq_len
        cos, sin = self.rotary_emb(None, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Adaptive gating
        adapt_gates = torch.sigmoid(self.adapt_gate(hidden_states.mean(dim=1)))  # [bsz, num_heads]
        adapt_gates = adapt_gates.unsqueeze(-1).unsqueeze(-1)  # [bsz, num_heads, 1, 1]
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights * adapt_gates
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class AdaptiveMoELayer(nn.Module):
    """Mixture of Experts layer with adaptive routing"""
    
    def __init__(self, config: LFMConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim
        
        # Router
        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
            )
            for _ in range(self.num_experts)
        ])
        
        # Adaptive scaling
        self.expert_scale = nn.Parameter(torch.ones(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Routing
        router_logits = self.router(hidden_states_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Process through experts
        expert_output = torch.zeros_like(hidden_states_flat)
        
        for i in range(self.num_experts_per_token):
            for expert_idx in range(self.num_experts):
                mask = expert_indices[:, i] == expert_idx
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_out = self.experts[expert_idx](expert_input)
                    expert_out = expert_out * self.expert_scale[expert_idx]
                    
                    weights = expert_weights[mask, i].unsqueeze(1)
                    expert_output[mask] += weights * expert_out
        
        return expert_output.view(batch_size, seq_len, hidden_dim)


class LFMDecoderLayer(nn.Module):
    """Transformer decoder layer with adaptive components"""
    
    def __init__(self, config: LFMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        
        self.self_attn = AdaptiveAttention(config)
        self.mlp = AdaptiveMoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LFMModel(nn.Module):
    """Liquid Foundation Model"""
    
    def __init__(self, config: LFMConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim, self.padding_idx)
        self.layers = nn.ModuleList([
            LFMDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class LFMForCausalLM(nn.Module):
    """LFM for causal language modeling"""
    
    def __init__(self, config: LFMConfig):
        super().__init__()
        self.config = config
        self.model = LFMModel(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        logits = self.lm_head(outputs)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return loss, logits

    @classmethod
    def from_config(cls, model_name: str):
        """Create model from pre-defined configuration"""
        config = get_config(model_name)
        return cls(config)
    
    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration to directory"""
        import os
        import json
        from pathlib import Path
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = save_path / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str, device='cpu'):
        """Load model from saved directory"""
        import json
        from pathlib import Path
        
        load_path = Path(model_directory)
        
        # Load configuration
        config_path = load_path / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = LFMConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = load_path / "pytorch_model.bin"
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        return model


# Helper function to create models
def create_lfm_model(model_name: str = "LFM-7B") -> LFMForCausalLM:
    """Create LFM model by name"""
    logger.info(f"Creating {model_name} model...")
    model = LFMForCausalLM.from_config(model_name)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params / 1e9:.2f}B parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_lfm_model("LFM-7B")
    
    # Test forward pass
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(0, 128256, (batch_size, seq_len))
    
    with torch.no_grad():
        loss, logits = model(input_ids)
    
    logger.info(f"Test passed! Output shape: {logits.shape}")