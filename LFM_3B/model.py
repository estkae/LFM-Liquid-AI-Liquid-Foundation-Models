import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math
from dataclasses import dataclass

from .config import LFM3BConfig


@dataclass
class LFM3BOutput:
    """Output class for LFM-3B model"""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    liquid_states: Optional[Tuple[torch.Tensor]] = None
    router_logits: Optional[Tuple[torch.Tensor]] = None
    expert_indices: Optional[Tuple[torch.Tensor]] = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LFM3BAttention(nn.Module):
    """Multi-head attention with rotary position embeddings"""
    
    def __init__(self, config: LFM3BConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle past key values for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads if necessary
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value


class LiquidLayer(nn.Module):
    """Liquid Neural Network layer for integration with transformer"""
    
    def __init__(self, config: LFM3BConfig):
        super().__init__()
        self.config = config
        self.num_reservoirs = config.liquid_num_reservoirs
        self.reservoir_size = config.liquid_reservoir_size
        self.leak_rate = config.liquid_leak_rate
        self.spectral_radius = config.liquid_spectral_radius
        
        # Input projection
        self.input_proj = nn.Linear(config.hidden_size, self.reservoir_size * self.num_reservoirs)
        
        # Initialize reservoir weights
        self.reservoir_weights = nn.ParameterList([
            nn.Parameter(self._init_reservoir_weights()) 
            for _ in range(self.num_reservoirs)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.reservoir_size * self.num_reservoirs, config.hidden_size)
        
        # Layer norm
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def _init_reservoir_weights(self) -> torch.Tensor:
        """Initialize reservoir weights with specific spectral radius"""
        W = torch.randn(self.reservoir_size, self.reservoir_size)
        
        # Make sparse
        mask = torch.rand(self.reservoir_size, self.reservoir_size) < self.config.liquid_sparse_connectivity
        W = W * mask
        
        # Normalize to spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        spectral_radius = torch.max(torch.abs(eigenvalues))
        W = W * (self.spectral_radius / spectral_radius)
        
        return W
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Initialize state if needed
        if prev_state is None:
            prev_state = torch.zeros(
                batch_size, self.num_reservoirs * self.reservoir_size,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        # Project input
        input_proj = self.input_proj(hidden_states)  # [B, L, R*D]
        
        # Process through reservoirs
        all_states = []
        for t in range(seq_len):
            reservoir_input = input_proj[:, t, :]  # [B, R*D]
            new_states = []
            
            for i in range(self.num_reservoirs):
                start_idx = i * self.reservoir_size
                end_idx = (i + 1) * self.reservoir_size
                
                res_input = reservoir_input[:, start_idx:end_idx]
                res_state = prev_state[:, start_idx:end_idx]
                
                # Liquid dynamics: x(t+1) = (1-α)x(t) + α*tanh(Wx(t) + Win*u(t))
                new_state = (1 - self.leak_rate) * res_state + \
                           self.leak_rate * torch.tanh(
                               torch.matmul(res_state, self.reservoir_weights[i].T) + res_input
                           )
                new_states.append(new_state)
            
            prev_state = torch.cat(new_states, dim=1)
            all_states.append(prev_state.unsqueeze(1))
        
        # Concatenate all states
        all_states = torch.cat(all_states, dim=1)  # [B, L, R*D]
        
        # Project back to hidden size
        output = self.output_proj(all_states)
        
        # Residual connection and layer norm
        output = self.layer_norm(hidden_states + output)
        
        return output, prev_state


class ExpertLayer(nn.Module):
    """Single expert FFN layer"""
    
    def __init__(self, config: LFM3BConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    
    def __init__(self, config: LFM3BConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(config) for _ in range(self.num_experts)
        ])
        
        # Router
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Calculate router logits
        router_logits = self.router(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Execute experts
        final_output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # Get expert input
            expert_input = hidden_states_flat[expert_mask]
            
            # Run expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Get routing weights for this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            for i, token_idx in enumerate(token_indices):
                # Find which position this expert is in for this token
                expert_positions = (selected_experts[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                for pos in expert_positions:
                    weight = routing_weights[token_idx, pos]
                    final_output[token_idx] += weight * expert_output[i]
        
        final_output = final_output.view(batch_size, seq_len, hidden_dim)
        
        return final_output, router_logits.view(batch_size, seq_len, -1), selected_experts.view(batch_size, seq_len, -1)


class LFM3BDecoderLayer(nn.Module):
    """Single decoder layer combining attention, liquid dynamics, and MoE"""
    
    def __init__(self, config: LFM3BConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Self-attention
        self.self_attn = LFM3BAttention(config)
        
        # Liquid layer (only on certain layers)
        self.use_liquid = config.use_liquid_layers and (layer_idx % config.liquid_update_interval == 0)
        if self.use_liquid:
            self.liquid_layer = LiquidLayer(config)
        
        # Mixture of Experts FFN
        self.moe = MixtureOfExperts(config)
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Dropout
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        liquid_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        hidden_states = residual + self.hidden_dropout(attn_output)
        
        # Liquid layer (if applicable)
        if self.use_liquid:
            hidden_states, liquid_state = self.liquid_layer(hidden_states, liquid_state)
        
        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits, expert_indices = self.moe(hidden_states)
        hidden_states = residual + self.hidden_dropout(hidden_states)
        
        return hidden_states, present_key_value, liquid_state, router_logits, expert_indices


class LFM3BModel(nn.Module):
    """LFM-3B transformer model with liquid neural network integration"""
    
    def __init__(self, config: LFM3BConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            LFM3BDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        liquid_states: Optional[List[torch.Tensor]] = None,
        use_cache: bool = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> LFM3BOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_ids.shape)
        
        # Initialize position ids
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Initialize past key values and liquid states
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        if liquid_states is None:
            liquid_states = [None] * len(self.layers)
        
        # Forward through layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_router_logits = [] if output_router_logits else None
        all_expert_indices = [] if output_router_logits else None
        next_decoder_cache = [] if use_cache else None
        next_liquid_states = []
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx],
                liquid_state=liquid_states[idx],
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])
            
            next_liquid_states.append(layer_outputs[2])
            
            if output_attentions:
                all_attentions.append(layer_outputs[1])
            
            if output_router_logits:
                all_router_logits.append(layer_outputs[3])
                all_expert_indices.append(layer_outputs[4])
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return LFM3BOutput(
            logits=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            liquid_states=next_liquid_states,
            router_logits=all_router_logits,
            expert_indices=all_expert_indices,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape):
        batch_size, seq_length = input_shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=attention_mask.device),
            diagonal=1
        )
        
        # Expand for batch and heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
            combined_mask = causal_mask | (~expanded_mask)
        else:
            combined_mask = causal_mask
        
        # Convert to attention values
        return combined_mask.to(dtype=self.embed_tokens.weight.dtype) * torch.finfo(self.embed_tokens.weight.dtype).min


class LFM3BForCausalLM(nn.Module):
    """LFM-3B model with language modeling head"""
    
    def __init__(self, config: LFM3BConfig):
        super().__init__()
        self.config = config
        self.model = LFM3BModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        liquid_states: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> Union[LFM3BOutput, Tuple]:
        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            liquid_states=liquid_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )
        
        # Get logits
        hidden_states = outputs.logits
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add auxiliary loss for load balancing if using MoE
            if output_router_logits and outputs.router_logits is not None:
                aux_loss = self._compute_auxiliary_loss(outputs.router_logits)
                loss = loss + self.config.router_aux_loss_coef * aux_loss
        
        return LFM3BOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            liquid_states=outputs.liquid_states,
            router_logits=outputs.router_logits,
            expert_indices=outputs.expert_indices,
        )
    
    def _compute_auxiliary_loss(self, router_logits_list):
        """Compute load balancing auxiliary loss for MoE"""
        aux_loss = 0.0
        for router_logits in router_logits_list:
            if router_logits is not None:
                # Compute load balancing loss
                router_probs = F.softmax(router_logits, dim=-1)
                expert_usage = router_probs.mean(dim=[0, 1])  # Average over batch and sequence
                aux_loss += torch.var(expert_usage)
        return aux_loss / len(router_logits_list)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> torch.LongTensor:
        """Generate text using the model"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize past key values and liquid states
        past_key_values = None
        liquid_states = None
        
        # Generate tokens
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                liquid_states=liquid_states,
                use_cache=True,
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample or take argmax
            probs = F.softmax(next_token_logits, dim=-1)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Update past key values and liquid states
            past_key_values = outputs.past_key_values
            liquid_states = outputs.liquid_states
            
            # Check for EOS token
            if (next_tokens == eos_token_id).all():
                break
        
        return input_ids