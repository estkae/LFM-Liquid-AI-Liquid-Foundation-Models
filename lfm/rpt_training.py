import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np
from loguru import logger
from dataclasses import dataclass
import math


@dataclass
class RPTConfig:
    """Configuration for Reinforcement Preference Training"""
    learning_rate: float = 1e-6  # Conservative learning rate as in paper
    temperature: float = 0.8  # Sampling temperature for exploration/exploitation balance
    batch_size: int = 256  # Questions per batch
    num_samples: int = 8  # G=8 responses per question
    max_seq_length: int = 8000  # Maximum token sequence length
    kl_penalty: float = 0.0  # No KL constraint from base model
    training_steps: int = 1000  # Total training steps
    dynamic_sampling_start: int = 500  # Enable dynamic sampling after this step
    entropy_threshold: float = 2.0  # Threshold for token-level filtering
    top_k_entropy: int = 16  # Top-k for entropy calculation
    gradient_accumulation_steps: int = 4
    reward_type: str = "prefix_matching"  # Reward mechanism


class TokenLevelFilter:
    """Implements token-level filtering based on entropy to focus on challenging predictions"""
    
    def __init__(self, config: RPTConfig, proxy_model: Optional[nn.Module] = None):
        self.config = config
        self.proxy_model = proxy_model
        self.top_k = config.top_k_entropy
        self.threshold = config.entropy_threshold
        
    def calculate_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy for each token position
        Args:
            logits: [batch_size, seq_len, vocab_size]
        Returns:
            entropy: [batch_size, seq_len]
        """
        # Get top-k logits for efficiency
        top_k_logits, _ = torch.topk(logits, k=min(self.top_k, logits.size(-1)), dim=-1)
        
        # Convert to probabilities
        probs = F.softmax(top_k_logits, dim=-1)
        
        # Calculate entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy
    
    def filter_high_entropy_positions(
        self, 
        input_ids: torch.Tensor, 
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter tokens to focus on high-entropy (challenging) positions
        """
        entropy = self.calculate_token_entropy(logits)
        
        # Create mask for high-entropy positions
        high_entropy_mask = entropy > self.threshold
        
        # Apply mask to get filtered positions
        filtered_positions = high_entropy_mask.nonzero(as_tuple=True)
        
        if len(filtered_positions[0]) == 0:
            # If no high-entropy positions, return original
            return input_ids, logits, labels
            
        # Extract high-entropy tokens
        filtered_input_ids = input_ids[filtered_positions]
        filtered_logits = logits[filtered_positions]
        filtered_labels = labels[filtered_positions] if labels is not None else None
        
        return filtered_input_ids, filtered_logits, filtered_labels


class PrefixMatchingReward:
    """Implements prefix-matching reward system for reinforcement learning"""
    
    def __init__(self, config: RPTConfig):
        self.config = config
        
    def compute_reward(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor,
        tokenizer = None
    ) -> torch.Tensor:
        """
        Compute prefix-matching rewards
        Args:
            predictions: [batch_size, seq_len] - predicted token ids
            ground_truth: [batch_size, seq_len] - ground truth token ids
            tokenizer: Optional tokenizer for handling special tokens
        Returns:
            rewards: [batch_size] - reward for each sequence
        """
        batch_size = predictions.size(0)
        rewards = torch.zeros(batch_size, device=predictions.device)
        
        for i in range(batch_size):
            pred_seq = predictions[i]
            gt_seq = ground_truth[i]
            
            # Find first mismatch position
            matches = pred_seq == gt_seq
            
            if tokenizer:
                # Handle special tokens and boundaries
                # Skip padding tokens
                pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                valid_mask = gt_seq != pad_token_id
                matches = matches & valid_mask
            
            # Reward = 1 if exact match, 0 otherwise
            if matches.all():
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
                
        return rewards
    
    def compute_partial_rewards(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute partial rewards based on prefix length matched
        Useful for encouraging longer correct prefixes
        """
        batch_size = predictions.size(0)
        seq_len = predictions.size(1)
        rewards = torch.zeros(batch_size, device=predictions.device)
        
        for i in range(batch_size):
            # Count matching prefix length
            matches = predictions[i] == ground_truth[i]
            
            # Find first mismatch
            first_mismatch = seq_len
            for j in range(seq_len):
                if not matches[j]:
                    first_mismatch = j
                    break
            
            # Reward proportional to matched prefix length
            rewards[i] = first_mismatch / seq_len
            
        return rewards


class DynamicSampling:
    """Implements dynamic sampling strategy for efficient exploration"""
    
    def __init__(self, config: RPTConfig):
        self.config = config
        self.enabled = False
        self.current_step = 0
        self.sampling_history = []
        
    def update_step(self, step: int):
        """Update current training step and enable if threshold reached"""
        self.current_step = step
        if step >= self.config.dynamic_sampling_start:
            self.enabled = True
            
    def sample_responses(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        num_samples: int = None,
        temperature: float = None
    ) -> List[torch.Tensor]:
        """
        Generate multiple response samples with dynamic sampling
        """
        if num_samples is None:
            num_samples = self.config.num_samples
        if temperature is None:
            temperature = self.config.temperature
            
        samples = []
        
        # Adjust sampling strategy based on training progress
        if self.enabled:
            # Use adaptive temperature based on training progress
            progress_ratio = min(1.0, self.current_step / self.config.training_steps)
            adaptive_temp = temperature * (1.0 - 0.5 * progress_ratio)
        else:
            adaptive_temp = temperature
            
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate sample with temperature scaling
                outputs = model.generate(
                    input_ids,
                    max_length=self.config.max_seq_length,
                    temperature=adaptive_temp,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else 0
                )
                samples.append(outputs)
                
        return samples


class RPTTrainer:
    """Main trainer class for Reinforcement Preference Training with LFM"""
    
    def __init__(
        self,
        model: nn.Module,
        config: RPTConfig,
        tokenizer = None,
        proxy_model: Optional[nn.Module] = None
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize components
        self.token_filter = TokenLevelFilter(config, proxy_model)
        self.reward_system = PrefixMatchingReward(config)
        self.dynamic_sampler = DynamicSampling(config)
        
        # Training state
        self.global_step = 0
        self.best_reward = -float('inf')
        
    def compute_policy_gradient_loss(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy gradient loss
        """
        # Get log probabilities for taken actions
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Expand rewards to match sequence length
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1).expand(-1, action_log_probs.size(1))
        
        # Policy gradient loss
        loss = -(action_log_probs * rewards).mean()
        
        # Add KL penalty if specified
        if self.config.kl_penalty > 0 and old_log_probs is not None:
            kl_div = F.kl_div(log_probs, old_log_probs.exp(), reduction='batchmean')
            loss += self.config.kl_penalty * kl_div
            
        return loss
    
    def train_step(
        self,
        input_batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one training step with RPT
        """
        self.global_step += 1
        self.dynamic_sampler.update_step(self.global_step)
        
        input_ids = input_batch['input_ids']
        labels = input_batch['labels']
        
        # Generate multiple response samples
        response_samples = self.dynamic_sampler.sample_responses(
            self.model, 
            input_ids
        )
        
        # Compute rewards for each sample
        all_rewards = []
        for response in response_samples:
            reward = self.reward_system.compute_reward(
                response[:, input_ids.size(1):],  # Only generated tokens
                labels,
                self.tokenizer
            )
            all_rewards.append(reward)
        
        # Stack rewards
        rewards = torch.stack(all_rewards)
        
        # Select best response for training (or use all with weights)
        best_idx = rewards.argmax(dim=0)
        best_responses = torch.stack(response_samples)[best_idx, torch.arange(len(best_idx))]
        
        # Forward pass with selected responses
        outputs = self.model(
            input_ids=best_responses,
            labels=labels,
            return_dict=True
        )
        
        # Apply token-level filtering if enabled
        if self.token_filter.proxy_model is not None:
            filtered_input, filtered_logits, filtered_labels = self.token_filter.filter_high_entropy_positions(
                best_responses,
                outputs.logits,
                labels
            )
            
            # Compute policy gradient loss on filtered tokens
            pg_loss = self.compute_policy_gradient_loss(
                filtered_logits,
                filtered_labels,
                rewards[best_idx]
            )
        else:
            # Standard policy gradient loss
            pg_loss = self.compute_policy_gradient_loss(
                outputs.logits,
                labels,
                rewards[best_idx]
            )
        
        # Combine with standard LM loss
        total_loss = outputs.loss + pg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'lm_loss': outputs.loss.item(),
            'pg_loss': pg_loss.item(),
            'avg_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'step': self.global_step
        }
        
        return metrics
    
    def evaluate(
        self,
        eval_dataloader,
        num_eval_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model performance with RPT metrics
        """
        self.model.eval()
        total_rewards = []
        
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= num_eval_samples:
                    break
                    
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Generate response
                outputs = self.model.generate(
                    input_ids,
                    max_length=self.config.max_seq_length,
                    temperature=0.1,  # Lower temperature for evaluation
                    do_sample=True
                )
                
                # Compute reward
                reward = self.reward_system.compute_reward(
                    outputs[:, input_ids.size(1):],
                    labels,
                    self.tokenizer
                )
                
                total_rewards.append(reward)
        
        # Aggregate metrics
        all_rewards = torch.cat(total_rewards)
        metrics = {
            'eval_avg_reward': all_rewards.mean().item(),
            'eval_success_rate': (all_rewards == 1.0).float().mean().item(),
            'eval_samples': len(all_rewards)
        }
        
        self.model.train()
        return metrics


def create_rpt_trainer(
    model: nn.Module,
    config: Optional[RPTConfig] = None,
    tokenizer = None,
    proxy_model: Optional[nn.Module] = None
) -> RPTTrainer:
    """
    Factory function to create RPT trainer
    """
    if config is None:
        config = RPTConfig()
        
    trainer = RPTTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        proxy_model=proxy_model
    )
    
    logger.info(f"Created RPT trainer with config: {config}")
    
    return trainer