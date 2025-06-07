#!/usr/bin/env python3
"""
Fine-tuning Pipeline for Medical LFM
Trains the medical-specialized model with safety and compliance features
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from lfm.medical_moe import MedicalLFM, MedicalConfig, create_medical_model
from prepare_medical_data import MedicalDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataset(Dataset):
    """Dataset for medical text with safety labels"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        if 'instruction' in item and 'output' in item:
            # Instruction-following format
            text = f"### Instruction:\n{item['instruction']}\n\n"
            if 'input' in item and item['input']:
                text += f"### Input:\n{item['input']}\n\n"
            text += f"### Response:\n{item['output']}"
        else:
            # Standard text format
            text = item['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Add labels for language modeling
        encoding['labels'] = encoding['input_ids'].clone()
        
        # Add metadata
        encoding['category'] = item.get('category', 'unknown')
        encoding['task_type'] = item.get('task_type', 'general')
        
        return {k: v.squeeze() if isinstance(v, torch.Tensor) else v 
                for k, v in encoding.items()}


class MedicalTrainer(Trainer):
    """Custom trainer with medical safety checks"""
    
    def __init__(self, *args, safety_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_config = safety_config or {}
        self.safety_violations = []
        self.confidence_history = []
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with medical safety penalties"""
        # Extract labels
        labels = inputs.pop("labels", None)
        
        # Forward pass
        outputs = model(**inputs, return_medical_outputs=True)
        
        # Get base language modeling loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Compute standard loss
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Add medical-specific losses if available
        total_loss = lm_loss
        
        if hasattr(outputs, 'aux_outputs') or (isinstance(outputs, tuple) and len(outputs) > 1):
            aux_outputs = outputs.aux_outputs if hasattr(outputs, 'aux_outputs') else outputs[1]
            
            # Confidence regularization
            if 'average_confidence' in aux_outputs:
                confidence = aux_outputs['average_confidence']
                self.confidence_history.append(confidence.mean().item())
                
                # Penalize low confidence
                min_confidence = self.safety_config.get('min_confidence', 0.7)
                confidence_penalty = torch.relu(min_confidence - confidence).mean()
                total_loss += 0.1 * confidence_penalty
            
            # Safety score monitoring
            if 'safety_score' in aux_outputs:
                safety_score = aux_outputs['safety_score']
                
                # Log safety violations
                violations = (safety_score < 0.5).sum().item()
                if violations > 0:
                    self.safety_violations.append({
                        'step': self.state.global_step,
                        'violations': violations,
                        'batch_size': safety_score.size(0)
                    })
            
            # Urgency-based loss weighting
            if 'urgency_score' in aux_outputs:
                urgency = aux_outputs['urgency_score']
                # Higher loss weight for urgent cases
                urgency_weight = 1.0 + 0.5 * urgency.mean()
                total_loss = total_loss * urgency_weight
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Log with additional medical metrics"""
        # Add custom metrics
        if self.confidence_history:
            logs['medical/avg_confidence'] = np.mean(self.confidence_history[-100:])
        
        if self.safety_violations:
            recent_violations = [v['violations'] for v in self.safety_violations[-10:]]
            logs['medical/safety_violations'] = sum(recent_violations)
        
        super().log(logs)


class MedicalSafetyCallback:
    """Callback for monitoring medical safety during training"""
    
    def __init__(self, threshold_confidence: float = 0.7, max_violations: int = 100):
        self.threshold_confidence = threshold_confidence
        self.max_violations = max_violations
        self.violation_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check safety metrics"""
        if logs:
            # Check confidence
            avg_confidence = logs.get('medical/avg_confidence', 1.0)
            if avg_confidence < self.threshold_confidence:
                logger.warning(f"Low average confidence: {avg_confidence:.3f}")
            
            # Check violations
            violations = logs.get('medical/safety_violations', 0)
            self.violation_count += violations
            
            if self.violation_count > self.max_violations:
                logger.error(f"Too many safety violations: {self.violation_count}")
                control.should_training_stop = True
        
        return control


def evaluate_medical_model(model, eval_dataset, tokenizer, device):
    """Evaluate model on medical benchmarks"""
    model.eval()
    results = {
        'accuracy': [],
        'confidence': [],
        'safety_scores': [],
        'specialty_accuracy': {}
    }
    
    with torch.no_grad():
        for item in tqdm(eval_dataset, desc="Evaluating"):
            inputs = tokenizer(
                item['text'],
                return_tensors='pt',
                truncation=True,
                max_length=2048
            ).to(device)
            
            outputs = model(**inputs, return_medical_outputs=True)
            
            # Collect metrics
            if hasattr(outputs, 'aux_outputs'):
                aux = outputs.aux_outputs
                if 'average_confidence' in aux:
                    results['confidence'].append(aux['average_confidence'].cpu().item())
                if 'safety_score' in aux:
                    results['safety_scores'].append(aux['safety_score'].cpu().item())
    
    # Compute aggregate metrics
    eval_metrics = {
        'avg_confidence': np.mean(results['confidence']) if results['confidence'] else 0,
        'avg_safety_score': np.mean(results['safety_scores']) if results['safety_scores'] else 0,
        'high_confidence_ratio': np.mean([c > 0.8 for c in results['confidence']]) if results['confidence'] else 0
    }
    
    return eval_metrics


def main():
    """Main training pipeline"""
    # Configuration
    model_name = "liquid/lfm-3b"
    output_dir = "./medical_lfm_model"
    data_dir = "./data/medical_dataset"
    
    # Initialize wandb (optional)
    # wandb.init(project="medical-lfm", name=f"medical-lfm-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create medical model
    logger.info("Creating medical model...")
    model = create_medical_model(
        base_model_name=model_name,
        num_medical_experts=12,
        enable_safety_checks=True,
        confidence_threshold=0.85,
        require_evidence=True
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = MedicalDataset(f"{data_dir}/train.jsonl", tokenizer)
    val_dataset = MedicalDataset(f"{data_dir}/validation.jsonl", tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=["tensorboard"],  # Add "wandb" if using wandb
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Safety configuration
    safety_config = {
        'min_confidence': 0.7,
        'max_safety_violations': 100
    }
    
    # Initialize trainer
    trainer = MedicalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        safety_config=safety_config,
        callbacks=[MedicalSafetyCallback()],
    )
    
    # Training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save training metrics
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    # Evaluate on test set
    if os.path.exists(f"{data_dir}/test.jsonl"):
        logger.info("Evaluating on test set...")
        test_dataset = MedicalDataset(f"{data_dir}/test.jsonl", tokenizer)
        eval_metrics = evaluate_medical_model(
            model, 
            test_dataset, 
            tokenizer, 
            device=next(model.parameters()).device
        )
        
        with open(f"{output_dir}/eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)
        
        logger.info(f"Evaluation metrics: {eval_metrics}")
    
    # Save model configuration
    config_dict = model.config.__dict__ if hasattr(model.config, '__dict__') else {}
    with open(f"{output_dir}/medical_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("Training completed successfully!")
    
    # Print safety summary
    if hasattr(trainer, 'safety_violations'):
        total_violations = sum(v['violations'] for v in trainer.safety_violations)
        logger.info(f"Total safety violations during training: {total_violations}")
    
    # Create model card
    create_model_card(output_dir, train_result.metrics, eval_metrics if 'eval_metrics' in locals() else {})


def create_model_card(output_dir: str, train_metrics: Dict, eval_metrics: Dict):
    """Create a model card with important information"""
    model_card = f"""# Medical LFM Model Card

## Model Description
This is a medical-specialized Language Foundation Model (LFM) fine-tuned for healthcare applications.

## Intended Use
- Medical text generation and understanding
- Clinical decision support (with human oversight)
- Medical information extraction
- Healthcare documentation assistance

## Limitations and Biases
- This model should NOT be used for direct medical diagnosis or treatment
- Always requires human medical professional oversight
- May contain biases present in training data
- Performance may vary across different medical specialties

## Training Information
- Base Model: liquid/lfm-3b
- Training Loss: {train_metrics.get('train_loss', 'N/A')}
- Evaluation Loss: {train_metrics.get('eval_loss', 'N/A')}
- Training Time: {train_metrics.get('train_runtime', 'N/A')} seconds

## Evaluation Metrics
- Average Confidence: {eval_metrics.get('avg_confidence', 'N/A')}
- Average Safety Score: {eval_metrics.get('avg_safety_score', 'N/A')}
- High Confidence Ratio: {eval_metrics.get('high_confidence_ratio', 'N/A')}

## Safety Features
- Confidence estimation for all outputs
- Safety scoring for high-risk content
- Evidence requirement for medical claims
- Urgency detection for critical cases

## Ethical Considerations
- Patient privacy: All training data was de-identified
- Bias mitigation: Model includes diverse medical specialties
- Transparency: Confidence scores provided for all outputs
- Human oversight: Designed to assist, not replace, medical professionals

## Citation
If you use this model, please cite:
```
@software{{medical_lfm,
  title = {{Medical-Specialized Language Foundation Model}},
  year = {{2024}},
  publisher = {{Your Organization}}
}}
```
"""
    
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(model_card)


if __name__ == "__main__":
    main()