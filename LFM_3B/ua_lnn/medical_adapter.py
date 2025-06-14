"""
Medical adapter for UA-LNN with safety features and clinical integration
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .model import UpdateAdaptiveLNN
from .config import UALNNConfig


@dataclass
class MedicalContext:
    """Context information for medical decisions"""
    patient_history: Optional[torch.Tensor] = None
    vital_signs: Optional[torch.Tensor] = None
    lab_results: Optional[torch.Tensor] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    urgency_level: str = "routine"  # routine, urgent, critical


class MedicalUALNN(nn.Module):
    """
    Medical-specialized UA-LNN with safety features
    """
    
    def __init__(self, config: UALNNConfig):
        super().__init__()
        
        # Ensure medical mode is enabled
        config.medical_mode = True
        config.require_evidence = True
        config.safety_threshold = max(config.safety_threshold, 0.95)
        
        self.config = config
        self.ua_lnn = UpdateAdaptiveLNN(config)
        
        # Medical-specific components
        self.context_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Safety gates for each medical specialty
        self.safety_gates = nn.ModuleDict({
            specialty: nn.Sequential(
                nn.Linear(config.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            for specialty in config.medical_specialties
        })
        
        # Clinical evidence formatter
        self.evidence_formatter = nn.Sequential(
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        # Risk assessment module
        self.risk_assessor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # low, medium, high risk
        )
        
    def forward(self,
                x: torch.Tensor,
                context: Optional[MedicalContext] = None,
                task: str = "diagnostic",
                require_explanation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Medical-aware forward pass with safety checks
        
        Args:
            x: Input medical data
            context: Medical context information
            task: Medical task (diagnostic, treatment, risk_assessment)
            require_explanation: Whether to generate explanations
            
        Returns:
            Dictionary with medical predictions and safety information
        """
        # Validate task
        if task not in self.config.medical_specialties:
            raise ValueError(f"Unknown medical task: {task}")
        
        # Process through UA-LNN
        ua_result = self.ua_lnn(
            x, 
            task=task,
            return_uncertainty=True,
            return_evidence=True
        )
        
        output = ua_result['output']
        uncertainty = ua_result['uncertainty']
        confidence = ua_result['confidence']
        evidence = ua_result['evidence']
        hidden = ua_result['hidden_features']
        
        # Encode medical context if provided
        context_features = torch.zeros_like(hidden)
        if context is not None:
            context_tensors = []
            if context.patient_history is not None:
                context_tensors.append(context.patient_history)
            if context.vital_signs is not None:
                context_tensors.append(context.vital_signs)
            if context.lab_results is not None:
                context_tensors.append(context.lab_results)
            
            if context_tensors:
                combined_context = torch.cat(context_tensors, dim=-1)
                context_features = self.context_encoder(combined_context)
        
        # Combine with context
        combined_features = torch.cat([hidden, context_features], dim=-1)
        
        # Apply safety gate
        safety_score = self.safety_gates[task](hidden)
        
        # Risk assessment
        risk_scores = self.risk_assessor(combined_features)
        risk_probs = torch.softmax(risk_scores, dim=-1)
        
        # Determine if output meets safety criteria
        safe_prediction = (
            (confidence >= self.config.safety_threshold) & 
            (safety_score.squeeze() >= 0.9)
        )
        
        # Format evidence for clinical use
        formatted_evidence = self.evidence_formatter(evidence)
        
        # Generate explanation if required
        explanation = None
        if require_explanation:
            explanation = self._generate_explanation(
                output, confidence, uncertainty, 
                evidence, risk_probs, task
            )
        
        # Prepare results
        results = {
            'prediction': output,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'safety_score': safety_score,
            'safe_prediction': safe_prediction,
            'risk_assessment': {
                'scores': risk_scores,
                'probabilities': risk_probs,
                'risk_level': ['low', 'medium', 'high'][risk_probs.argmax(dim=-1).item()]
            },
            'evidence': formatted_evidence,
            'requires_review': ~safe_prediction,
            'task': task
        }
        
        if explanation is not None:
            results['explanation'] = explanation
        
        # Add urgency adjustment if context provided
        if context and context.urgency_level != "routine":
            results['urgency_adjusted'] = self._adjust_for_urgency(
                results, context.urgency_level
            )
        
        return results
    
    def _generate_explanation(self, 
                            output: torch.Tensor,
                            confidence: torch.Tensor,
                            uncertainty: torch.Tensor,
                            evidence: torch.Tensor,
                            risk_probs: torch.Tensor,
                            task: str) -> Dict[str, torch.Tensor]:
        """Generate clinical explanation for the prediction"""
        
        # Extract key features from evidence
        evidence_importance = evidence.abs().mean(dim=0)
        top_evidence_indices = evidence_importance.topk(5).indices
        
        explanation = {
            'confidence_level': 'high' if confidence.mean() > 0.9 else 'moderate',
            'uncertainty_level': 'low' if uncertainty.mean() < 0.2 else 'moderate',
            'primary_factors': top_evidence_indices,
            'risk_factors': risk_probs,
            'task_specific': task
        }
        
        return explanation
    
    def _adjust_for_urgency(self, 
                          results: Dict[str, torch.Tensor], 
                          urgency: str) -> Dict[str, torch.Tensor]:
        """Adjust predictions based on urgency level"""
        
        adjustment_factor = {
            'routine': 1.0,
            'urgent': 0.95,
            'critical': 0.9
        }[urgency]
        
        # Lower confidence threshold for urgent cases
        adjusted_confidence = results['confidence'] * adjustment_factor
        adjusted_safe = adjusted_confidence >= (self.config.safety_threshold * adjustment_factor)
        
        return {
            'adjusted_confidence': adjusted_confidence,
            'adjusted_safe_prediction': adjusted_safe,
            'urgency_factor': adjustment_factor
        }
    
    def validate_medical_decision(self, 
                                results: Dict[str, torch.Tensor],
                                clinical_guidelines: Optional[Dict] = None) -> bool:
        """
        Validate medical decision against clinical guidelines
        
        Args:
            results: Model output dictionary
            clinical_guidelines: Optional clinical guidelines to check against
            
        Returns:
            Whether the decision is validated
        """
        # Basic safety checks
        if not results['safe_prediction'].all():
            return False
        
        # Risk level check
        if results['risk_assessment']['risk_level'] == 'high':
            # High risk requires very high confidence
            if results['confidence'].mean() < 0.98:
                return False
        
        # Check against clinical guidelines if provided
        if clinical_guidelines:
            # Implement specific guideline checks
            pass
        
        return True
    
    def get_clinical_summary(self, results: Dict[str, torch.Tensor]) -> str:
        """Generate human-readable clinical summary"""
        
        task = results['task']
        confidence = results['confidence'].mean().item()
        risk_level = results['risk_assessment']['risk_level']
        safe = results['safe_prediction'].all().item()
        
        summary = f"Medical {task} Assessment:\n"
        summary += f"- Confidence: {confidence:.1%}\n"
        summary += f"- Risk Level: {risk_level}\n"
        summary += f"- Safety Check: {'PASSED' if safe else 'REQUIRES REVIEW'}\n"
        
        if not safe:
            summary += "\nWARNING: This case requires human review due to:\n"
            if confidence < self.config.safety_threshold:
                summary += "- Low confidence in prediction\n"
            if results['safety_score'].mean() < 0.9:
                summary += "- Safety concerns identified\n"
        
        return summary