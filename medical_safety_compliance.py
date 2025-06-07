#!/usr/bin/env python3
"""
Medical Safety and Compliance Module
Implements HIPAA compliance, safety checks, and medical validation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import re
import json
from datetime import datetime
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Medical risk levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceLevel(Enum):
    """Compliance requirement levels"""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    FDA = "fda"
    STANDARD = "standard"


@dataclass
class SafetyCheck:
    """Result of a safety check"""
    passed: bool
    risk_level: RiskLevel
    confidence: float
    issues: List[str]
    recommendations: List[str]


class MedicalValidator:
    """Validates medical content for accuracy and safety"""
    
    def __init__(self):
        self.drug_database = self._load_drug_database()
        self.contraindications = self._load_contraindications()
        self.dosage_ranges = self._load_dosage_ranges()
        
    def _load_drug_database(self) -> Dict[str, Dict]:
        """Load database of approved drugs"""
        # In production, this would connect to a real drug database
        return {
            "aspirin": {
                "generic": "acetylsalicylic acid",
                "classes": ["nsaid", "antiplatelet"],
                "max_daily_dose": 4000  # mg
            },
            "metformin": {
                "generic": "metformin hydrochloride",
                "classes": ["antidiabetic", "biguanide"],
                "max_daily_dose": 2550  # mg
            },
            # Add more drugs as needed
        }
    
    def _load_contraindications(self) -> Dict[str, List[str]]:
        """Load drug contraindications"""
        return {
            "aspirin": ["bleeding disorders", "aspirin allergy", "children with viral infections"],
            "metformin": ["severe kidney disease", "metabolic acidosis"],
        }
    
    def _load_dosage_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load safe dosage ranges"""
        return {
            "aspirin": {
                "adult": (81, 4000),  # mg per day
                "pediatric": (10, 90)  # mg/kg per day
            },
            "metformin": {
                "adult": (500, 2550),
                "pediatric": (0, 0)  # Not recommended for children
            }
        }
    
    def validate_drug_mention(self, text: str) -> SafetyCheck:
        """Validate drug mentions in text"""
        issues = []
        recommendations = []
        risk_level = RiskLevel.LOW
        
        # Extract drug mentions
        drug_pattern = r'\b(?:' + '|'.join(self.drug_database.keys()) + r')\b'
        drugs_mentioned = re.findall(drug_pattern, text.lower())
        
        # Extract dosages
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)'
        dosages = re.findall(dosage_pattern, text.lower())
        
        # Validate each drug mention
        for drug in drugs_mentioned:
            # Check for contraindications mentioned
            if drug in self.contraindications:
                for contraindication in self.contraindications[drug]:
                    if contraindication.lower() in text.lower():
                        issues.append(f"Potential contraindication: {drug} with {contraindication}")
                        risk_level = RiskLevel.HIGH
            
            # Check dosages
            if drug in self.dosage_ranges and dosages:
                for dose_value, unit in dosages:
                    dose_mg = self._convert_to_mg(float(dose_value), unit)
                    safe_range = self.dosage_ranges[drug].get("adult", (0, float('inf')))
                    
                    if dose_mg > safe_range[1]:
                        issues.append(f"Dosage exceeds maximum: {dose_value}{unit} for {drug}")
                        risk_level = RiskLevel.CRITICAL
                        recommendations.append(f"Review dosage for {drug}. Maximum recommended: {safe_range[1]}mg")
        
        passed = len(issues) == 0
        confidence = 0.9 if passed else 0.6
        
        return SafetyCheck(
            passed=passed,
            risk_level=risk_level,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations
        )
    
    def _convert_to_mg(self, value: float, unit: str) -> float:
        """Convert dosage to milligrams"""
        conversions = {
            'mcg': 0.001,
            'mg': 1.0,
            'g': 1000.0
        }
        return value * conversions.get(unit, 1.0)
    
    def validate_medical_advice(self, text: str) -> SafetyCheck:
        """Validate medical advice for safety"""
        issues = []
        recommendations = []
        risk_level = RiskLevel.LOW
        
        # Check for dangerous advice patterns
        dangerous_patterns = [
            (r'stop taking .* medication', "Advising to stop medication without consultation"),
            (r'ignore .* symptoms', "Advising to ignore symptoms"),
            (r'don\'t (?:see|consult|visit) .* doctor', "Discouraging medical consultation"),
            (r'cure .* disease', "Claiming to cure diseases"),
            (r'guaranteed .* treatment', "Making guaranteed treatment claims")
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(description)
                risk_level = RiskLevel.HIGH
                recommendations.append("Include disclaimer about consulting healthcare professionals")
        
        # Check for emergency symptoms
        emergency_patterns = [
            'chest pain', 'difficulty breathing', 'severe bleeding',
            'loss of consciousness', 'stroke symptoms', 'heart attack'
        ]
        
        for pattern in emergency_patterns:
            if pattern in text.lower():
                if 'call 911' not in text.lower() and 'emergency' not in text.lower():
                    issues.append(f"Emergency symptom '{pattern}' mentioned without emergency guidance")
                    risk_level = RiskLevel.CRITICAL
                    recommendations.append("Add emergency contact information (911 or local emergency number)")
        
        passed = len(issues) == 0
        confidence = 0.85 if passed else 0.5
        
        return SafetyCheck(
            passed=passed,
            risk_level=risk_level,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations
        )


class HIPAACompliance:
    """Ensures HIPAA compliance for medical data"""
    
    def __init__(self):
        self.phi_patterns = self._create_phi_patterns()
        self.audit_log = []
    
    def _create_phi_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Create patterns for detecting PHI"""
        return [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "SSN"),
            (re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+, (?:MD|DO|RN|NP)\b'), "Provider Name"),
            (re.compile(r'\bMRN:?\s*\d+\b', re.IGNORECASE), "Medical Record Number"),
            (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), "Date"),
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "Email"),
            (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), "Phone Number"),
            (re.compile(r'\b\d{5}(?:-\d{4})?\b'), "ZIP Code"),
            (re.compile(r'(?:policy|member|group)\s*#?\s*\d+', re.IGNORECASE), "Insurance ID"),
        ]
    
    def check_phi(self, text: str) -> Dict[str, Any]:
        """Check text for PHI"""
        phi_found = []
        
        for pattern, phi_type in self.phi_patterns:
            matches = pattern.findall(text)
            if matches:
                phi_found.append({
                    'type': phi_type,
                    'count': len(matches),
                    'pattern': pattern.pattern
                })
        
        return {
            'contains_phi': len(phi_found) > 0,
            'phi_types': phi_found,
            'risk_score': min(len(phi_found) * 0.2, 1.0)
        }
    
    def deidentify(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """Remove PHI from text"""
        deidentified = text
        removed_phi = {}
        
        for pattern, phi_type in self.phi_patterns:
            matches = pattern.findall(deidentified)
            if matches:
                removed_phi[phi_type] = matches
                # Replace with placeholder
                deidentified = pattern.sub(f'[{phi_type.upper()}]', deidentified)
        
        # Log the deidentification
        self._log_deidentification(removed_phi)
        
        return deidentified, removed_phi
    
    def _log_deidentification(self, removed_phi: Dict[str, List[str]]):
        """Log deidentification for audit trail"""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'deidentification',
            'phi_types_removed': list(removed_phi.keys()),
            'hash': hashlib.sha256(str(removed_phi).encode()).hexdigest()
        })
    
    def create_audit_report(self) -> Dict[str, Any]:
        """Create HIPAA compliance audit report"""
        return {
            'total_operations': len(self.audit_log),
            'deidentifications': sum(1 for log in self.audit_log if log['action'] == 'deidentification'),
            'audit_trail': self.audit_log[-100:]  # Last 100 entries
        }


class MedicalSafetyGate(nn.Module):
    """Neural safety gate for medical outputs"""
    
    def __init__(self, hidden_dim: int, num_risk_levels: int = 4):
        super().__init__()
        
        # Risk assessment layers
        self.risk_assessor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_risk_levels)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Evidence requirement checker
        self.evidence_checker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Assess safety of model outputs
        
        Returns:
            Dictionary with risk_scores, confidence, evidence_score
        """
        # Pool hidden states
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        
        # Assess risk
        risk_logits = self.risk_assessor(pooled)
        risk_probs = torch.softmax(risk_logits, dim=-1)
        
        # Estimate confidence
        confidence = self.confidence_estimator(pooled)
        
        # Check evidence requirement
        evidence_score = self.evidence_checker(pooled)
        
        return {
            'risk_probabilities': risk_probs,
            'risk_level': torch.argmax(risk_probs, dim=-1),
            'confidence': confidence,
            'evidence_score': evidence_score,
            'safety_score': confidence * (1 - risk_probs[:, -1:])  # High risk is last class
        }


class MedicalComplianceWrapper:
    """Wrapper to ensure medical compliance for model outputs"""
    
    def __init__(self, model, compliance_level: ComplianceLevel = ComplianceLevel.HIPAA):
        self.model = model
        self.compliance_level = compliance_level
        self.validator = MedicalValidator()
        self.hipaa = HIPAACompliance()
        self.safety_gate = MedicalSafetyGate(
            hidden_dim=model.config.dim if hasattr(model.config, 'dim') else 768
        )
    
    def generate_safe(
        self,
        input_text: str,
        max_length: int = 512,
        require_evidence: bool = True,
        min_confidence: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with safety checks"""
        
        # Check input for PHI
        phi_check = self.hipaa.check_phi(input_text)
        if phi_check['contains_phi']:
            input_text, removed_phi = self.hipaa.deidentify(input_text)
        
        # Generate with model
        outputs = self.model.generate(
            input_text,
            max_length=max_length,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **kwargs
        )
        
        generated_text = outputs.sequences[0]
        
        # Safety assessment
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1][-1]  # Last layer, last token
            safety_assessment = self.safety_gate(last_hidden)
        else:
            safety_assessment = {
                'confidence': torch.tensor(0.5),
                'risk_level': torch.tensor(1),
                'safety_score': torch.tensor(0.5)
            }
        
        # Validate generated content
        validation_result = self.validator.validate_medical_advice(generated_text)
        
        # Check confidence threshold
        confidence = safety_assessment['confidence'].item()
        if confidence < min_confidence:
            generated_text += f"\n\n⚠️ Low confidence ({confidence:.2f}). Please verify with medical professionals."
        
        # Add evidence requirement notice if needed
        if require_evidence and safety_assessment.get('evidence_score', 0).item() < 0.5:
            generated_text += "\n\n📋 Note: This response may lack sufficient evidence. Consult medical literature."
        
        return {
            'text': generated_text,
            'safety_check': validation_result,
            'confidence': confidence,
            'risk_level': RiskLevel(safety_assessment['risk_level'].item()),
            'phi_removed': phi_check['contains_phi'],
            'compliance_level': self.compliance_level.value
        }
    
    def create_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        return {
            'hipaa_audit': self.hipaa.create_audit_report(),
            'compliance_level': self.compliance_level.value,
            'safety_features': {
                'phi_detection': True,
                'deidentification': True,
                'risk_assessment': True,
                'confidence_scoring': True,
                'evidence_checking': True
            },
            'timestamp': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Example validation
    validator = MedicalValidator()
    
    # Test drug validation
    text1 = "Patient prescribed aspirin 5000mg daily for headache"
    result1 = validator.validate_drug_mention(text1)
    print(f"Drug validation: {result1}")
    
    # Test medical advice validation  
    text2 = "If you have chest pain, wait a few days to see if it improves"
    result2 = validator.validate_medical_advice(text2)
    print(f"Advice validation: {result2}")
    
    # Test HIPAA compliance
    hipaa = HIPAACompliance()
    text3 = "John Doe (MRN: 12345) was seen on 01/15/2024. Contact: john@email.com"
    phi_check = hipaa.check_phi(text3)
    print(f"PHI check: {phi_check}")
    
    deidentified, removed = hipaa.deidentify(text3)
    print(f"Deidentified: {deidentified}")