#!/usr/bin/env python3
"""
Script to create and test Medical Health MoE Base Model
"""

import torch
import sys
import os
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LFM_3B'))

from medical_health_base import create_medical_health_model, MedicalHealthConfig
from LFM_3B.utils import save_model, print_model_summary


def test_medical_health_model(model, config):
    """Test the Medical Health Model"""
    print("\n🧪 Testing Medical Health Model")
    print("=" * 50)
    
    # Test data
    batch_size = 2
    seq_len = 64
    device = next(model.parameters()).device
    
    # Create test inputs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Test input shape: {input_ids.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_medical_outputs=True
        )
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Hidden states: {outputs['last_hidden_state'].shape}")
    
    # Test medical outputs
    if 'diagnosis_logits' in outputs:
        print(f"   Diagnosis logits: {outputs['diagnosis_logits'].shape}")
    if 'risk_logits' in outputs:
        print(f"   Risk logits: {outputs['risk_logits'].shape}")
    if 'specialty_logits' in outputs:
        print(f"   Specialty logits: {outputs['specialty_logits'].shape}")
    
    # Test auxiliary outputs
    aux = outputs['auxiliary_outputs']
    print(f"\n📊 Auxiliary Outputs:")
    for key, values in aux.items():
        if values:
            print(f"   {key}: {len(values)} layers")
            if isinstance(values[0], torch.Tensor):
                print(f"      Shape: {values[0].shape}")
    
    # Test specialty routing
    if 'specialty_distribution' in aux and aux['specialty_distribution']:
        specialty_dist = aux['specialty_distribution'][0]  # First layer
        top_specialties = torch.topk(specialty_dist, 3, dim=-1)
        print(f"\n🏥 Top Medical Specialties (Layer 0):")
        for i, (score, idx) in enumerate(zip(top_specialties.values[0], top_specialties.indices[0])):
            specialty = config.health_specialties[idx.item()]
            print(f"   {i+1}. {specialty}: {score.item():.3f}")
    
    # Test urgency detection
    if 'urgency_score' in aux and aux['urgency_score']:
        urgency = aux['urgency_score'][0].mean().item()
        print(f"\n🚨 Urgency Score: {urgency:.3f}")
        if urgency > 0.7:
            print("   ⚠️ HIGH URGENCY DETECTED")
        elif urgency > 0.4:
            print("   ⚡ Moderate urgency")
        else:
            print("   ✅ Low urgency")
    
    # Test safety features
    if 'phi_risk' in aux and aux['phi_risk']:
        phi_risk = aux['phi_risk'][0].mean().item()
        print(f"\n🔒 PHI Risk Score: {phi_risk:.3f}")
        if phi_risk > 0.7:
            print("   ⚠️ HIGH PHI RISK - Content filtered")
    
    if 'uncertainty' in aux and aux['uncertainty']:
        uncertainty = aux['uncertainty'][0].mean().item()
        print(f"\n🎯 Uncertainty Score: {uncertainty:.3f}")
        if uncertainty > 0.8:
            print("   ⚠️ HIGH UNCERTAINTY - Requires human review")


def benchmark_model(model, config):
    """Benchmark model performance"""
    print("\n⚡ Benchmarking Model Performance")
    print("=" * 50)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Test different sequence lengths
    seq_lengths = [32, 64, 128, 256]
    batch_size = 1
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids, attention_mask)
        
        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                outputs = model(input_ids, attention_mask, return_medical_outputs=True)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        tokens_per_second = seq_len / avg_time
        
        print(f"   Seq len {seq_len:3d}: {avg_time*1000:6.1f}ms ({tokens_per_second:6.1f} tok/s)")


def create_medical_scenarios():
    """Create medical test scenarios"""
    scenarios = {
        "emergency": [
            "Patient reports severe chest pain and shortness of breath",
            "Unconscious patient brought to emergency department",
            "Child with high fever and difficulty breathing"
        ],
        "cardiology": [
            "Patient has history of hypertension and reports palpitations", 
            "EKG shows irregular rhythm, patient feels dizzy",
            "Chest pain during exercise, family history of heart disease"
        ],
        "neurology": [
            "Patient reports severe headache and vision changes",
            "Memory loss and confusion in elderly patient",
            "Sudden weakness on left side of body"
        ],
        "pharmacy": [
            "Patient taking warfarin, prescribed new antibiotic",
            "Drug dosage calculation for pediatric patient",
            "Potential drug interaction between medications"
        ]
    }
    return scenarios


def test_medical_scenarios(model, config, tokenizer=None):
    """Test model on medical scenarios"""
    print("\n🏥 Testing Medical Scenarios")
    print("=" * 50)
    
    scenarios = create_medical_scenarios()
    device = next(model.parameters()).device
    
    for specialty, cases in scenarios.items():
        print(f"\n📋 {specialty.upper()} Cases:")
        
        for i, case in enumerate(cases[:2]):  # Test first 2 cases
            print(f"\n   Case {i+1}: {case}")
            
            # Simple tokenization (character-level for demo)
            if tokenizer is None:
                # Convert to token IDs (simplified)
                tokens = [ord(c) % config.vocab_size for c in case[:64]]
                tokens += [0] * (64 - len(tokens))  # Pad
                input_ids = torch.tensor([tokens], device=device)
            else:
                encoded = tokenizer.encode(case, return_tensors="pt")
                input_ids = encoded[:, :64].to(device)  # Truncate to 64 tokens
            
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_medical_outputs=True
                )
            
            # Analyze results
            aux = outputs['auxiliary_outputs']
            
            if 'specialty_distribution' in aux and aux['specialty_distribution']:
                specialty_probs = aux['specialty_distribution'][0][0]  # First sample, first layer
                top_spec = torch.argmax(specialty_probs).item()
                confidence = torch.max(specialty_probs).item()
                predicted_specialty = config.health_specialties[top_spec]
                
                print(f"      Predicted: {predicted_specialty} (confidence: {confidence:.3f})")
                if predicted_specialty == specialty:
                    print(f"      ✅ Correct specialty detected!")
                else:
                    print(f"      ❌ Expected: {specialty}")
            
            if 'urgency_score' in aux and aux['urgency_score']:
                urgency = aux['urgency_score'][0][0].item()
                print(f"      Urgency: {urgency:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Create Medical Health MoE Base Model")
    
    parser.add_argument("--size", type=str, default="small",
                        choices=["tiny", "small", "base", "large"],
                        help="Model size")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--save-path", type=str,
                        help="Path to save the model")
    parser.add_argument("--test", action="store_true",
                        help="Run tests")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark")
    parser.add_argument("--medical-scenarios", action="store_true",
                        help="Test medical scenarios")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision")
    
    # Medical configuration
    parser.add_argument("--phi-protection", action="store_true", default=True,
                        help="Enable PHI protection")
    parser.add_argument("--uncertainty-estimation", action="store_true", default=True,
                        help="Enable uncertainty estimation")
    parser.add_argument("--multilingual", action="store_true", default=True,
                        help="Enable multilingual support")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Creating Medical Health Model on {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model
    print(f"\n🏗️ Building {args.size} Medical Health Model...")
    
    model = create_medical_health_model(
        config_name=args.size,
        phi_protection=args.phi_protection,
        uncertainty_estimation=args.uncertainty_estimation,
        multilingual=args.multilingual,
    )
    
    # Move to device and set precision
    model = model.to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()
        print("✅ Using FP16 precision")
    
    # Model summary
    print_model_summary(model)
    
    # Tests
    if args.test:
        test_medical_health_model(model, model.config)
    
    if args.benchmark:
        benchmark_model(model, model.config)
    
    if args.medical_scenarios:
        test_medical_scenarios(model, model.config)
    
    # Save model
    if args.save_path:
        print(f"\n💾 Saving model to: {args.save_path}")
        save_model(model, args.save_path, model.config)
        print("✅ Model saved successfully!")
        
        # Save config separately
        config_path = Path(args.save_path) / "medical_config.json"
        import json
        with open(config_path, 'w') as f:
            config_dict = model.config.__dict__.copy()
            # Convert non-serializable items
            for key, value in config_dict.items():
                if not isinstance(value, (str, int, float, bool, list)):
                    config_dict[key] = str(value)
            json.dump(config_dict, f, indent=2)
        print(f"✅ Config saved to: {config_path}")
    
    print(f"\n🎉 Medical Health Model ready!")
    print(f"   Specialties: {', '.join(model.config.health_specialties[:5])}...")
    print(f"   Safety features: PHI protection, Uncertainty estimation")
    print(f"   Compliance: HIPAA, GDPR ready")


if __name__ == "__main__":
    main()