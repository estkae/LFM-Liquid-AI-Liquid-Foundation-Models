#!/usr/bin/env python3
"""
Find and use the best trained Municipal MoE model
"""

import os
from pathlib import Path
import json
import torch


def find_best_models():
    """Find all best_model directories in the current directory tree"""
    
    print("🔍 Searching for best models...\n")
    
    best_models = []
    
    # Search for all best_model directories
    for root, dirs, files in os.walk("."):
        if "best_model" in dirs:
            best_model_path = Path(root) / "best_model"
            
            # Check if it's a valid model directory
            if (best_model_path / "pytorch_model.bin").exists():
                # Try to get training info
                training_info_path = best_model_path.parent / "training_info.json"
                info = {}
                
                if training_info_path.exists():
                    with open(training_info_path, 'r') as f:
                        info = json.load(f)
                
                # Get model size
                model_size = (best_model_path / "pytorch_model.bin").stat().st_size / (1024**3)
                
                best_models.append({
                    "path": str(best_model_path),
                    "size_gb": model_size,
                    "parent": best_model_path.parent.name,
                    "info": info
                })
    
    if not best_models:
        print("❌ No best_model directories found!")
        print("\nHave you trained a model yet? Run:")
        print("  sh train_speed_optimized.sh")
        return None
    
    # Display found models
    print(f"✅ Found {len(best_models)} best model(s):\n")
    
    for i, model in enumerate(best_models, 1):
        print(f"{i}. {model['path']}")
        print(f"   Size: {model['size_gb']:.2f} GB")
        print(f"   Parent: {model['parent']}")
        
        if model['info']:
            if 'final_loss' in model['info']:
                print(f"   Final Loss: {model['info']['final_loss']:.4f}")
            if 'epochs' in model['info']:
                print(f"   Epochs: {model['info']['epochs']}")
            if 'training_data' in model['info']:
                print(f"   Training Data: {model['info']['training_data']}")
        print()
    
    return best_models


def test_best_model(model_path):
    """Test the best model with sample prompts"""
    
    print(f"\n🧪 Testing model: {model_path}\n")
    
    try:
        from municipal_tokenizer_integration import MunicipalTokenizerIntegration
        
        # Load model
        integrator = MunicipalTokenizerIntegration(model_path)
        
        # Test prompts
        test_prompts = [
            "Frage: Was kostet eine Geburtsurkunde?\nAntwort:",
            "Frage: Wo kann ich mich ummelden?\nAntwort:",
            "Frage: Wie beantrage ich einen Personalausweis?\nAntwort:"
        ]
        
        for prompt in test_prompts:
            print(f"📝 {prompt}")
            
            # Generate response
            inputs = integrator.tokenizer(prompt, return_tensors="pt").to(integrator.device)
            generated = integrator.generate_step_by_step(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            
            # Decode
            response = integrator.tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"💬 {response}\n")
            print("-" * 60 + "\n")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        print("\nMake sure you have the tokenizer integration installed.")


def compare_models():
    """Compare different trained models"""
    
    models = find_best_models()
    
    if not models:
        return
    
    print("\n📊 Model Comparison:")
    print("=" * 60)
    
    # Find model with lowest loss
    best_loss = float('inf')
    best_model = None
    
    for model in models:
        if 'final_loss' in model['info']:
            if model['info']['final_loss'] < best_loss:
                best_loss = model['info']['final_loss']
                best_model = model
    
    if best_model:
        print(f"\n🏆 Best model (lowest loss): {best_model['path']}")
        print(f"   Loss: {best_loss:.4f}")
    
    # Find largest model (most training)
    largest_model = max(models, key=lambda x: x['size_gb'])
    print(f"\n📦 Largest model: {largest_model['path']}")
    print(f"   Size: {largest_model['size_gb']:.2f} GB")
    
    return models


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and test best Municipal MoE models")
    parser.add_argument("--test", type=str, help="Test a specific model path")
    parser.add_argument("--compare", action="store_true", help="Compare all found models")
    parser.add_argument("--use-best", action="store_true", help="Test the best model found")
    
    args = parser.parse_args()
    
    if args.test:
        test_best_model(args.test)
    elif args.compare:
        compare_models()
    elif args.use_best:
        models = find_best_models()
        if models:
            # Use the first found model (or the one with lowest loss)
            best = models[0]
            for model in models:
                if 'final_loss' in model['info'] and 'final_loss' in best['info']:
                    if model['info']['final_loss'] < best['info']['final_loss']:
                        best = model
            
            print(f"\n🎯 Using best model: {best['path']}")
            test_best_model(best['path'])
    else:
        # Just find and list models
        find_best_models()
        
        print("\n💡 Usage:")
        print("  python3 find_best_model.py --compare        # Compare all models")
        print("  python3 find_best_model.py --use-best       # Test best model")
        print("  python3 find_best_model.py --test <path>    # Test specific model")


if __name__ == "__main__":
    main()