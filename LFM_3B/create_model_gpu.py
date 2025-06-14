#!/usr/bin/env python3
"""
Optimiertes Script für GPU-Erstellung des LFM-3B Modells
"""

import torch
import sys
import os
import gc

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.config import LFM3BConfig
from LFM_3B.model import LFM3BForCausalLM
from LFM_3B.utils import save_model, print_model_summary
import argparse


def check_gpu():
    """Überprüft GPU-Verfügbarkeit und zeigt Informationen"""
    if not torch.cuda.is_available():
        print("⚠️  Keine GPU gefunden!")
        return False
    
    print("GPU-Informationen:")
    print(f"- GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"- CUDA Version: {torch.version.cuda}")
    
    # Aktueller Speicher
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"- Bereits belegt: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
    
    return True


def create_model_on_gpu(args):
    """Erstellt das Modell direkt auf der GPU"""
    
    # GPU Check
    if args.cuda:
        if not check_gpu():
            print("Fahre ohne GPU fort...")
            args.cuda = False
        else:
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
    
    # Konfiguration
    print("\n📋 Erstelle Modell-Konfiguration...")
    config = LFM3BConfig(
        medical_mode=args.medical_mode,
        use_liquid_layers=args.use_liquid,
        num_hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_experts=args.num_experts,
        torch_dtype="float16" if args.fp16 else "float32",
    )
    
    print(f"\nModell-Parameter:")
    print(f"- Geschätzte Größe: {config.total_params_in_billions:.2f}B Parameter")
    print(f"- Precision: {'float16' if args.fp16 else 'float32'}")
    
    # Device festlegen
    device = torch.device("cuda" if args.cuda else "cpu")
    
    try:
        # Modell initialisieren
        print(f"\n🔨 Initialisiere LFM-3B Modell auf {device}...")
        
        if args.cuda:
            # Direkt auf GPU erstellen
            with torch.cuda.device(0):
                model = LFM3BForCausalLM(config)
                if args.fp16:
                    model = model.half()
                model = model.to(device)
        else:
            model = LFM3BForCausalLM(config)
        
        # GPU Memory nach Modell-Erstellung
        if args.cuda:
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"✅ Modell erstellt! GPU-Speicher verwendet: {allocated:.1f} GB")
        
        # Modell-Zusammenfassung
        print_model_summary(model)
        
        # Test Generation
        if args.test_generation:
            print("\n🧪 Teste Textgenerierung...")
            test_model_generation(model, config, device)
        
        # Speichern
        if args.save_path:
            print(f"\n💾 Speichere Modell nach: {args.save_path}")
            # Modell auf CPU für's Speichern
            if args.cuda:
                model = model.cpu()
            save_model(model, args.save_path, config)
            print("✅ Modell erfolgreich gespeichert!")
        
        return model, config
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU-Speicher reicht nicht aus!")
            print("Versuche:")
            print("  1. --fp16 für halbe Precision")
            print("  2. --num-layers 16 für weniger Layer")
            print("  3. --hidden-size 2048 für kleinere Hidden Size")
            raise
        else:
            raise e
    finally:
        if args.cuda:
            torch.cuda.empty_cache()


def test_model_generation(model, config, device):
    """Testet die Modell-Generierung"""
    # Test Input
    test_prompt = torch.randint(1, min(1000, config.vocab_size), (1, 20), device=device)
    
    print(f"Input shape: {test_prompt.shape}")
    
    with torch.no_grad():
        # Teste Forward Pass
        output = model(test_prompt)
        print(f"✅ Forward pass erfolgreich! Logits shape: {output.logits.shape}")
        
        # Teste Generation
        generated = model.generate(
            input_ids=test_prompt,
            max_length=50,
            temperature=0.8,
            do_sample=True,
            top_k=50,
        )
        
        print(f"✅ Generation erfolgreich! Generated shape: {generated.shape}")
        print(f"   Generierte {generated.shape[1] - test_prompt.shape[1]} neue Tokens")
    
    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   Max GPU-Speicher während Generation: {max_memory:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Erstelle LFM-3B Modell optimiert für GPU")
    
    # Modell-Konfiguration
    parser.add_argument("--hidden-size", type=int, default=3072,
                        help="Hidden size (default: 3072)")
    parser.add_argument("--num-layers", type=int, default=20,
                        help="Anzahl Layer (default: 20)")
    parser.add_argument("--num-heads", type=int, default=24,
                        help="Anzahl Attention Heads (default: 24)")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Anzahl Experten (default: 8)")
    
    # Features
    parser.add_argument("--use-liquid", action="store_true", default=True,
                        help="Liquid Neural Networks verwenden")
    parser.add_argument("--no-liquid", dest="use_liquid", action="store_false")
    parser.add_argument("--medical-mode", action="store_true",
                        help="Medical Mode aktivieren")
    
    # GPU & Precision
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="GPU verwenden (default: True)")
    parser.add_argument("--cpu", dest="cuda", action="store_false",
                        help="Nur CPU verwenden")
    parser.add_argument("--fp16", action="store_true",
                        help="Float16 (halbe Precision) verwenden")
    
    # Speichern & Testen
    parser.add_argument("--save-path", type=str,
                        help="Pfad zum Speichern")
    parser.add_argument("--test-generation", action="store_true", default=True,
                        help="Generation testen")
    
    args = parser.parse_args()
    
    print("🚀 LFM-3B GPU-optimierte Modell-Erstellung")
    print("=" * 50)
    
    # Modell erstellen
    model, config = create_model_on_gpu(args)
    
    print("\n✅ Fertig!")


if __name__ == "__main__":
    main()