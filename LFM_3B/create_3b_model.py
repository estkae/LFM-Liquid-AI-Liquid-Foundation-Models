#!/usr/bin/env python3
"""
Erstellt ein echtes 3B Parameter LFM Modell
"""

import torch
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.config import LFM3BConfig
from LFM_3B.model import LFM3BForCausalLM
from LFM_3B.utils import save_model, print_model_summary
import argparse


def create_3b_config():
    """Erstellt eine Konfiguration für genau 3B Parameter"""
    
    # Angepasste Werte für ~3B Parameter
    config = LFM3BConfig(
        # Reduzierte Dimensionen
        hidden_size=2048,           # Kleiner als 3072
        num_hidden_layers=24,       # Mehr Layer, aber kleiner
        num_attention_heads=16,     # Weniger Heads
        
        # Kleineres FFN
        intermediate_size=5120,     # Kleiner als 8192
        
        # Weniger Experten für Speichereffizienz
        num_experts=4,              # Nur 4 statt 8 Experten
        num_experts_per_token=2,
        
        # Kleineres Vokabular
        vocab_size=50257,           # Wie GPT-2, viel kleiner als 128k
        
        # Liquid parameters bleiben
        liquid_num_reservoirs=2,    # Reduziert von 3
        liquid_reservoir_size=256,  # Reduziert von 512
        
        # Embeddings nicht teilen um Parameter zu erhöhen
        tie_word_embeddings=False,
    )
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Erstelle echtes 3B LFM Modell")
    
    parser.add_argument("--medical-mode", action="store_true",
                        help="Medical Mode aktivieren")
    parser.add_argument("--save-path", type=str, default="./lfm_3b_model",
                        help="Speicherpfad")
    parser.add_argument("--fp16", action="store_true",
                        help="Float16 verwenden")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="GPU verwenden")
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    
    args = parser.parse_args()
    
    print("🚀 Erstelle echtes LFM-3B Modell")
    print("=" * 50)
    
    # GPU Info
    if args.cuda and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Konfiguration
    config = create_3b_config()
    config.medical_mode = args.medical_mode
    if args.fp16:
        config.torch_dtype = "float16"
    
    # Parameter berechnen
    print(f"\nModell-Konfiguration:")
    print(f"- Hidden size: {config.hidden_size}")
    print(f"- Layers: {config.num_hidden_layers}")
    print(f"- Attention heads: {config.num_attention_heads}")
    print(f"- FFN size: {config.intermediate_size}")
    print(f"- Experts: {config.num_experts}")
    print(f"- Vocabulary: {config.vocab_size:,}")
    print(f"- Geschätzte Parameter: {config.total_params_in_billions:.2f}B")
    
    if config.total_params_in_billions > 3.5:
        print(f"\n⚠️ Warnung: Modell hat {config.total_params_in_billions:.2f}B Parameter (Ziel: 3B)")
    
    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    try:
        # Modell erstellen
        print("\n🔨 Initialisiere Modell...")
        
        if device.type == "cuda":
            with torch.cuda.device(0):
                model = LFM3BForCausalLM(config)
                if args.fp16:
                    model = model.half()
                model = model.to(device)
        else:
            model = LFM3BForCausalLM(config)
            if args.fp16:
                model = model.half()
        
        # Memory info
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"✅ Modell erstellt! GPU-Speicher: {allocated:.1f} GB")
        
        # Test
        print("\n🧪 Teste Forward Pass...")
        with torch.no_grad():
            test_input = torch.randint(0, config.vocab_size, (1, 10), device=device)
            output = model(test_input)
            print(f"✅ Output shape: {output.logits.shape}")
        
        # Zusammenfassung
        print_model_summary(model)
        
        # Speichern
        if args.save_path:
            print(f"\n💾 Speichere nach: {args.save_path}")
            if device.type == "cuda":
                model = model.cpu()
            save_model(model, args.save_path, config)
            print("✅ Gespeichert!")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        if "out of memory" in str(e).lower():
            print("\nVersuche --fp16 für weniger Speicher")
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()