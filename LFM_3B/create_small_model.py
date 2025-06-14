#!/usr/bin/env python3
"""
Script zum Erstellen einer kleineren LFM-3B Modellvariante für Tests
"""

import torch
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.config import LFM3BConfig
from LFM_3B.model import LFM3BForCausalLM
from LFM_3B.utils import save_model, print_model_summary, get_model_size_mb
import argparse


def create_small_model(model_size="tiny"):
    """Erstellt kleinere Modellvarianten für Tests"""
    
    configs = {
        "tiny": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_experts": 2,
            "intermediate_size": 512,
            "liquid_reservoir_size": 64,
            "liquid_num_reservoirs": 1,
            "vocab_size": 10000,  # Kleineres Vokabular
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_experts": 4,
            "intermediate_size": 1024,
            "liquid_reservoir_size": 128,
            "liquid_num_reservoirs": 2,
            "vocab_size": 32000,
        },
        "medium": {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "num_experts": 4,
            "intermediate_size": 2048,
            "liquid_reservoir_size": 256,
            "liquid_num_reservoirs": 2,
            "vocab_size": 50000,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size must be one of: {list(configs.keys())}")
    
    # Konfiguration erstellen
    config_params = configs[model_size]
    config = LFM3BConfig(**config_params)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Erstelle kleine LFM-3B Modellvarianten")
    
    parser.add_argument("--size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"],
                        help="Modellgröße: tiny (~10M), small (~50M), medium (~200M)")
    parser.add_argument("--medical-mode", action="store_true",
                        help="Medical Mode aktivieren")
    parser.add_argument("--save-path", type=str,
                        help="Pfad zum Speichern des Modells")
    parser.add_argument("--cuda", action="store_true",
                        help="GPU verwenden wenn verfügbar")
    parser.add_argument("--test", action="store_true",
                        help="Nur testen, nicht speichern")
    
    args = parser.parse_args()
    
    print(f"Erstelle {args.size} Modell...")
    
    # Konfiguration erstellen
    config = create_small_model(args.size)
    config.medical_mode = args.medical_mode
    
    print(f"\nModell-Konfiguration ({args.size}):")
    print(f"- Hidden size: {config.hidden_size}")
    print(f"- Layers: {config.num_hidden_layers}")
    print(f"- Attention heads: {config.num_attention_heads}")
    print(f"- Experts: {config.num_experts}")
    print(f"- Vocabulary: {config.vocab_size:,}")
    print(f"- Geschätzte Parameter: {config.total_params_in_billions:.3f}B")
    
    if args.test:
        print("\n✅ Konfiguration erfolgreich erstellt (Testmodus)")
        return
    
    try:
        # Modell initialisieren
        print("\nInitialisiere Modell...")
        model = LFM3BForCausalLM(config)
        
        # Modellgröße anzeigen
        size_mb = get_model_size_mb(model)
        print(f"Modellgröße: {size_mb:.2f} MB")
        
        # Auf GPU verschieben wenn gewünscht
        if args.cuda and torch.cuda.is_available():
            print("\nVerschiebe auf GPU...")
            model = model.cuda()
            print(f"Läuft auf: {torch.cuda.get_device_name(0)}")
        
        # Test Forward Pass
        print("\nTeste Forward Pass...")
        device = next(model.parameters()).device
        test_input = torch.randint(0, config.vocab_size, (1, 10), device=device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward Pass erfolgreich! Output shape: {output.logits.shape}")
        
        # Speichern
        if args.save_path:
            print(f"\nSpeichere Modell nach: {args.save_path}")
            save_model(model, args.save_path, config)
            print("✅ Modell gespeichert!")
        
        # Zusammenfassung
        print("\n" + "="*60)
        print_model_summary(model)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ Nicht genug Speicher! Versuche eine kleinere Variante:")
            print("   python3 create_small_model.py --size tiny")
        else:
            raise e


if __name__ == "__main__":
    main()