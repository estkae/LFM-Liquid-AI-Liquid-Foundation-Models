#!/usr/bin/env python3
"""
Script zum Erstellen und Speichern eines LFM-3B Modells
"""

import torch
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.config import LFM3BConfig
from LFM_3B.model import LFM3BForCausalLM
from LFM_3B.utils import save_model, print_model_summary
import argparse


def create_model(args):
    """Erstellt ein neues LFM-3B Modell"""
    
    # Konfiguration erstellen
    print("Erstelle Modell-Konfiguration...")
    config = LFM3BConfig(
        medical_mode=args.medical_mode,
        use_liquid_layers=args.use_liquid,
        num_hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_experts=args.num_experts,
    )
    
    # Modell initialisieren
    print("\nInitialisiere LFM-3B Modell...")
    model = LFM3BForCausalLM(config)
    
    # Modell-Zusammenfassung anzeigen
    print_model_summary(model)
    
    # Optional: Modell auf GPU verschieben
    if args.cuda and torch.cuda.is_available():
        print("\nVerschiebe Modell auf GPU...")
        model = model.cuda()
        print(f"Modell läuft auf: {torch.cuda.get_device_name(0)}")
    
    # Modell speichern
    if args.save_path:
        print(f"\nSpeichere Modell nach: {args.save_path}")
        save_model(model, args.save_path, config)
    
    return model, config


def test_generation(model, config):
    """Testet die Textgenerierung"""
    print("\n" + "="*60)
    print("Teste Textgenerierung...")
    print("="*60)
    
    # Beispiel-Input (normalerweise würde man einen Tokenizer verwenden)
    batch_size = 1
    prompt_length = 5
    device = next(model.parameters()).device
    
    # Zufällige Token-IDs als Beispiel
    input_ids = torch.randint(1, 1000, (batch_size, prompt_length), device=device)
    
    print(f"\nInput-Shape: {input_ids.shape}")
    print("Generiere Text...")
    
    # Generierung
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=50,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    
    print(f"Output-Shape: {output.shape}")
    print(f"Generierte {output.shape[1] - prompt_length} neue Tokens")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Erstelle ein LFM-3B Modell")
    
    # Modell-Konfiguration
    parser.add_argument("--hidden-size", type=int, default=3072,
                        help="Hidden size des Modells (default: 3072)")
    parser.add_argument("--num-layers", type=int, default=20,
                        help="Anzahl der Transformer-Layer (default: 20)")
    parser.add_argument("--num-heads", type=int, default=24,
                        help="Anzahl der Attention-Heads (default: 24)")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Anzahl der Experten im MoE (default: 8)")
    
    # Liquid Neural Network
    parser.add_argument("--use-liquid", action="store_true", default=True,
                        help="Liquid Neural Networks verwenden")
    parser.add_argument("--no-liquid", dest="use_liquid", action="store_false",
                        help="Liquid Neural Networks deaktivieren")
    
    # Medical Mode
    parser.add_argument("--medical-mode", action="store_true",
                        help="Medical Mode aktivieren")
    
    # Speichern
    parser.add_argument("--save-path", type=str, default="./lfm3b_model",
                        help="Pfad zum Speichern des Modells")
    
    # GPU
    parser.add_argument("--cuda", action="store_true",
                        help="GPU verwenden wenn verfügbar")
    
    # Test
    parser.add_argument("--test-generation", action="store_true",
                        help="Textgenerierung testen")
    
    args = parser.parse_args()
    
    # Modell erstellen
    model, config = create_model(args)
    
    # Optional: Generation testen
    if args.test_generation:
        test_generation(model, config)
    
    print("\n✅ Modell erfolgreich erstellt!")
    
    if args.save_path:
        print(f"\nModell gespeichert in: {args.save_path}/")
        print("\nZum Laden des Modells:")
        print(f"  from LFM_3B.utils import load_model")
        print(f"  from LFM_3B.model import LFM3BForCausalLM")
        print(f"  model = load_model(LFM3BForCausalLM, '{args.save_path}')")


if __name__ == "__main__":
    main()