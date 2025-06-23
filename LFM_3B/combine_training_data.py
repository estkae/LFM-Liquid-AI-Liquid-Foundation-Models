#!/usr/bin/env python3
"""
Kombiniert verschiedene Training-Datenquellen für das Municipal MoE Model
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Lädt JSONL-Daten"""
    data = []
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def combine_training_data(
    pdf_data_file: str = "pdf_training_data.jsonl",
    existing_data_file: str = "massive_municipal_training_data.jsonl",
    output_file: str = "combined_municipal_training_data.jsonl",
    balance_ratio: float = 0.3  # 30% PDF-Daten, 70% generierte Daten
):
    """Kombiniert PDF-Daten mit bestehenden Training-Daten"""
    
    print("🔄 Kombiniere Training-Daten...")
    
    # Lade PDF-Daten
    pdf_data = load_jsonl_data(pdf_data_file)
    print(f"📄 PDF-Daten: {len(pdf_data)} Beispiele")
    
    # Lade bestehende Daten
    existing_data = load_jsonl_data(existing_data_file)
    print(f"📚 Bestehende Daten: {len(existing_data)} Beispiele")
    
    if not pdf_data and not existing_data:
        print("❌ Keine Daten zum Kombinieren gefunden")
        return
    
    # Berechne Ziel-Anzahl
    total_pdf = len(pdf_data)
    total_existing = len(existing_data)
    
    if total_pdf > 0 and total_existing > 0:
        # Verwende Balance-Ratio
        target_pdf = int(total_pdf * balance_ratio / (1 - balance_ratio))
        target_existing = min(target_pdf, total_existing)
        target_pdf = min(total_pdf, int(target_existing * balance_ratio / (1 - balance_ratio)))
    else:
        target_pdf = total_pdf
        target_existing = total_existing
    
    print(f"🎯 Ziel-Balance: {target_pdf} PDF + {target_existing} generierte Daten")
    
    # Sampling
    combined_data = []
    
    if pdf_data:
        sampled_pdf = random.sample(pdf_data, min(target_pdf, len(pdf_data)))
        combined_data.extend(sampled_pdf)
        print(f"✅ {len(sampled_pdf)} PDF-Beispiele hinzugefügt")
    
    if existing_data:
        sampled_existing = random.sample(existing_data, min(target_existing, len(existing_data)))
        combined_data.extend(sampled_existing)
        print(f"✅ {len(sampled_existing)} generierte Beispiele hinzugefügt")
    
    # Mische die Daten
    random.shuffle(combined_data)
    
    # Speichere kombinierte Daten
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"🎉 {len(combined_data)} kombinierte Beispiele gespeichert in {output_file}")
    
    # Zeige Beispiele
    print("\n📋 Beispiele aus kombinierten Daten:")
    for i, example in enumerate(combined_data[:3]):
        source = example.get('source', 'generated')
        text_preview = example['text'][:100] + "..." if len(example['text']) > 100 else example['text']
        print(f"{i+1}. [{source}] {text_preview}")

def main():
    parser = argparse.ArgumentParser(description="Kombiniert Training-Daten")
    parser.add_argument("--pdf-data", type=str, default="pdf_training_data.jsonl",
                       help="PDF Training-Daten Datei")
    parser.add_argument("--existing-data", type=str, default="massive_municipal_training_data.jsonl",
                       help="Bestehende Training-Daten")
    parser.add_argument("--output", type=str, default="combined_municipal_training_data.jsonl",
                       help="Output-Datei")
    parser.add_argument("--balance-ratio", type=float, default=0.3,
                       help="Anteil PDF-Daten (0.0-1.0)")
    
    args = parser.parse_args()
    
    combine_training_data(
        pdf_data_file=args.pdf_data,
        existing_data_file=args.existing_data,
        output_file=args.output,
        balance_ratio=args.balance_ratio
    )

if __name__ == "__main__":
    main()