#!/usr/bin/env python3
"""
Training Script für deutsches Gemeindeverwaltungs-LFM-3B mit MoE
"""

import torch
import torch.nn as nn
import json
import sys
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig


def create_municipal_sample_data(output_file: str):
    """Erstellt Beispiel-Trainingsdaten für Gemeindeverwaltung"""
    
    sample_data = [
        {
            "text": "Bürger: Ich möchte einen neuen Personalausweis beantragen. Sachbearbeiter: Gerne. Bringen Sie bitte ein biometrisches Passfoto, Ihren alten Ausweis und 37 Euro Gebühr mit.",
            "type": "buergerservice_dialog",
            "language": "de",
            "category": "einwohnermeldeamt"
        },
        {
            "text": "Antrag auf Baugenehmigung für Einfamilienhaus, Flurstück 123/4. Bauvoranfrage wurde positiv beschieden. Einreichung der Bauunterlagen erfolgt fristgerecht. Stellungnahme der Nachbarn liegt vor.",
            "type": "verwaltungsakt",
            "language": "de",
            "category": "bauamt"
        },
        {
            "text": "Bescheid: Die beantragte Gewerbeerlaubnis für einen Imbissbetrieb in der Hauptstraße 15 wird unter Auflagen erteilt. Öffnungszeiten: 10-22 Uhr. Lärmschutzauflagen sind zu beachten.",
            "type": "bescheid",
            "language": "de",
            "category": "ordnungsamt"
        },
        {
            "text": "Bürger: Wo kann ich die Hundesteuer anmelden? Mitarbeiter: Die Anmeldung erfolgt bei der Stadtkasse im Erdgeschoss. Für einen Hund beträgt die jährliche Steuer 120 Euro.",
            "type": "buergerservice_dialog",
            "language": "de",
            "category": "stadtkasse"
        },
        {
            "text": "Protokoll der Gemeinderatssitzung vom 15.03.2024: TOP 1: Haushalt 2024 einstimmig beschlossen. TOP 2: Sanierung Grundschule, Kosten 2,5 Mio Euro, Förderantrag bewilligt.",
            "type": "sitzungsprotokoll",
            "language": "de",
            "category": "gemeinderat"
        },
        {
            "text": "Standesamtliche Trauung: Terminvereinbarung für den 20.06.2024 um 14:00 Uhr. Benötigte Unterlagen: Personalausweise, Geburtsurkunden, Ehefähigkeitszeugnis. Gebühr: 80 Euro.",
            "type": "verwaltungsvorgang",
            "language": "de",
            "category": "standesamt"
        },
        {
            "text": "Bürger: Mein Müll wurde nicht abgeholt. Sachbearbeiter: Die gelben Säcke werden jeden zweiten Donnerstag abgeholt. Nächster Termin ist der 28.03. Die Abfuhrtermine finden Sie auch in unserer App.",
            "type": "buergerservice_dialog",
            "language": "de",
            "category": "abfallwirtschaft"
        },
        {
            "text": "Antrag auf Wohngeld wurde geprüft. Bewilligungszeitraum: 01.04.2024 bis 31.03.2025. Monatlicher Zuschuss: 215 Euro. Auszahlung erfolgt zum Monatsersten auf das angegebene Konto.",
            "type": "bescheid",
            "language": "de",
            "category": "sozialamt"
        },
        {
            "text": "Verkehrsrechtliche Anordnung: Einrichtung einer Tempo-30-Zone in der Schulstraße. Begründung: Schulwegsicherung. Gültigkeit ab 01.05.2024. Beschilderung durch Bauhof.",
            "type": "verwaltungsakt",
            "language": "de",
            "category": "ordnungsamt"
        },
        {
            "text": "Bürger: Ich möchte einen Kitaplatz anmelden. Mitarbeiter: Die Anmeldung erfolgt über unser Online-Portal. Sie können bis zu drei Wunsch-Kitas angeben. Die Platzvergabe erfolgt nach Punktesystem.",
            "type": "buergerservice_dialog",
            "language": "de",
            "category": "jugendamt"
        },
        {
            "text": "Grundsteuerbescheid 2024: Grundstück Hauptstraße 42, Einheitswert: 45.000 Euro, Hebesatz: 380%, Jahressteuer: 684 Euro, zahlbar in vier Raten. Widerspruchsfrist: ein Monat.",
            "type": "steuerbescheid",
            "language": "de",
            "category": "stadtkasse"
        },
        {
            "text": "Meldung Straßenschaden: Schlagloch in der Bahnhofstraße, Höhe Hausnummer 15. Größe ca. 50x30cm, Tiefe 10cm. Gefährdung für Radfahrer. Weiterleitung an Bauhof zur Reparatur.",
            "type": "meldung",
            "language": "de",
            "category": "tiefbauamt"
        },
        {
            "text": "Bürgerversammlung 20.03.2024: Vorstellung Bebauungsplan Neubaugebiet Süd. 45 Bauplätze geplant. Bürgeranregungen: Mehr Grünflächen, Spielplatz, Verkehrsberuhigung. Protokoll online einsehbar.",
            "type": "veranstaltungsprotokoll",
            "language": "de",
            "category": "stadtplanung"
        },
        {
            "text": "Führungszeugnis beantragt für Tätigkeit im Kindergarten. Gebühr 13 Euro bezahlt. Bearbeitungszeit ca. 2 Wochen. Versand direkt an Arbeitgeber oder Abholung im Rathaus möglich.",
            "type": "verwaltungsvorgang",
            "language": "de",
            "category": "einwohnermeldeamt"
        },
        {
            "text": "Vollzugsmeldung: Gaststätte 'Zum Löwen' - Kontrolle Nichtraucherschutz durchgeführt. Keine Verstöße festgestellt. Hygienekonzept liegt vor. Nächste Regelkontrolle in 6 Monaten.",
            "type": "kontrollbericht",
            "language": "de",
            "category": "ordnungsamt"
        }
    ]
    
    # Erweitere die Daten durch Variationen
    extended_data = []
    for item in sample_data:
        extended_data.append(item)
        
        # Füge Variationen hinzu (für Demo)
        for i in range(2):
            variation = item.copy()
            variation["text"] = f"Fall {i+1}: " + variation["text"]
            extended_data.append(variation)
    
    # Speichere als JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extended_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(extended_data)} Beispiele in {output_file} gespeichert")
    
    # Zeige Statistiken
    print("\n📊 Datensatz-Statistiken:")
    types = {}
    categories = {}
    
    for item in sample_data:
        types[item["type"]] = types.get(item["type"], 0) + 1
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    
    print("\nTexttypen:")
    for t, count in sorted(types.items()):
        print(f"  - {t}: {count}")
    
    print("\nAbteilungen:")
    for c, count in sorted(categories.items()):
        print(f"  - {c}: {count}")


class MunicipalDataset(Dataset):
    """Dataset für deutsche Gemeindeverwaltungstexte"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Lade Daten
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        print(f"✅ {len(self.data)} Trainingsbeispiele geladen")
        
        # Statistiken
        self.categories = {}
        for item in self.data:
            cat = item.get('category', 'unknown')
            self.categories[cat] = self.categories.get(cat, 0) + 1
        
        print("\n📊 Daten nach Abteilungen:")
        for cat, count in sorted(self.categories.items()):
            print(f"  - {cat}: {count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format text with category info
        text = f"[{item['category'].upper()}] {item['text']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels for causal LM
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "category": item['category']
        }


def train_municipal_moe(
    model_path: str,
    data_file: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    device: str = "cuda"
):
    """Train Municipal MoE Model"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training auf: {device}")
    
    # Load model
    logger.info(f"Lade MoE Model von: {model_path}")
    model = MunicipalMoEModel.from_pretrained(model_path)
    model = model.to(device)
    model.train()
    
    # Tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        logger.info("✅ Deutscher Tokenizer geladen")
    except:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("⚠️ Fallback zu GPT-2 Tokenizer")
    
    # Dataset
    dataset = MunicipalDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(dataloader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Training metrics
    expert_usage = {i: 0 for i in range(model.config.num_experts)}
    category_losses = {cat: [] for cat in dataset.categories.keys()}
    
    # Training loop
    logger.info("🚀 Starte Training...")
    global_step = 0
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            categories = batch["category"]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # Track loss by category
            for i, cat in enumerate(categories):
                category_losses[cat].append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Stats
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Save checkpoint every 500 steps
            if global_step % 500 == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                logger.info(f"✅ Checkpoint gespeichert: {checkpoint_dir}")
        
        # Epoch stats
        avg_loss = total_loss / num_batches
        logger.info(f"\nEpoch {epoch+1}: Durchschnittlicher Loss = {avg_loss:.4f}")
        
        # Category-specific stats
        logger.info("\n📊 Loss nach Abteilungen:")
        for cat, losses in category_losses.items():
            if losses:
                avg_cat_loss = sum(losses[-100:]) / len(losses[-100:])  # Last 100
                logger.info(f"  - {cat}: {avg_cat_loss:.4f}")
    
    # Save final model
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    logger.info(f"\n✅ Finales Modell gespeichert: {final_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training für Municipal MoE Model")
    
    parser.add_argument("--create-sample-data", type=str,
                        help="Erstelle Beispieldaten und speichere sie")
    parser.add_argument("--create-base-model", action="store_true",
                        help="Erstelle Basis MoE Modell")
    parser.add_argument("--model-path", type=str,
                        help="Pfad zum MoE Modell")
    parser.add_argument("--data-file", type=str,
                        help="JSONL Datei mit Trainingsdaten")
    parser.add_argument("--output-dir", type=str, default="./municipal_trained",
                        help="Ausgabeverzeichnis")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Anzahl Epochen")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch Size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning Rate")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_municipal_sample_data(args.create_sample_data)
        return
    
    if args.create_base_model:
        from LFM_3B.municipal_moe_model import create_municipal_moe_model
        model, config = create_municipal_moe_model("./municipal_moe_base")
        print("✅ Basis MoE Modell erstellt in: ./municipal_moe_base")
        return
    
    if not args.model_path or not args.data_file:
        print("❌ Für Training werden --model-path und --data-file benötigt")
        print("\nBeispiel Workflow:")
        print("1. python train_german_municipal.py --create-base-model")
        print("2. python train_german_municipal.py --create-sample-data municipal_data.jsonl")
        print("3. python train_german_municipal.py --model-path ./municipal_moe_base --data-file municipal_data.jsonl")
        return
    
    # Check dependencies
    try:
        import torch
        import transformers
        from tqdm import tqdm
    except ImportError:
        print("❌ Installiere: pip install torch transformers tqdm")
        return
    
    # Start training
    train_municipal_moe(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    print(f"\n📝 DATENFORMAT (JSONL):")
    print("Jede Zeile ist ein JSON-Objekt mit folgender Struktur:")
    print("""
{
    "text": "Der Dialog oder Verwaltungstext",
    "type": "Art des Dokuments/Dialogs",
    "language": "de",
    "category": "Zuständige Abteilung"
}

Verfügbare Typen:
- buergerservice_dialog: Bürger-Mitarbeiter Dialog
- verwaltungsakt: Formeller Verwaltungsakt
- bescheid: Offizieller Bescheid
- sitzungsprotokoll: Protokoll einer Sitzung
- verwaltungsvorgang: Allgemeiner Verwaltungsvorgang
- steuerbescheid: Steuerbescheid
- meldung: Bürgermeldung/Hinweis
- veranstaltungsprotokoll: Protokoll öffentlicher Veranstaltungen
- kontrollbericht: Bericht über Kontrollen/Überprüfungen

Kategorien (Abteilungen):
- einwohnermeldeamt: Meldewesen, Ausweise
- bauamt: Baugenehmigungen, Bauvorhaben
- ordnungsamt: Ordnungswidrigkeiten, Genehmigungen
- stadtkasse: Steuern, Gebühren
- gemeinderat: Politische Gremien
- standesamt: Personenstandswesen
- abfallwirtschaft: Müllentsorgung
- sozialamt: Sozialleistungen
- jugendamt: Kinder- und Jugendhilfe
- tiefbauamt: Straßen, Infrastruktur
- stadtplanung: Bebauungspläne, Entwicklung
""")


if __name__ == "__main__":
    main()