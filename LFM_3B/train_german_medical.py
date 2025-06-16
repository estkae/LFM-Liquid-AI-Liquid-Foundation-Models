#!/usr/bin/env python3
"""
Training Script für deutsches medizinisches LFM-3B
"""

import torch
import torch.nn as nn
import json
import sys
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.utils import load_model, save_model
from LFM_3B.model import LFM3BForCausalLM
from lfm.medical_health_base import MedicalHealthBaseModel


class GermanMedicalDataset(Dataset):
    """Dataset für deutsche medizinische Texte"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Lade Daten
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        print(f"✅ {len(self.data)} Trainingsbeispiele geladen")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: {"text": "...", "type": "medical", "language": "de"}
        text = item["text"]
        
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
        
        # Labels für Causal LM (input_ids shifted)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_sample_data(output_file: str):
    """Erstellt Beispiel-Trainingsdaten"""
    
    sample_data = [
        {
            "text": "Patient: Ich habe seit drei Tagen starke Kopfschmerzen. Arzt: Können Sie die Schmerzen genauer beschreiben? Sind sie pulsierend oder drückend?",
            "type": "medical_dialogue",
            "language": "de",
            "category": "neurologie"
        },
        {
            "text": "Diagnose: Migräne ohne Aura. Anamnese: 35-jährige Patientin mit wiederkehrenden einseitigen Kopfschmerzen. Behandlung: Triptane bei akuten Episoden.",
            "type": "medical_report",
            "language": "de", 
            "category": "neurologie"
        },
        {
            "text": "Symptome: Fieber 38.5°C, Halsschmerzen, geschwollene Lymphknoten. Verdachtsdiagnose: Virale Pharyngitis. Empfehlung: Symptomatische Behandlung, Wiedervorstellung bei Verschlechterung.",
            "type": "medical_assessment",
            "language": "de",
            "category": "hno"
        },
        {
            "text": "Patient klagt über Atemnot bei Belastung. EKG zeigt Sinusrhythmus. Echokardiographie empfohlen zur weiteren Abklärung einer möglichen Herzinsuffizienz.",
            "type": "medical_report",
            "language": "de",
            "category": "kardiologie"
        },
        {
            "text": "Laborwerte: Hb 12.1 g/dl, Leukozyten 8.500/μl, CRP 2.1 mg/l. Befund: Werte im Normbereich. Empfehlung: Kontrolle in 6 Monaten.",
            "type": "lab_report",
            "language": "de",
            "category": "labor"
        }
    ]
    
    # Erweitere die Daten durch Variationen
    extended_data = []
    for item in sample_data:
        extended_data.append(item)
        
        # Füge Variationen hinzu (für Demo)
        for i in range(3):
            variation = item.copy()
            variation["text"] = f"Fallbericht {i+1}: " + variation["text"]
            extended_data.append(variation)
    
    # Speichere als JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extended_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(extended_data)} Beispiele in {output_file} gespeichert")


def train_model(
    model_path: str,
    data_file: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    device: str = "cuda"
):
    """Haupttraining-Funktion"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training auf: {device}")
    
    # Lade Modell
    logger.info(f"Lade Modell von: {model_path}")
    
    # Check if it's a medical model
    config_path = os.path.join(model_path, "medical_config.json")
    if os.path.exists(config_path):
        # Load Medical Health Model
        logger.info("📋 Lade Medical Health Model")
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create model from config
        from lfm.medical_health_base import MedicalHealthConfig, MedicalHealthBaseModel
        config = MedicalHealthConfig(**{k: v for k, v in config_dict.items() if not k.startswith('_')})
        model = MedicalHealthBaseModel(config)
        
        # Load weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            # Try safetensors
            weights_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(weights_path):
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")
    else:
        # Load standard LFM3B model
        model = load_model(LFM3BForCausalLM, model_path)
    
    model = model.to(device)
    model.train()
    
    # Tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
        logger.info("✅ Deutscher Tokenizer geladen")
    except:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info("⚠️ Fallback zu GPT-2")
    
    # Dataset
    dataset = GermanMedicalDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    
    # Training loop
    logger.info("🚀 Starte Training...")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
            if loss is None:
                # Manual loss calculation
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}: Durchschnittlicher Loss = {avg_loss:.4f}")
        
        # Learning rate step
        scheduler.step()
        
        # Speichere Checkpoint
        checkpoint_dir = f"{output_dir}/checkpoint-epoch-{epoch+1}"
        save_model(model, checkpoint_dir)
        logger.info(f"✅ Checkpoint gespeichert: {checkpoint_dir}")
    
    # Finales Modell speichern
    final_dir = f"{output_dir}/final"
    save_model(model, final_dir)
    logger.info(f"✅ Finales Modell gespeichert: {final_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training für deutsches medizinisches LFM-3B")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Pfad zum vortrainierten Modell")
    parser.add_argument("--data-file", type=str,
                        help="JSONL Datei mit Trainingsdaten")
    parser.add_argument("--output-dir", type=str, default="./trained_model",
                        help="Ausgabeverzeichnis")
    parser.add_argument("--create-sample-data", type=str,
                        help="Erstelle Beispieldaten und speichere sie")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Anzahl Epochen")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch Size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning Rate")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data(args.create_sample_data)
        print(f"\n📝 DATENFORMAT (JSONL):")
        print("Jede Zeile ist ein JSON-Objekt:")
        print('{"text": "Patient: Ich habe Kopfschmerzen...", "type": "medical", "language": "de"}')
        return
    
    if not args.data_file:
        print("❌ --data-file erforderlich für Training")
        print("Erstelle zuerst Beispieldaten: --create-sample-data data.jsonl")
        return
    
    # Überprüfe Dependencies
    try:
        import transformers
        from tqdm import tqdm
    except ImportError:
        print("❌ Installiere: pip install transformers tqdm")
        return
    
    # Starte Training
    train_model(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )


if __name__ == "__main__":
    main()