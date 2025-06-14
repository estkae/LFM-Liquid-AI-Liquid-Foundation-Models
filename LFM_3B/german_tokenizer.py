#!/usr/bin/env python3
"""
Deutsche Tokenizer-Integration für LFM-3B
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.utils import load_model
from LFM_3B.model import LFM3BForCausalLM


class GermanLFM3B:
    """LFM-3B mit deutschem Tokenizer"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        # Lade Modell
        print(f"📂 Lade Modell von: {model_path}")
        self.model = load_model(LFM3BForCausalLM, model_path)
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Deutscher Tokenizer
        try:
            from transformers import AutoTokenizer
            # Verwende deutschen BERT oder multilingual
            tokenizer_options = [
                "german-nlp-group/electra-base-german-uncased",
                "dbmdz/bert-base-german-uncased", 
                "xlm-roberta-base"  # Multilingual fallback
            ]
            
            self.tokenizer = None
            for tokenizer_name in tokenizer_options:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    print(f"✅ Deutscher Tokenizer: {tokenizer_name}")
                    break
                except:
                    continue
            
            if self.tokenizer is None:
                # Fallback zu GPT-2
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                print("⚠️ Fallback zu GPT-2 Tokenizer")
            
        except ImportError:
            print("❌ Installiere transformers: pip install transformers")
            raise
    
    def generate_german_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> str:
        """Generiert deutschen Text"""
        
        print(f"\n🇩🇪 Prompt: '{prompt}'")
        
        # Encode
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def medical_chat_german(self):
        """Deutscher Medical Chat"""
        print("\n🏥 LFM-3B Deutscher Medical Chat")
        print("=" * 50)
        print("WARNUNG: Das Modell ist noch nicht trainiert!")
        print("Für sinnvolle Ergebnisse muss es erst auf deutschen medizinischen Texten trainiert werden.")
        print("-" * 50)
        
        while True:
            prompt = input("\n👤 Patient: ")
            if prompt.lower() in ["exit", "quit", "ende", "tschüss"]:
                print("👋 Auf Wiedersehen!")
                break
            
            # Füge medizinischen Kontext hinzu
            medical_prompt = f"Patient sagt: {prompt}\nArzt antwortet:"
            
            response = self.generate_german_text(
                medical_prompt,
                max_length=150,
                temperature=0.6,
            )
            
            print(f"\n🩺 Arzt: {response}")


def demo_german_prompts(german_model):
    """Demo mit deutschen medizinischen Prompts"""
    print("\n🇩🇪 Deutsche Medical Text Generation")
    print("=" * 50)
    print("⚠️ WICHTIG: Das Modell ist noch NICHT trainiert!")
    print("Die Ausgaben sind zufällig und nicht medizinisch korrekt.")
    print("-" * 50)
    
    german_prompts = [
        "Der Patient klagt über Kopfschmerzen",
        "Diagnose: Aufgrund der Symptome",
        "Behandlungsplan: Die empfohlene Therapie",
        "Anamnese: Der Patient hat eine Vorgeschichte von",
    ]
    
    for prompt in german_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        response = german_model.generate_german_text(
            prompt,
            max_length=80,
            temperature=0.8,
        )
        print(f"🤖 Generiert: {response}")


def training_info():
    """Information über Training"""
    print("\n📚 WIE MAN DAS MODELL FÜR DEUTSCH TRAINIERT:")
    print("=" * 60)
    print("1. Deutsche medizinische Texte sammeln:")
    print("   - Medizinische Fachliteratur")
    print("   - Anonymisierte Patientenberichte") 
    print("   - Deutsche Wikipedia Medizin-Artikel")
    print()
    print("2. Daten vorbereiten:")
    print("   - Texte tokenisieren")
    print("   - In Trainings-Format konvertieren")
    print("   - Validierungs-Split erstellen")
    print()
    print("3. Fine-Tuning durchführen:")
    print("   - Kleine Learning Rate (1e-5)")
    print("   - Gradient Accumulation")
    print("   - Medical Safety Checks")
    print()
    print("4. Evaluation:")
    print("   - Deutsche medizinische Benchmarks")
    print("   - Manuelle Qualitätsprüfung")
    print("   - Safety Tests")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LFM-3B mit deutschem Tokenizer")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Pfad zum Modell")
    parser.add_argument("--chat", action="store_true",
                        help="Deutscher Medical Chat")
    parser.add_argument("--demo", action="store_true",
                        help="Deutsche Demo")
    parser.add_argument("--training-info", action="store_true",
                        help="Zeige Training-Informationen")
    parser.add_argument("--prompt", type=str,
                        help="Deutscher Prompt zum Testen")
    
    args = parser.parse_args()
    
    if args.training_info:
        training_info()
        return
    
    # Lade deutsches Modell
    german_model = GermanLFM3B(args.model_path)
    
    if args.chat:
        german_model.medical_chat_german()
    elif args.demo:
        demo_german_prompts(german_model)
    elif args.prompt:
        response = german_model.generate_german_text(args.prompt)
        print(f"\n🤖 Antwort: {response}")
    else:
        print("\n⚠️ WICHTIGER HINWEIS:")
        print("Das Modell ist noch NICHT auf deutschen Texten trainiert!")
        print("Für Training-Informationen verwende: --training-info")
        
        test_prompt = "Der Patient hat Kopfschmerzen"
        response = german_model.generate_german_text(test_prompt)
        print(f"\n🧪 Test: '{test_prompt}'")
        print(f"🤖 Output: {response}")


if __name__ == "__main__":
    main()