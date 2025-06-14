#!/usr/bin/env python3
"""
Tokenizer Integration für LFM-3B
Verwendet GPT-2 Tokenizer als Basis (50k Vokabular)
"""

import torch
import sys
import os
from typing import List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.utils import load_model
from LFM_3B.model import LFM3BForCausalLM


class LFM3BTokenizer:
    """Wrapper für Tokenizer mit LFM-3B"""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Spezielle Tokens hinzufügen
            special_tokens = {
                "pad_token": "<pad>",
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "unk_token": "<unk>",
            }
            self.tokenizer.add_special_tokens(special_tokens)
            
            print(f"✅ Tokenizer geladen: {tokenizer_name}")
            print(f"   Vocabulary size: {len(self.tokenizer)}")
            
        except ImportError:
            print("❌ Bitte installiere transformers: pip install transformers")
            raise
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Enkodiert Text zu Token IDs"""
        return self.tokenizer(text, return_tensors="pt", **kwargs)
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Dekodiert Token IDs zu Text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(self, token_ids: torch.Tensor, **kwargs) -> List[str]:
        """Dekodiert Batch von Token IDs"""
        return self.tokenizer.batch_decode(token_ids, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


class LFM3BWithTokenizer:
    """LFM-3B Modell mit integriertem Tokenizer"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        # Lade Modell
        print(f"📂 Lade Modell von: {model_path}")
        self.model = load_model(LFM3BForCausalLM, model_path)
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = LFM3BTokenizer()
        
        print(f"✅ Modell geladen auf {self.device}")
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generiert Text basierend auf einem Prompt"""
        
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_attention_mask=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        print(f"\n🔤 Prompt: '{prompt}'")
        print(f"   Token IDs: {input_ids.shape}")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
    
    def chat(self):
        """Interaktiver Chat-Modus"""
        print("\n💬 LFM-3B Chat (tippe 'exit' zum Beenden)")
        print("-" * 50)
        
        while True:
            prompt = input("\n👤 You: ")
            if prompt.lower() in ["exit", "quit", "bye"]:
                print("👋 Auf Wiedersehen!")
                break
            
            responses = self.generate_text(
                prompt,
                max_length=150,
                temperature=0.7,
                num_return_sequences=1,
            )
            
            print(f"\n🤖 LFM-3B: {responses[0]}")


def demo_medical_generation(model_with_tokenizer):
    """Demo für medizinische Textgenerierung"""
    print("\n🏥 Medical Text Generation Demo")
    print("=" * 50)
    
    medical_prompts = [
        "The patient presents with symptoms of",
        "Diagnosis: Based on the lab results showing",
        "Treatment plan: The recommended approach for",
        "Medical history: The patient has a history of",
    ]
    
    for prompt in medical_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        responses = model_with_tokenizer.generate_text(
            prompt,
            max_length=80,
            temperature=0.6,  # Niedrigere Temperatur für medizinische Texte
            top_p=0.9,
        )
        print(f"🤖 Generated: {responses[0]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LFM-3B mit Tokenizer")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Pfad zum gespeicherten Modell")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--chat", action="store_true",
                        help="Interaktiver Chat-Modus")
    parser.add_argument("--medical-demo", action="store_true",
                        help="Medical Text Generation Demo")
    parser.add_argument("--prompt", type=str,
                        help="Einzelner Prompt zum Testen")
    
    args = parser.parse_args()
    
    # Check if transformers is installed
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers nicht installiert!")
        print("   Installiere mit: pip install transformers")
        return
    
    # Lade Modell mit Tokenizer
    model_with_tokenizer = LFM3BWithTokenizer(args.model_path, args.device)
    
    if args.chat:
        model_with_tokenizer.chat()
    elif args.medical_demo:
        demo_medical_generation(model_with_tokenizer)
    elif args.prompt:
        responses = model_with_tokenizer.generate_text(args.prompt)
        print(f"\n🤖 Generated: {responses[0]}")
    else:
        # Standard Demo
        test_prompt = "The future of artificial intelligence in medicine"
        print(f"\n🧪 Test mit: '{test_prompt}'")
        responses = model_with_tokenizer.generate_text(test_prompt, max_length=100)
        print(f"\n🤖 Generated: {responses[0]}")


if __name__ == "__main__":
    main()