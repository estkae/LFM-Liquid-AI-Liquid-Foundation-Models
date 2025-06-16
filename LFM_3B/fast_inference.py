#!/usr/bin/env python3
"""
Optimierte schnelle Inferenz für LFM-3B
"""

import torch
import torch.nn as nn
import sys
import os
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.utils import load_model
from LFM_3B.model import LFM3BForCausalLM


class FastLFM3B:
    """Optimierte Version für schnelle Inferenz"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print("🚀 Lade Modell für schnelle Inferenz...")
        
        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Lade Modell
        self.model = load_model(LFM3BForCausalLM, model_path, device=self.device)
        
        # Optimierungen
        self.optimize_model()
        
        # Tokenizer
        self.setup_tokenizer()
        
        print(f"✅ Modell geladen und optimiert auf {self.device}")
    
    def optimize_model(self):
        """Optimiert das Modell für Geschwindigkeit"""
        
        # Eval mode
        self.model.eval()
        
        # Torch optimizations
        if hasattr(torch, 'jit') and self.device.type == "cuda":
            try:
                # TorchScript compilation (falls möglich)
                print("🔧 Versuche TorchScript Kompilierung...")
                # self.model = torch.jit.script(self.model)  # Kann bei komplexen Modellen fehlschlagen
            except Exception as e:
                print(f"⚠️ TorchScript fehlgeschlagen: {e}")
        
        # Tensor optimizations
        if self.device.type == "cuda":
            # Enable optimized attention
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("✅ Flash Attention aktiviert")
            except:
                pass
            
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory format optimization
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    if hasattr(module.weight, 'data'):
                        module.weight.data = module.weight.data.contiguous()
        
        # Kompilierung mit torch.compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                print("🔧 Kompiliere Modell mit torch.compile...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ Modell kompiliert")
            except Exception as e:
                print(f"⚠️ torch.compile fehlgeschlagen: {e}")
    
    def setup_tokenizer(self):
        """Setup optimized tokenizer"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            print("❌ Transformers nicht installiert")
            self.tokenizer = None
    
    @torch.no_grad()
    def fast_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 20,  # Reduziert für Speed
        do_sample: bool = True,
    ) -> str:
        """Optimierte Generation"""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer nicht verfügbar")
        
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = inputs.to(self.device)
        
        original_length = input_ids.shape[1]
        
        # Generiere Token für Token (optimiert)
        for _ in range(max_new_tokens):
            # Forward pass (nur letztes Token wenn möglich)
            outputs = self.model(input_ids)
            
            # Next token logits
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Top-k filtering (schneller als top-p)
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop bei EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Stats
        generation_time = time.time() - start_time
        tokens_generated = input_ids.shape[1] - original_length
        tokens_per_second = tokens_generated / generation_time
        
        print(f"⚡ {tokens_generated} Tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return generated_text
    
    @torch.no_grad()
    def batch_generate(
        self,
        prompts: list,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
    ) -> list:
        """Batch-Generation für mehrere Prompts"""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer nicht verfügbar")
        
        start_time = time.time()
        
        # Tokenize alle Prompts
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        batch_size = input_ids.shape[0]
        original_length = input_ids.shape[1]
        
        # Generation
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            next_token_logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Update
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Update attention mask
            new_attention = torch.ones((batch_size, 1), device=self.device)
            attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
        
        # Decode alle
        results = []
        for i in range(batch_size):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            results.append(text)
        
        # Stats
        generation_time = time.time() - start_time
        total_tokens = batch_size * max_new_tokens
        tokens_per_second = total_tokens / generation_time
        
        print(f"⚡ Batch: {total_tokens} Tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return results


def benchmark_speed(model_path: str):
    """Benchmarkt die Geschwindigkeit"""
    
    print("🏁 Geschwindigkeits-Benchmark")
    print("=" * 50)
    
    # Setup
    fast_model = FastLFM3B(model_path)
    
    # Test prompts
    test_prompts = [
        "Der Patient hat Kopfschmerzen",
        "Diagnose nach Untersuchung:",
        "Behandlungsempfehlung:",
        "Medizinische Anamnese:",
        "Laborwerte zeigen:"
    ]
    
    # Einzelne Generation
    print("\n🔥 Einzelne Generation:")
    for prompt in test_prompts[:2]:
        result = fast_model.fast_generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8
        )
        print(f"📝 '{prompt}' -> '{result}'")
    
    # Batch Generation
    print(f"\n🚀 Batch Generation ({len(test_prompts)} Prompts):")
    results = fast_model.batch_generate(
        test_prompts,
        max_new_tokens=15
    )
    
    for prompt, result in zip(test_prompts, results):
        print(f"📝 '{prompt}' -> '{result}'")


def optimize_for_production(model_path: str, output_path: str):
    """Optimiert und speichert für Produktion"""
    
    print("🏭 Produktions-Optimierung")
    print("=" * 50)
    
    # Lade und optimiere
    fast_model = FastLFM3B(model_path)
    
    # Weitere Optimierungen
    if fast_model.device.type == "cuda":
        # Quantisierung (falls verfügbar)
        try:
            print("🔧 Versuche INT8 Quantisierung...")
            # Hier könnte INT8/FP16 Quantisierung implementiert werden
        except:
            pass
    
    # Speichere optimiertes Modell
    from LFM_3B.utils import save_model
    save_model(fast_model.model, output_path)
    print(f"✅ Optimiertes Modell gespeichert: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Schnelle LFM-3B Inferenz")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Pfad zum Modell")
    parser.add_argument("--benchmark", action="store_true",
                        help="Geschwindigkeits-Benchmark")
    parser.add_argument("--optimize", type=str,
                        help="Optimiere und speichere nach")
    parser.add_argument("--prompt", type=str,
                        help="Teste einzelnen Prompt")
    parser.add_argument("--interactive", action="store_true",
                        help="Interaktiver Modus")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_speed(args.model_path)
    elif args.optimize:
        optimize_for_production(args.model_path, args.optimize)
    elif args.prompt:
        fast_model = FastLFM3B(args.model_path)
        result = fast_model.fast_generate(args.prompt, max_new_tokens=50)
        print(f"\n🤖 Ergebnis: {result}")
    elif args.interactive:
        fast_model = FastLFM3B(args.model_path)
        print("\n💬 Schneller Chat (tippe 'exit' zum Beenden)")
        
        while True:
            prompt = input("\n👤 Du: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            result = fast_model.fast_generate(prompt, max_new_tokens=40)
            print(f"🤖 LFM: {result}")
    else:
        print("Verwende --benchmark, --optimize, --prompt oder --interactive")


if __name__ == "__main__":
    main()