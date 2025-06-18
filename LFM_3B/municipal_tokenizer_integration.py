#!/usr/bin/env python3
"""
Municipal MoE Model with Tokenizer Integration
"""

import torch
import argparse
from transformers import AutoTokenizer
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig


class MunicipalTokenizerIntegration:
    """Integration class for Municipal MoE model with tokenizer"""
    
    def __init__(self, model_path: str):
        """Initialize model and tokenizer"""
        print(f"🏛️ Loading Municipal MoE model from {model_path}...")
        
        # Load model
        self.model = MunicipalMoEModel.from_pretrained(model_path)
        self.model.eval()
        
        # Load tokenizer (using GPT-2 tokenizer as base)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📊 Model config: {self.model.config.num_experts} experts, {self.model.config.num_experts_per_tok} experts per token")
        
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text from prompt"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def generate_step_by_step(self, input_ids: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
        """Generate tokens step by step (since model doesn't have built-in generate)"""
        generated = input_ids
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=generated)
                logits = outputs["logits"]
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return generated
    
    def municipal_demo(self):
        """Run municipal administration demos"""
        print("\n🏛️ Municipal Administration Demo\n")
        
        demos = [
            {
                "department": "Einwohnermeldeamt",
                "prompt": "Sehr geehrte Damen und Herren, ich möchte meinen Wohnsitz ummelden",
                "context": "Anmeldung eines neuen Wohnsitzes"
            },
            {
                "department": "Bauamt", 
                "prompt": "Antrag auf Baugenehmigung für einen Wintergarten",
                "context": "Bauantrag einreichen"
            },
            {
                "department": "Standesamt",
                "prompt": "Ich möchte eine Geburtsurkunde beantragen",
                "context": "Urkundenbestellung"
            },
            {
                "department": "Ordnungsamt",
                "prompt": "Beschwerde über Lärmbelästigung in der Nachbarschaft",
                "context": "Ordnungswidrigkeit melden"
            }
        ]
        
        for demo in demos:
            print(f"\n📋 {demo['department']} - {demo['context']}:")
            print(f"📝 Eingabe: {demo['prompt']}")
            
            # Tokenize and generate
            inputs = self.tokenizer(demo['prompt'], return_tensors="pt").to(self.device)
            generated = self.generate_step_by_step(inputs.input_ids, max_new_tokens=50)
            
            # Decode
            response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"💬 Antwort: {response}")
            print("-" * 80)
    
    def interactive_chat(self):
        """Interactive chat mode for municipal queries"""
        print("\n🏛️ Willkommen beim Digitalen Bürgeramt!")
        print("💬 Stellen Sie Ihre Fragen zu kommunalen Dienstleistungen.")
        print("📝 Tippen Sie 'exit' zum Beenden.\n")
        
        while True:
            # Get user input
            user_input = input("🧑 Bürger: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'beenden']:
                print("👋 Auf Wiedersehen! Vielen Dank für Ihren Besuch im digitalen Bürgeramt.")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("🤖 Sachbearbeiter: ", end="", flush=True)
            
            # Tokenize and generate
            inputs = self.tokenizer(user_input, return_tensors="pt").to(self.device)
            generated = self.generate_step_by_step(inputs.input_ids, max_new_tokens=100)
            
            # Decode and print only the generated part
            full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            response = full_response[len(user_input):].strip()
            print(response)
            print()
    
    def test_expert_routing(self, prompt: str):
        """Test which experts are activated for a given prompt"""
        print(f"\n🔍 Analyzing expert routing for: '{prompt}'")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Track expert activations
        expert_activations = {}
        
        def hook_fn(module, input, output):
            if hasattr(module, 'router'):
                _, indices = module.router(input[0])
                for idx in indices.flatten().tolist():
                    expert_activations[idx] = expert_activations.get(idx, 0) + 1
        
        # Register hooks
        hooks = []
        for layer in self.model.layers:
            if hasattr(layer, 'router'):
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids=inputs.input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Display results
        print("\n📊 Expert Activation Summary:")
        for expert_id, count in sorted(expert_activations.items()):
            domain = self.model.config.expert_domains.get(expert_id, "unknown")
            print(f"   Expert {expert_id} ({domain}): {count} activations")


def main():
    parser = argparse.ArgumentParser(description="Municipal MoE Model with Tokenizer")
    parser.add_argument("--model-path", type=str, default="./municipal_moe_base",
                        help="Path to the Municipal MoE model")
    parser.add_argument("--prompt", type=str, 
                        help="Text prompt for generation")
    parser.add_argument("--municipal-demo", action="store_true",
                        help="Run municipal administration demos")
    parser.add_argument("--chat", action="store_true",
                        help="Start interactive chat mode")
    parser.add_argument("--analyze-routing", type=str,
                        help="Analyze expert routing for a given prompt")
    parser.add_argument("--max-length", type=int, default=100,
                        help="Maximum length for text generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for text generation")
    
    args = parser.parse_args()
    
    # Initialize model
    integrator = MunicipalTokenizerIntegration(args.model_path)
    
    # Run requested mode
    if args.municipal_demo:
        integrator.municipal_demo()
    elif args.chat:
        integrator.interactive_chat()
    elif args.analyze_routing:
        integrator.test_expert_routing(args.analyze_routing)
    elif args.prompt:
        print(f"\n📝 Prompt: {args.prompt}")
        
        # Generate response
        inputs = integrator.tokenizer(args.prompt, return_tensors="pt").to(integrator.device)
        generated = integrator.generate_step_by_step(inputs.input_ids, max_new_tokens=args.max_length)
        
        # Decode and display
        response = integrator.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"💬 Generated: {response}")
    else:
        print("❌ Please specify --prompt, --municipal-demo, --chat, or --analyze-routing")
        parser.print_help()


if __name__ == "__main__":
    main()