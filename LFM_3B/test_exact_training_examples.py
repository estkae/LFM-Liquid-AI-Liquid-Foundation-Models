#!/usr/bin/env python3
"""
Test model with exact training examples to see if it learned correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LFM_3B.municipal_tokenizer_integration import MunicipalTokenizerIntegration

def test_exact_examples():
    """Test with examples that were in the training data"""
    
    print("🧪 Testing with exact training examples...")
    
    # Load model
    integrator = MunicipalTokenizerIntegration("./municipal_moe_trained_v2/best_model")
    
    # Test prompts that were in training data
    test_prompts = [
        "Frage: Ich möchte meinen Wohnsitz ummelden. Was muss ich tun?\nAntwort:",
        "Frage: Wie beantrage ich eine Meldebescheinigung?\nAntwort:", 
        "Frage: Ich möchte einen Wintergarten anbauen. Brauche ich eine Genehmigung?\nAntwort:",
        "Frage: Wie beantrage ich eine Geburtsurkunde?\nAntwort:",
        "Frage: Ich möchte mich über Lärmbelästigung beschweren. An wen wende ich mich?\nAntwort:",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/5:")
        print(f"Prompt: {prompt.split('Antwort:')[0]}Antwort:")
        
        # Generate
        inputs = integrator.tokenizer(prompt, return_tensors="pt").to(integrator.device)
        generated = integrator.generate_step_by_step(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.5,  # More focused
            top_p=0.8
        )
        
        # Decode
        full_response = integrator.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract answer part
        if "Antwort:" in full_response:
            answer_part = full_response.split("Antwort:")[-1].strip()
            print(f"💬 Generated Answer: {answer_part}")
        else:
            print(f"💬 Full Response: {full_response}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_exact_examples()