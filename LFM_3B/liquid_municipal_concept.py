#!/usr/bin/env python3
"""
Liquid Foundation Model Concept für Municipal Administration
Separiert feste Strukturen von dynamischen Inhalten
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

@dataclass
class LiquidMunicipalConfig:
    """Konfiguration für Liquid Municipal Model"""
    
    # Feste Strukturen (statische Komponenten)
    fixed_patterns: List[str] = None
    fixed_vocab_size: int = 1000  # Nur für feste Muster
    
    # Dynamische Komponenten (Liquid Neural Networks)
    liquid_hidden_size: int = 512
    liquid_layers: int = 6
    
    # Hybrid-Parameter
    context_size: int = 256
    adaptation_rate: float = 0.1
    
    def __post_init__(self):
        if self.fixed_patterns is None:
            self.fixed_patterns = [
                "was kostet",
                "wie beantrage ich", 
                "wo kann ich",
                "welche unterlagen",
                "wie lange dauert",
                "wann muss ich",
                "was brauche ich",
                "wie funktioniert",
                "können sie mir sagen",
                "ich hätte gerne",
                "frage:"
            ]

class FixedPatternRecognizer(nn.Module):
    """Erkennt feste Sprachmuster ohne Training"""
    
    def __init__(self, config: LiquidMunicipalConfig):
        super().__init__()
        self.config = config
        self.patterns = config.fixed_patterns
        
        # Erstelle Pattern-Embedding (trainierbar aber stabil)
        self.pattern_embeddings = nn.Embedding(len(self.patterns), 64)
        
        # Pattern-Matcher (regelbasiert)
        self.pattern_weights = nn.Parameter(torch.ones(len(self.patterns)))
    
    def forward(self, text: str) -> Tuple[torch.Tensor, int]:
        """Erkennt Pattern im Text"""
        text_lower = text.lower()
        
        # Finde übereinstimmende Patterns
        matches = []
        for i, pattern in enumerate(self.patterns):
            if pattern in text_lower:
                matches.append((i, pattern))
        
        if matches:
            # Verwende das erste/beste Match
            pattern_id, pattern_text = matches[0]
            pattern_embedding = self.pattern_embeddings(torch.tensor(pattern_id))
            return pattern_embedding, pattern_id
        
        # Kein Pattern gefunden
        return torch.zeros(64), -1

class LiquidContentGenerator(nn.Module):
    """Liquid Neural Network für dynamische Inhalte"""
    
    def __init__(self, config: LiquidMunicipalConfig):
        super().__init__()
        self.config = config
        
        # Liquid Neuron Cells (zeitabhängig)
        self.liquid_cells = nn.ModuleList([
            LiquidCell(config.liquid_hidden_size) 
            for _ in range(config.liquid_layers)
        ])
        
        # Context-sensitive attention
        self.context_attention = nn.MultiheadAttention(
            config.liquid_hidden_size, 
            num_heads=8,
            batch_first=True
        )
        
        # Adaptive output layer
        self.output_projection = nn.Linear(config.liquid_hidden_size, config.liquid_hidden_size)
    
    def forward(self, 
                context: torch.Tensor, 
                pattern_embedding: torch.Tensor,
                adaptation_context: Optional[Dict] = None) -> torch.Tensor:
        """Generiert dynamischen Content basierend auf Pattern und Kontext"""
        
        # Kombiniere Pattern mit Context
        if pattern_embedding.dim() == 1:
            pattern_embedding = pattern_embedding.unsqueeze(0).unsqueeze(0)
        
        # Liquid processing through cells
        liquid_state = context
        for cell in self.liquid_cells:
            liquid_state = cell(liquid_state, pattern_embedding, adaptation_context)
        
        # Context-sensitive attention
        attended_state, _ = self.context_attention(
            liquid_state, liquid_state, liquid_state
        )
        
        # Adaptive output
        output = self.output_projection(attended_state)
        
        return output

class LiquidCell(nn.Module):
    """Einzelne Liquid Neural Network Zelle"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Zeitkonstanten (lernbar)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # Synaptic weights (adaptive)
        self.W_in = nn.Linear(hidden_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size) 
        self.W_pattern = nn.Linear(64, hidden_size)  # Pattern influence
        
        # Activation
        self.activation = nn.Tanh()
    
    def forward(self, 
                input_state: torch.Tensor,
                pattern_embedding: torch.Tensor,
                adaptation_context: Optional[Dict] = None) -> torch.Tensor:
        """Liquid dynamics mit Pattern-Einfluss"""
        
        # Input processing
        input_contrib = self.W_in(input_state)
        
        # Recurrent processing  
        rec_contrib = self.W_rec(input_state)
        
        # Pattern influence
        pattern_contrib = self.W_pattern(pattern_embedding)
        if pattern_contrib.shape != input_contrib.shape:
            pattern_contrib = pattern_contrib.expand_as(input_contrib)
        
        # Liquid dynamics (vereinfacht)
        combined = input_contrib + rec_contrib + pattern_contrib
        
        # Zeitbasierte Integration
        dt = 0.1  # Zeitschritt
        decay = torch.exp(-dt / (self.tau + 1e-6))
        
        new_state = decay * input_state + (1 - decay) * self.activation(combined)
        
        return new_state

class HybridMunicipalModel(nn.Module):
    """Hybrid Model: Feste Patterns + Liquid Content Generation"""
    
    def __init__(self, config: LiquidMunicipalConfig):
        super().__init__()
        self.config = config
        
        # Komponenten
        self.pattern_recognizer = FixedPatternRecognizer(config)
        self.liquid_generator = LiquidContentGenerator(config)
        
        # Input processing
        self.input_embedding = nn.Embedding(5000, config.liquid_hidden_size)  # Tokenizer vocab
        
        # Knowledge base (statisch)
        self.knowledge_base = self._create_knowledge_base()
        
        # Output layers
        self.output_layer = nn.Linear(config.liquid_hidden_size, 5000)
    
    def _create_knowledge_base(self) -> Dict[str, str]:
        """Statische Wissensbasis (keine Million Wiederholungen nötig)"""
        return {
            "personalausweis_kosten": "Ein Personalausweis kostet 37 Euro.",
            "personalausweis_gueltigkeitsdauer": "Ein Personalausweis ist 10 Jahre gültig.",
            "reisepass_kosten": "Ein Reisepass kostet 60 Euro.",
            "geburtsurkunde_kosten": "Eine Geburtsurkunde kostet 12 Euro.",
            "ummeldung_frist": "Eine Ummeldung muss innerhalb von 14 Tagen erfolgen.",
            "baugenehmigung_dauer": "Eine Baugenehmigung dauert 2-6 Monate.",
            "oeffnungszeiten_buergerbuero": "Das Bürgerbüro ist Mo-Fr 8-16 Uhr geöffnet.",
            # Erweitere nach Bedarf...
        }
    
    def forward(self, input_text: str, context: Optional[Dict] = None) -> str:
        """Hauptverarbeitung: Pattern → Knowledge → Liquid Generation"""
        
        # 1. Pattern Recognition (FEST, keine Million Wiederholungen)
        pattern_embedding, pattern_id = self.pattern_recognizer(input_text)
        
        # 2. Knowledge Lookup (STATISCH)
        knowledge_response = self._lookup_knowledge(input_text, pattern_id)
        
        # 3. Liquid Generation (DYNAMISCH, situativ anpassbar)
        if knowledge_response:
            # Basis-Antwort vorhanden, verfeinere mit Liquid
            context_tensor = self._text_to_tensor(knowledge_response)
            liquid_output = self.liquid_generator(
                context_tensor, 
                pattern_embedding, 
                context
            )
            
            # Kombiniere statisches Wissen mit dynamischer Verfeinerung
            final_response = self._combine_responses(knowledge_response, liquid_output)
        else:
            # Keine statische Antwort, vollständig liquid
            context_tensor = self._text_to_tensor(input_text)
            liquid_output = self.liquid_generator(
                context_tensor,
                pattern_embedding,
                context
            )
            final_response = self._tensor_to_text(liquid_output)
        
        return final_response
    
    def _lookup_knowledge(self, text: str, pattern_id: int) -> Optional[str]:
        """Lookup in statischer Wissensbasis"""
        text_lower = text.lower()
        
        # Direkte Keyword-Suche (regelbasiert)
        if "personalausweis" in text_lower and "kostet" in text_lower:
            return self.knowledge_base["personalausweis_kosten"]
        elif "personalausweis" in text_lower and ("gültig" in text_lower or "dauer" in text_lower):
            return self.knowledge_base["personalausweis_gueltigkeitsdauer"]
        elif "reisepass" in text_lower and "kostet" in text_lower:
            return self.knowledge_base["reisepass_kosten"]
        elif "geburtsurkunde" in text_lower and "kostet" in text_lower:
            return self.knowledge_base["geburtsurkunde_kosten"]
        elif "ummeld" in text_lower and ("frist" in text_lower or "wann" in text_lower):
            return self.knowledge_base["ummeldung_frist"]
        elif "baugenehmigung" in text_lower and ("dauer" in text_lower or "lange" in text_lower):
            return self.knowledge_base["baugenehmigung_dauer"]
        elif "öffnungszeit" in text_lower or "wann geöffnet" in text_lower:
            return self.knowledge_base["oeffnungszeiten_buergerbuero"]
        
        return None
    
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Vereinfachte Text→Tensor Konvertierung"""
        # In echt: Tokenizer verwenden
        words = text.lower().split()[:32]  # Max 32 Wörter
        
        # Dummy-Konvertierung (in echt: proper tokenization)
        token_ids = [hash(word) % 5000 for word in words]
        if len(token_ids) < 32:
            token_ids.extend([0] * (32 - len(token_ids)))
        
        tokens = torch.tensor(token_ids[:32]).unsqueeze(0)
        return self.input_embedding(tokens)
    
    def _tensor_to_text(self, tensor: torch.Tensor) -> str:
        """Vereinfachte Tensor→Text Konvertierung"""
        # In echt: Proper decoding
        return "Das kann ich Ihnen gerne beantworten..."
    
    def _combine_responses(self, static_response: str, liquid_tensor: torch.Tensor) -> str:
        """Kombiniert statische Antwort mit Liquid-Verfeinerung"""
        
        # Analyse der Liquid-Ausgabe für Verfeinerungen
        # In echt: Sophisticated combination
        
        # Beispiel: Füge Kontext hinzu
        enhanced_response = f"{static_response} "
        
        # Liquid könnte Höflichkeitsformeln, Zusatzinfos, etc. hinzufügen
        if torch.mean(liquid_tensor) > 0.5:  # Dummy-Condition
            enhanced_response += "Gerne helfe ich Ihnen bei weiteren Fragen."
        
        return enhanced_response.strip()

def test_hybrid_model():
    """Test des Hybrid-Ansatzes"""
    
    print("🧪 Teste Hybrid Municipal Model...")
    
    config = LiquidMunicipalConfig()
    model = HybridMunicipalModel(config)
    
    test_queries = [
        "Was kostet ein Personalausweis?",
        "Wie lange ist ein Personalausweis gültig?", 
        "Wo kann ich mich ummelden?",
        "Wann muss ich mich ummelden?",
        "Was kostet eine Geburtsurkunde?",
        "Wie lange dauert eine Baugenehmigung?",
        "Können Sie mir sagen, wann das Bürgerbüro geöffnet ist?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Frage: {query}")
        response = model(query)
        print(f"💬 Antwort: {response}")

if __name__ == "__main__":
    test_hybrid_model()