#!/usr/bin/env python3
"""
Vollständige Liquid Municipal Implementation
Trennt feste Strukturen von dynamischen Inhalten - keine Million Wiederholungen mehr!
"""

import torch
import torch.nn as nn
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

class PatternType(Enum):
    """Typen von Fragemustern"""
    COST = "cost"               # Was kostet...
    PROCESS = "process"         # Wie beantrage ich...
    LOCATION = "location"       # Wo kann ich...
    DOCUMENTS = "documents"     # Welche Unterlagen...
    DURATION = "duration"       # Wie lange dauert...
    DEADLINE = "deadline"       # Wann muss ich...
    REQUIREMENTS = "requirements" # Was brauche ich...
    FUNCTION = "function"       # Wie funktioniert...

@dataclass
class KnowledgeEntry:
    """Eintrag in der Wissensbasis"""
    entity: str
    pattern_type: PatternType
    value: str
    context_hints: Dict[str, str] = None

class PatternMatcher:
    """Erkennt feste Sprachmuster ohne Training"""
    
    def __init__(self):
        self.patterns = {
            PatternType.COST: [
                r"was kostet", r"wie teuer", r"gebühr", r"preis", 
                r"wie viel", r"kosten für"
            ],
            PatternType.PROCESS: [
                r"wie beantrage ich", r"wo beantrage", r"antrag stellen",
                r"wie kann ich.*beantragen", r"beantragung von"
            ],
            PatternType.LOCATION: [
                r"wo kann ich", r"wo finde ich", r"wo ist", 
                r"wohin muss ich", r"an welcher stelle"
            ],
            PatternType.DOCUMENTS: [
                r"welche unterlagen", r"was brauche ich für", 
                r"welche dokumente", r"was muss ich mitbringen"
            ],
            PatternType.DURATION: [
                r"wie lange dauert", r"wann fertig", r"bearbeitungszeit",
                r"wie schnell", r"dauer der"
            ],
            PatternType.DEADLINE: [
                r"wann muss ich", r"bis wann", r"frist für",
                r"innerhalb welcher zeit", r"deadline"
            ],
            PatternType.REQUIREMENTS: [
                r"was brauche ich", r"voraussetzungen", r"bedingungen",
                r"was benötige ich", r"was ist nötig"
            ],
            PatternType.FUNCTION: [
                r"wie funktioniert", r"wie läuft.*ab", r"ablauf",
                r"prozess für", r"wie geht"
            ]
        }
        
        # Entities erkennen
        self.entities = {
            "personalausweis": ["personalausweis", "perso", "ausweis"],
            "reisepass": ["reisepass", "pass"],
            "führerschein": ["führerschein", "fahrerlaubnis"],
            "geburtsurkunde": ["geburtsurkunde", "geburt"],
            "heiratsurkunde": ["heiratsurkunde", "heirat", "eheurkunde"],
            "ummeldung": ["ummeldung", "umzug", "wohnsitz"],
            "anmeldung": ["anmeldung", "erstanmeldung"],
            "baugenehmigung": ["baugenehmigung", "bauantrag"],
            "gewerbeanmeldung": ["gewerbeanmeldung", "gewerbe"],
            "kindergeld": ["kindergeld"],
            "elterngeld": ["elterngeld"],
            "wohngeld": ["wohngeld"],
            "bürgerbüro": ["bürgerbüro", "bürgeramt", "rathaus"]
        }
    
    def match(self, text: str) -> Tuple[Optional[PatternType], Optional[str], float]:
        """Matched Pattern und Entity mit Confidence"""
        text_lower = text.lower()
        
        # Pattern matching
        best_pattern = None
        best_pattern_score = 0.0
        
        for pattern_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text_lower):
                    # Längere Patterns bekommen höhere Scores
                    score = len(pattern) / 20.0
                    if score > best_pattern_score:
                        best_pattern = pattern_type
                        best_pattern_score = score
        
        # Entity matching
        best_entity = None
        best_entity_score = 0.0
        
        for entity, keywords in self.entities.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score = len(keyword) / 15.0
                    if score > best_entity_score:
                        best_entity = entity
                        best_entity_score = score
        
        # Combined confidence
        confidence = (best_pattern_score + best_entity_score) / 2.0
        
        return best_pattern, best_entity, confidence

class StaticKnowledgeBase:
    """Statische Wissensbasis - keine Wiederholungen im Training nötig!"""
    
    def __init__(self):
        self.knowledge = {
            # Kosten
            ("personalausweis", PatternType.COST): KnowledgeEntry(
                entity="personalausweis",
                pattern_type=PatternType.COST,
                value="37,00 Euro",
                context_hints={"reduced": "22,80 Euro für Personen unter 24 Jahren"}
            ),
            ("reisepass", PatternType.COST): KnowledgeEntry(
                entity="reisepass",
                pattern_type=PatternType.COST,
                value="60,00 Euro",
                context_hints={"express": "92,00 Euro im Expressverfahren"}
            ),
            ("geburtsurkunde", PatternType.COST): KnowledgeEntry(
                entity="geburtsurkunde",
                pattern_type=PatternType.COST,
                value="12,00 Euro",
                context_hints={"international": "15,00 Euro für internationale Urkunde"}
            ),
            
            # Prozesse
            ("personalausweis", PatternType.PROCESS): KnowledgeEntry(
                entity="personalausweis",
                pattern_type=PatternType.PROCESS,
                value="Termin im Bürgerbüro vereinbaren, Unterlagen mitbringen, Antrag stellen",
                context_hints={"online": "Vorausgefülltes Formular online verfügbar"}
            ),
            
            # Orte
            ("ummeldung", PatternType.LOCATION): KnowledgeEntry(
                entity="ummeldung",
                pattern_type=PatternType.LOCATION,
                value="Bürgerbüro oder Bürgeramt Ihrer Stadt",
                context_hints={"online": "Teilweise auch online möglich"}
            ),
            
            # Dokumente
            ("personalausweis", PatternType.DOCUMENTS): KnowledgeEntry(
                entity="personalausweis",
                pattern_type=PatternType.DOCUMENTS,
                value="Altes Ausweisdokument, biometrisches Passfoto, Meldebescheinigung",
                context_hints={"erstantrag": "Bei Erstantrag zusätzlich Geburtsurkunde"}
            ),
            
            # Dauer
            ("baugenehmigung", PatternType.DURATION): KnowledgeEntry(
                entity="baugenehmigung",
                pattern_type=PatternType.DURATION,
                value="2-6 Monate je nach Komplexität",
                context_hints={"simple": "Einfache Bauvorhaben: 4-8 Wochen"}
            ),
            
            # Fristen
            ("ummeldung", PatternType.DEADLINE): KnowledgeEntry(
                entity="ummeldung",
                pattern_type=PatternType.DEADLINE,
                value="Innerhalb von 14 Tagen nach Umzug",
                context_hints={"penalty": "Bei Verspätung droht Bußgeld"}
            )
        }
    
    def lookup(self, entity: str, pattern_type: PatternType) -> Optional[KnowledgeEntry]:
        """Direkter Lookup ohne Training"""
        return self.knowledge.get((entity, pattern_type))
    
    def add_entry(self, entry: KnowledgeEntry):
        """Neue Einträge zur Laufzeit hinzufügen"""
        self.knowledge[(entry.entity, entry.pattern_type)] = entry

class LiquidContextAdapter(nn.Module):
    """Liquid Neural Network für situative Anpassung"""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Kontext-Encoder
        self.context_encoder = nn.ModuleDict({
            "user_age": nn.Linear(1, 32),
            "user_language_level": nn.Linear(1, 32),
            "urgency": nn.Linear(1, 32),
            "time_of_day": nn.Linear(1, 32),
            "previous_interactions": nn.Linear(1, 32)
        })
        
        # Liquid cells für Adaptation
        self.liquid_cell = nn.GRUCell(160, hidden_size)  # 5 * 32 = 160
        
        # Stil-Modifikatoren
        self.style_modifiers = nn.ModuleDict({
            "formality": nn.Linear(hidden_size, 64),
            "detail_level": nn.Linear(hidden_size, 64),
            "empathy": nn.Linear(hidden_size, 64),
            "simplicity": nn.Linear(hidden_size, 64)
        })
        
        # Output formatter
        self.output_formatter = nn.Linear(256, hidden_size)
    
    def forward(self, base_response: str, context: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Passt Antwort an Kontext an"""
        
        # Encode context
        context_vectors = []
        for key, encoder in self.context_encoder.items():
            value = context.get(key, 0.5)  # Default 0.5
            context_vectors.append(encoder(torch.tensor([value])))
        
        context_tensor = torch.cat(context_vectors, dim=-1)
        
        # Liquid processing
        hidden = torch.zeros(1, self.hidden_size)
        hidden = self.liquid_cell(context_tensor.unsqueeze(0), hidden)
        
        # Compute style modifications
        style_scores = {}
        for style, modifier in self.style_modifiers.items():
            score = torch.sigmoid(modifier(hidden)).mean().item()
            style_scores[style] = score
        
        # Apply modifications to response
        adapted_response = self._apply_style_modifications(base_response, style_scores, context)
        
        return adapted_response, style_scores
    
    def _apply_style_modifications(self, 
                                   response: str, 
                                   style_scores: Dict[str, float],
                                   context: Dict[str, float]) -> str:
        """Wendet Stil-Modifikationen an"""
        
        # Formalität
        if style_scores["formality"] > 0.7:
            response = response.replace("Hi", "Guten Tag")
            response = response.replace("Euro", "Euro (Gebühr)")
        elif style_scores["formality"] < 0.3:
            response = "Hey! " + response
        
        # Detailgrad
        if style_scores["detail_level"] > 0.7:
            response += " Weitere Details finden Sie auf unserer Webseite."
        elif style_scores["detail_level"] < 0.3:
            # Kürze Antwort
            sentences = response.split(".")
            response = sentences[0] + "."
        
        # Empathie
        if style_scores["empathy"] > 0.7:
            response = "Gerne helfe ich Ihnen weiter. " + response
        
        # Einfachheit (für niedrigen Sprachlevel)
        if context.get("user_language_level", 1.0) < 0.5:
            response = self._simplify_german(response)
        
        # Dringlichkeit
        if context.get("urgency", 0.0) > 0.8:
            response = "WICHTIG: " + response + " Bitte beachten Sie die Fristen!"
        
        return response
    
    def _simplify_german(self, text: str) -> str:
        """Vereinfacht deutsches Amtsdeutsch"""
        simplifications = {
            "Gebühr": "Preis",
            "Antragstellung": "Antrag machen",
            "Personalausweis": "Ausweis",
            "Bürgerbüro": "Amt",
            "vereinbaren": "machen",
            "Dokument": "Papier"
        }
        
        for complex_word, simple_word in simplifications.items():
            text = text.replace(complex_word, simple_word)
        
        return text

class LiquidMunicipalSystem:
    """Komplettes Liquid Municipal System"""
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.knowledge_base = StaticKnowledgeBase()
        self.context_adapter = LiquidContextAdapter()
        
        # Conversation memory
        self.conversation_history = []
        self.user_profile = {
            "interactions": 0,
            "language_level": 1.0,
            "preferred_formality": 0.5
        }
    
    def process_query(self, 
                      query: str, 
                      context: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Hauptverarbeitung einer Anfrage"""
        
        # Default context
        if context is None:
            context = self._create_default_context()
        
        # 1. Pattern & Entity Recognition (FEST, kein Training)
        pattern_type, entity, confidence = self.pattern_matcher.match(query)
        
        response_data = {
            "query": query,
            "pattern_type": pattern_type.value if pattern_type else None,
            "entity": entity,
            "confidence": confidence,
            "response": None,
            "style_scores": None
        }
        
        # 2. Knowledge Lookup (STATISCH, kein Training)
        if pattern_type and entity:
            knowledge_entry = self.knowledge_base.lookup(entity, pattern_type)
            
            if knowledge_entry:
                base_response = self._format_base_response(knowledge_entry, pattern_type)
                
                # 3. Liquid Adaptation (DYNAMISCH, situativ)
                adapted_response, style_scores = self.context_adapter(base_response, context)
                
                response_data["response"] = adapted_response
                response_data["style_scores"] = style_scores
            else:
                # Fallback für unbekannte Kombinationen
                response_data["response"] = self._generate_fallback(pattern_type, entity)
        else:
            # Keine Pattern/Entity erkannt
            response_data["response"] = "Entschuldigung, ich konnte Ihre Frage nicht richtig verstehen. Können Sie sie anders formulieren?"
        
        # Update conversation history
        self.conversation_history.append(response_data)
        self.user_profile["interactions"] += 1
        
        return response_data
    
    def _create_default_context(self) -> Dict[str, float]:
        """Erstellt Standard-Kontext"""
        import datetime
        
        hour = datetime.datetime.now().hour
        time_score = 0.5 if 9 <= hour <= 17 else 0.2  # Geschäftszeiten
        
        return {
            "user_age": 0.5,  # Unbekannt
            "user_language_level": self.user_profile["language_level"],
            "urgency": 0.3,  # Normal
            "time_of_day": time_score,
            "previous_interactions": min(self.user_profile["interactions"] / 10, 1.0)
        }
    
    def _format_base_response(self, 
                             knowledge_entry: KnowledgeEntry, 
                             pattern_type: PatternType) -> str:
        """Formatiert Basis-Antwort aus Knowledge Entry"""
        
        templates = {
            PatternType.COST: "Ein {entity} kostet {value}.",
            PatternType.PROCESS: "Für {entity}: {value}",
            PatternType.LOCATION: "{entity} können Sie hier erledigen: {value}",
            PatternType.DOCUMENTS: "Für {entity} benötigen Sie: {value}",
            PatternType.DURATION: "{entity}: {value}",
            PatternType.DEADLINE: "{entity}: {value}",
            PatternType.REQUIREMENTS: "Für {entity} brauchen Sie: {value}",
            PatternType.FUNCTION: "{entity} funktioniert so: {value}"
        }
        
        template = templates.get(pattern_type, "{entity}: {value}")
        
        # Entity-Namen verschönern
        entity_display = knowledge_entry.entity.replace("_", " ").title()
        
        return template.format(
            entity=entity_display,
            value=knowledge_entry.value
        )
    
    def _generate_fallback(self, pattern_type: PatternType, entity: str) -> str:
        """Generiert Fallback-Antwort"""
        
        if pattern_type and entity:
            return f"Leider habe ich keine spezifischen Informationen über {pattern_type.value} für {entity}. Bitte wenden Sie sich an das Bürgerbüro."
        else:
            return "Ich konnte Ihre Frage nicht vollständig verstehen. Bitte formulieren Sie sie um."
    
    def add_knowledge(self, entity: str, pattern_type: PatternType, value: str, context_hints: Dict = None):
        """Fügt neues Wissen zur Laufzeit hinzu"""
        entry = KnowledgeEntry(
            entity=entity,
            pattern_type=pattern_type,
            value=value,
            context_hints=context_hints or {}
        )
        self.knowledge_base.add_entry(entry)
        print(f"✅ Neues Wissen hinzugefügt: {entity} - {pattern_type.value}")

def demo_liquid_system():
    """Demo des Liquid Municipal Systems"""
    
    print("🚀 Liquid Municipal System Demo")
    print("=" * 50)
    
    system = LiquidMunicipalSystem()
    
    # Test-Anfragen mit verschiedenen Kontexten
    test_cases = [
        {
            "query": "Was kostet ein Personalausweis?",
            "context": {"user_age": 0.2, "user_language_level": 1.0}  # Junger Erwachsener
        },
        {
            "query": "Was kostet ein Personalausweis?",
            "context": {"user_age": 0.8, "urgency": 0.9}  # Ältere Person, dringend
        },
        {
            "query": "Wie beantrage ich einen Reisepass?",
            "context": {"user_language_level": 0.3}  # Niedriger Sprachlevel
        },
        {
            "query": "Wo kann ich mich ummelden?",
            "context": {"time_of_day": 0.9, "previous_interactions": 0.0}  # Abends, neuer Nutzer
        },
        {
            "query": "Welche Unterlagen brauche ich für einen Personalausweis?",
            "context": {"urgency": 0.1, "user_language_level": 0.5}  # Keine Eile, mittlerer Sprachlevel
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}:")
        print(f"Frage: {test['query']}")
        print(f"Kontext: {test['context']}")
        
        result = system.process_query(test['query'], test['context'])
        
        print(f"Erkannt: {result['pattern_type']} + {result['entity']} (Confidence: {result['confidence']:.2f})")
        print(f"Antwort: {result['response']}")
        
        if result['style_scores']:
            print("Stil-Anpassungen:")
            for style, score in result['style_scores'].items():
                print(f"  - {style}: {score:.2f}")
    
    # Zeige dass kein Training nötig ist
    print("\n\n✨ WICHTIG: Keine Million Wiederholungen nötig!")
    print("- Patterns sind fest definiert (kein Training)")
    print("- Wissensbasis ist statisch (kein Training)")
    print("- Nur Liquid Adapter wird trainiert (für Stil-Anpassung)")
    
    # Neues Wissen zur Laufzeit hinzufügen
    print("\n\n➕ Füge neues Wissen hinzu (zur Laufzeit!):")
    system.add_knowledge(
        entity="hundesteuer",
        pattern_type=PatternType.COST,
        value="120 Euro pro Jahr für den ersten Hund"
    )
    
    # Teste neues Wissen
    result = system.process_query("Was kostet die Hundesteuer?")
    print(f"Neue Anfrage: Was kostet die Hundesteuer?")
    print(f"Antwort: {result['response']}")

if __name__ == "__main__":
    demo_liquid_system()