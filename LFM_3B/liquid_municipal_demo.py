#!/usr/bin/env python3
"""
Liquid Municipal Demo - Zeigt das Konzept ohne PyTorch Dependencies
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

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

class LiquidMunicipalDemo:
    """Demo des Liquid-Konzepts ohne ML-Dependencies"""
    
    def __init__(self):
        # Feste Patterns (KEIN TRAINING!)
        self.patterns = {
            PatternType.COST: [
                r"was kostet", r"wie teuer", r"gebühr", r"preis"
            ],
            PatternType.PROCESS: [
                r"wie beantrage ich", r"wo beantrage", r"antrag stellen"
            ],
            PatternType.LOCATION: [
                r"wo kann ich", r"wo finde ich", r"wo ist"
            ],
            PatternType.DOCUMENTS: [
                r"welche unterlagen", r"was brauche ich für", r"welche dokumente"
            ],
            PatternType.DURATION: [
                r"wie lange dauert", r"wann fertig", r"bearbeitungszeit"
            ],
            PatternType.DEADLINE: [
                r"wann muss ich", r"bis wann", r"frist für"
            ]
        }
        
        # Entities (FEST DEFINIERT)
        self.entities = {
            "personalausweis": ["personalausweis", "perso", "ausweis"],
            "reisepass": ["reisepass", "pass"],
            "geburtsurkunde": ["geburtsurkunde", "geburt"],
            "ummeldung": ["ummeldung", "umzug", "wohnsitz"],
            "baugenehmigung": ["baugenehmigung", "bauantrag"]
        }
        
        # Statische Wissensbasis (KEIN TRAINING!)
        self.knowledge = {
            ("personalausweis", PatternType.COST): "37,00 Euro (22,80 Euro für Personen unter 24)",
            ("reisepass", PatternType.COST): "60,00 Euro (92,00 Euro im Express)",
            ("geburtsurkunde", PatternType.COST): "12,00 Euro",
            ("personalausweis", PatternType.PROCESS): "Termin im Bürgerbüro vereinbaren, Unterlagen mitbringen",
            ("ummeldung", PatternType.LOCATION): "Bürgerbüro oder Bürgeramt Ihrer Stadt",
            ("personalausweis", PatternType.DOCUMENTS): "Altes Ausweisdokument, biometrisches Foto, Meldebescheinigung",
            ("baugenehmigung", PatternType.DURATION): "2-6 Monate je nach Komplexität",
            ("ummeldung", PatternType.DEADLINE): "Innerhalb von 14 Tagen nach Umzug"
        }
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Verarbeitet eine Anfrage"""
        
        # 1. Pattern Recognition (FEST, kein Training)
        pattern_type = self._match_pattern(query)
        
        # 2. Entity Recognition (FEST, kein Training)  
        entity = self._match_entity(query)
        
        # 3. Knowledge Lookup (STATISCH)
        if pattern_type and entity:
            knowledge_key = (entity, pattern_type)
            if knowledge_key in self.knowledge:
                base_response = self.knowledge[knowledge_key]
                
                # 4. Liquid Adaptation (situativ)
                adapted_response = self._adapt_response(base_response, entity, pattern_type, context)
                
                return {
                    "query": query,
                    "pattern": pattern_type.value,
                    "entity": entity,
                    "base_response": base_response,
                    "adapted_response": adapted_response,
                    "success": True
                }
        
        return {
            "query": query,
            "pattern": pattern_type.value if pattern_type else None,
            "entity": entity,
            "adapted_response": "Ich konnte Ihre Frage leider nicht verstehen.",
            "success": False
        }
    
    def _match_pattern(self, text: str) -> Optional[PatternType]:
        """Erkennt Pattern ohne Training"""
        text_lower = text.lower()
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return pattern_type
        
        return None
    
    def _match_entity(self, text: str) -> Optional[str]:
        """Erkennt Entity ohne Training"""
        text_lower = text.lower()
        
        for entity, keywords in self.entities.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return entity
        
        return None
    
    def _adapt_response(self, 
                       base_response: str, 
                       entity: str,
                       pattern_type: PatternType,
                       context: Optional[Dict] = None) -> str:
        """Liquid Adaptation - passt Antwort an Kontext an"""
        
        if context is None:
            context = {"formality": 0.5, "urgency": 0.3, "language_level": 1.0}
        
        # Templates für verschiedene Pattern-Typen
        templates = {
            PatternType.COST: {
                "formal": f"Die Gebühr für {entity} beträgt {base_response}.",
                "informal": f"{entity.title()} kostet {base_response}.",
                "urgent": f"WICHTIG: {entity.title()} = {base_response}!"
            },
            PatternType.PROCESS: {
                "formal": f"Für {entity} ist folgender Ablauf vorgesehen: {base_response}",
                "informal": f"{entity.title()}: {base_response}",
                "urgent": f"DRINGEND für {entity}: {base_response}"
            },
            PatternType.DEADLINE: {
                "formal": f"Die Frist für {entity}: {base_response}",
                "informal": f"{entity.title()} - {base_response}",
                "urgent": f"ACHTUNG FRIST! {entity}: {base_response}"
            }
        }
        
        # Wähle Template basierend auf Kontext
        pattern_templates = templates.get(pattern_type, {})
        
        if context.get("urgency", 0) > 0.7:
            template_key = "urgent"
        elif context.get("formality", 0.5) > 0.7:
            template_key = "formal"
        else:
            template_key = "informal"
        
        if template_key in pattern_templates:
            response = pattern_templates[template_key]
        else:
            # Fallback
            response = f"{entity.title()}: {base_response}"
        
        # Weitere Anpassungen basierend auf Sprachlevel
        if context.get("language_level", 1.0) < 0.5:
            # Vereinfache für niedrigen Sprachlevel
            response = response.replace("Gebühr", "Preis")
            response = response.replace("beträgt", "ist")
            response = response.replace("vorgesehen", "nötig")
        
        return response

def demonstrate_liquid_advantage():
    """Zeigt die Vorteile des Liquid-Ansatzes"""
    
    print("🌊 LIQUID FOUNDATION MODEL - Municipal Demo")
    print("=" * 60)
    
    demo = LiquidMunicipalDemo()
    
    # Verschiedene Anfragen mit gleichem Inhalt aber verschiedenen Kontexten
    test_query = "Was kostet ein Personalausweis?"
    
    contexts = [
        {
            "name": "Normal",
            "context": {"formality": 0.5, "urgency": 0.3, "language_level": 1.0}
        },
        {
            "name": "Formal",
            "context": {"formality": 0.9, "urgency": 0.3, "language_level": 1.0}
        },
        {
            "name": "Dringend",
            "context": {"formality": 0.5, "urgency": 0.9, "language_level": 1.0}
        },
        {
            "name": "Einfache Sprache",
            "context": {"formality": 0.3, "urgency": 0.3, "language_level": 0.3}
        }
    ]
    
    print(f"\n🔍 Gleiche Frage, verschiedene Kontexte:")
    print(f"Frage: {test_query}\n")
    
    for ctx in contexts:
        result = demo.process_query(test_query, ctx["context"])
        print(f"📌 Kontext: {ctx['name']}")
        print(f"   → {result['adapted_response']}\n")
    
    # Weitere Beispiele
    print("\n🔍 Verschiedene Fragen:")
    
    more_queries = [
        ("Wo kann ich mich ummelden?", None),
        ("Wann muss ich mich ummelden?", {"urgency": 0.8}),
        ("Wie lange dauert eine Baugenehmigung?", {"formality": 0.8}),
        ("Welche Unterlagen brauche ich für einen Personalausweis?", {"language_level": 0.4})
    ]
    
    for query, context in more_queries:
        result = demo.process_query(query, context)
        print(f"\n❓ {query}")
        if result["success"]:
            print(f"✅ Pattern: {result['pattern']} | Entity: {result['entity']}")
            print(f"💬 {result['adapted_response']}")
        else:
            print(f"❌ {result['adapted_response']}")
    
    # Zeige den Unterschied
    print("\n\n🎯 DER GROSSE UNTERSCHIED:")
    print("=" * 60)
    
    print("\n❌ TRANSFORMER (Alte Methode):")
    print("- Muss 'Was kostet' 1 Million mal trainieren")
    print("- Muss 'Personalausweis = 37 Euro' auswendig lernen")
    print("- Kann sich NICHT an Kontext anpassen")
    print("- Braucht RIESIGE Datenmengen")
    
    print("\n✅ LIQUID FOUNDATION MODEL (Neue Methode):")
    print("- Patterns sind FEST DEFINIERT (kein Training!)")
    print("- Wissensbasis ist STATISCH (Update zur Laufzeit!)")
    print("- Liquid Adapter passt sich AN KONTEXT an")
    print("- Braucht NUR Training für Stil-Anpassung")
    
    print("\n💡 BEISPIEL - Neues Wissen hinzufügen:")
    print("Transformer: Muss komplett neu trainiert werden")
    print("Liquid: Einfach zur Wissensbasis hinzufügen!\n")
    
    # Füge neues Wissen hinzu
    demo.knowledge[("hundesteuer", PatternType.COST)] = "120 Euro pro Jahr"
    demo.entities["hundesteuer"] = ["hundesteuer", "hund"]
    
    result = demo.process_query("Was kostet die Hundesteuer?")
    print(f"NEU: {result['query']}")
    print(f"→ {result['adapted_response']}")
    print("\n✨ Kein Training nötig! Sofort verfügbar!")

if __name__ == "__main__":
    demonstrate_liquid_advantage()