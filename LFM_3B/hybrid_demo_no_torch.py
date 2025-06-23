#!/usr/bin/env python3
"""
Hybrid LFM + Liquid Demo (ohne PyTorch Dependencies)
Zeigt das Konzept der Hybrid-Architektur
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class QueryType(Enum):
    """Typen von Anfragen"""
    MUNICIPAL_SPECIFIC = "municipal"  # Spezifische Municipal-Fragen → Liquid
    GENERAL_KNOWLEDGE = "general"     # Allgemeine Fragen → LFM_3B
    HYBRID = "hybrid"                 # Kombination aus beiden

@dataclass
class HybridQueryResult:
    """Ergebnis der Hybrid-Verarbeitung"""
    query_id: str
    original_query: str
    query_type: QueryType
    municipal_pattern: Optional[str]
    municipal_entity: Optional[str]
    liquid_response: Optional[str]
    lfm_response: Optional[str]
    final_response: str
    confidence: float
    processing_time_ms: float
    components_used: List[str]
    success: bool

class QueryRouter:
    """Entscheidet ob Municipal-Liquid oder allgemeines LFM verwendet wird"""
    
    def __init__(self):
        # Municipal-spezifische Keywords
        self.municipal_keywords = {
            'dokumente': ['personalausweis', 'reisepass', 'führerschein', 'geburtsurkunde', 'heiratsurkunde'],
            'prozesse': ['ummeldung', 'anmeldung', 'baugenehmigung', 'gewerbeanmeldung'],
            'ämter': ['bürgerbüro', 'bürgeramt', 'rathaus', 'standesamt', 'ordnungsamt'],
            'gebühren': ['kosten', 'gebühr', 'preis', 'steuer'],
            'fristen': ['frist', 'deadline', 'wann muss', 'bis wann'],
            'öffnungszeiten': ['öffnungszeit', 'sprechzeit', 'wann geöffnet']
        }
        
        # Allgemeine Wissens-Keywords (für LFM_3B)
        self.general_keywords = {
            'wissenschaft': ['physik', 'chemie', 'biologie', 'mathematik', 'einstein', 'newton'],
            'geschichte': ['weltkrieg', 'mittelalter', 'antike', 'revolution', 'napoleon', 'caesar'],
            'geographie': ['hauptstadt', 'kontinent', 'ozean', 'berg', 'frankreich', 'deutschland'],
            'kultur': ['literatur', 'musik', 'kunst', 'film', 'shakespeare', 'mozart'],
            'technologie': ['computer', 'internet', 'ki', 'roboter', 'algorithmus'],
            'gesellschaft': ['politik', 'wirtschaft', 'philosophie', 'religion']
        }
    
    def route_query(self, query: str) -> Tuple[QueryType, float, Dict[str, List[str]]]:
        """Entscheidet welcher Ansatz verwendet werden soll"""
        query_lower = query.lower()
        
        # Score für Municipal-Relevanz
        municipal_matches = []
        municipal_score = 0.0
        for category, keywords in self.municipal_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    municipal_matches.append(f"{category}:{keyword}")
                    municipal_score += 1.0
        
        # Score für allgemeines Wissen
        general_matches = []
        general_score = 0.0
        for category, keywords in self.general_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    general_matches.append(f"{category}:{keyword}")
                    general_score += 1.0
        
        # Entscheidungslogik
        matches_info = {
            "municipal": municipal_matches,
            "general": general_matches
        }
        
        if municipal_score > general_score and municipal_score >= 1.0:
            return QueryType.MUNICIPAL_SPECIFIC, municipal_score / (municipal_score + general_score + 1), matches_info
        elif general_score > municipal_score and general_score >= 1.0:
            return QueryType.GENERAL_KNOWLEDGE, general_score / (municipal_score + general_score + 1), matches_info
        elif municipal_score > 0 and general_score > 0:
            return QueryType.HYBRID, 0.5, matches_info
        else:
            # Default: Verwende LFM für unklare Fragen
            return QueryType.GENERAL_KNOWLEDGE, 0.3, matches_info

class MockLFMWrapper:
    """Mock LFM_3B Model für Demo"""
    
    def __init__(self):
        # Simulierte LFM-Antworten für allgemeine Fragen
        self.knowledge_base = {
            "einstein": "Albert Einstein (1879-1955) war ein deutscher Physiker, der die Relativitätstheorie entwickelte und für seine Beiträge zur theoretischen Physik bekannt ist.",
            "frankreich": "Frankreich ist ein Land in Westeuropa mit der Hauptstadt Paris. Es ist bekannt für seine Kultur, Küche und Geschichte.",
            "photosynthese": "Photosynthese ist der Prozess, bei dem Pflanzen Lichtenergie nutzen, um Kohlendioxid und Wasser in Glukose und Sauerstoff umzuwandeln.",
            "computer": "Ein Computer ist eine elektronische Maschine, die Daten verarbeitet und Programme ausführt. Moderne Computer basieren auf digitaler Technologie.",
            "weltkrieg": "Der Erste Weltkrieg (1914-1918) war ein globaler Konflikt, der hauptsächlich in Europa stattfand und tiefgreifende politische Veränderungen zur Folge hatte."
        }
    
    def generate_response(self, query: str) -> Optional[str]:
        """Simuliert LFM_3B Generierung"""
        query_lower = query.lower()
        
        # Suche nach bekannten Begriffen
        for key, response in self.knowledge_base.items():
            if key in query_lower:
                return response
        
        # Fallback für unbekannte Anfragen
        if any(word in query_lower for word in ['was', 'wer', 'wie', 'wo', 'wann', 'warum']):
            return "Das ist eine interessante Frage. Basierend auf meinem Training kann ich dazu verschiedene Aspekte erläutern, aber für spezifische Details empfehle ich weitere Recherche."
        
        return None

class LiquidMunicipalProcessor:
    """Liquid Municipal Processing (vereinfacht)"""
    
    def __init__(self):
        # Municipal Knowledge Base
        self.knowledge = {
            ("personalausweis", "cost"): "37,00 Euro (22,80 Euro für Personen unter 24 Jahren)",
            ("reisepass", "cost"): "60,00 Euro (92,00 Euro im Expressverfahren)",
            ("geburtsurkunde", "cost"): "12,00 Euro",
            ("ummeldung", "deadline"): "Innerhalb von 14 Tagen nach Umzug",
            ("personalausweis", "duration"): "4-6 Wochen (Express: 3 Werktage)",
            ("baugenehmigung", "duration"): "2-6 Monate je nach Komplexität",
            ("bürgerbüro", "hours"): "Mo-Fr: 8:00-16:00 Uhr, Do: 8:00-18:00 Uhr",
            ("personalausweis", "documents"): "Altes Ausweisdokument, biometrisches Passfoto, Meldebescheinigung",
            ("ummeldung", "location"): "Bürgerbüro oder Bürgeramt Ihrer Stadt",
        }
        
        # Pattern Recognition
        self.patterns = {
            "cost": ["was kostet", "wie teuer", "gebühr", "preis"],
            "process": ["wie beantrage", "wo beantrage", "antrag"],
            "location": ["wo kann ich", "wo finde ich", "wo ist"],
            "documents": ["welche unterlagen", "was brauche ich", "dokumente"],
            "duration": ["wie lange dauert", "bearbeitungszeit", "dauer"],
            "deadline": ["wann muss ich", "bis wann", "frist"],
            "hours": ["öffnungszeit", "wann geöffnet", "sprechzeit"]
        }
        
        # Entity Recognition
        self.entities = {
            "personalausweis": ["personalausweis", "perso", "ausweis"],
            "reisepass": ["reisepass", "pass"],
            "geburtsurkunde": ["geburtsurkunde"],
            "ummeldung": ["ummeldung", "umzug"],
            "baugenehmigung": ["baugenehmigung", "bauantrag"],
            "bürgerbüro": ["bürgerbüro", "bürgeramt", "rathaus"]
        }
    
    def process(self, query: str, context: Dict = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Verarbeitet Municipal Query"""
        query_lower = query.lower()
        
        # Pattern erkennen
        pattern = None
        for p, keywords in self.patterns.items():
            if any(kw in query_lower for kw in keywords):
                pattern = p
                break
        
        # Entity erkennen
        entity = None
        for e, keywords in self.entities.items():
            if any(kw in query_lower for kw in keywords):
                entity = e
                break
        
        # Knowledge Lookup
        if pattern and entity:
            key = (entity, pattern)
            if key in self.knowledge:
                base_response = self.knowledge[key]
                
                # Einfache Kontext-Anpassung
                if context and context.get("formality", 0.5) > 0.7:
                    adapted = f"Die {pattern} für {entity} beträgt: {base_response}"
                elif context and context.get("urgency", 0.0) > 0.7:
                    adapted = f"WICHTIG: {entity} - {base_response}!"
                else:
                    adapted = f"{entity.title()}: {base_response}"
                
                return pattern, entity, adapted
        
        return pattern, entity, None

class HybridLiquidDemo:
    """Demo der Hybrid LFM + Liquid Pipeline"""
    
    def __init__(self):
        self.query_router = QueryRouter()
        self.lfm_wrapper = MockLFMWrapper()
        self.liquid_processor = LiquidMunicipalProcessor()
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> HybridQueryResult:
        """Hauptverarbeitung mit Hybrid-Ansatz"""
        
        start_time = datetime.now()
        query_id = self._generate_query_id(query)
        
        if context is None:
            context = {"formality": 0.5, "urgency": 0.3}
        
        # 1. Query Routing
        query_type, routing_confidence, matches = self.query_router.route_query(query)
        
        components_used = []
        liquid_response = None
        lfm_response = None
        municipal_pattern = None
        municipal_entity = None
        
        if query_type == QueryType.MUNICIPAL_SPECIFIC:
            # Verwende Liquid für Municipal-spezifische Fragen
            components_used.append("Liquid Municipal")
            
            pattern, entity, response = self.liquid_processor.process(query, context)
            municipal_pattern = pattern
            municipal_entity = entity
            liquid_response = response
            
            # Fallback zu LFM wenn kein Municipal-Wissen
            if not liquid_response:
                components_used.append("LFM Fallback")
                lfm_response = self.lfm_wrapper.generate_response(query)
            
            final_response = liquid_response or lfm_response or "Keine spezifischen Municipal-Informationen verfügbar."
            confidence = routing_confidence
            
        elif query_type == QueryType.GENERAL_KNOWLEDGE:
            # Verwende LFM für allgemeine Fragen
            components_used.append("LFM General")
            
            lfm_response = self.lfm_wrapper.generate_response(query)
            final_response = lfm_response or "Ich kann diese allgemeine Frage leider nicht beantworten."
            confidence = routing_confidence
            
        else:  # HYBRID
            # Verwende beide Ansätze und kombiniere
            components_used.extend(["Liquid Municipal", "LFM General"])
            
            # Liquid-Versuch
            pattern, entity, liquid_resp = self.liquid_processor.process(query, context)
            municipal_pattern = pattern
            municipal_entity = entity
            liquid_response = liquid_resp
            
            # LFM-Versuch
            lfm_response = self.lfm_wrapper.generate_response(query)
            
            # Kombiniere beide Antworten
            final_response = self._combine_responses(liquid_response, lfm_response, query)
            confidence = 0.7
        
        # Berechne Verarbeitungszeit
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return HybridQueryResult(
            query_id=query_id,
            original_query=query,
            query_type=query_type,
            municipal_pattern=municipal_pattern,
            municipal_entity=municipal_entity,
            liquid_response=liquid_response,
            lfm_response=lfm_response,
            final_response=final_response,
            confidence=confidence,
            processing_time_ms=processing_time,
            components_used=components_used,
            success=bool(final_response)
        )
    
    def _combine_responses(self, liquid_response: Optional[str], lfm_response: Optional[str], query: str) -> str:
        """Kombiniert Liquid- und LFM-Antworten intelligente"""
        
        if liquid_response and lfm_response:
            # Beide Antworten verfügbar
            return f"{liquid_response} Zusätzliche Information: {lfm_response}"
        elif liquid_response:
            return liquid_response
        elif lfm_response:
            return lfm_response
        else:
            return "Leider konnte ich keine passende Antwort finden."
    
    def _generate_query_id(self, query: str) -> str:
        """Generiert Query ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        hash_input = f"{query}{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

def demonstrate_hybrid_architecture():
    """Demo der Hybrid-Architektur"""
    
    print("🌊🤖 Hybrid LFM + Liquid Architecture Demo")
    print("=" * 60)
    
    demo = HybridLiquidDemo()
    
    # Test-Anfragen verschiedener Typen
    test_queries = [
        # Municipal-spezifisch (sollte Liquid verwenden)
        ("Was kostet ein Personalausweis?", {"formality": 0.5}),
        ("Wo kann ich mich ummelden?", {"formality": 0.8}),
        ("Wie lange dauert eine Baugenehmigung?", {"urgency": 0.9}),
        
        # Allgemeines Wissen (sollte LFM verwenden)
        ("Wer war Albert Einstein?", {"formality": 0.5}),
        ("Was ist die Hauptstadt von Frankreich?", {"formality": 0.3}),
        ("Wie funktioniert Photosynthese?", {"formality": 0.7}),
        
        # Hybrid (könnte beide verwenden)
        ("Brauche ich einen Personalausweis für Reisen nach Frankreich?", {"formality": 0.5}),
        ("Was kostet ein Computer und wo kann ich ihn anmelden?", {"formality": 0.4})
    ]
    
    print("\n🔍 Test verschiedener Query-Typen:\n")
    
    for query, context in test_queries:
        result = demo.process_query(query, context)
        
        print(f"❓ {query}")
        print(f"🎯 Query Type: {result.query_type.value.upper()}")
        print(f"🔧 Komponenten: {', '.join(result.components_used)}")
        
        if result.municipal_pattern:
            print(f"🏛️  Municipal: Pattern='{result.municipal_pattern}' Entity='{result.municipal_entity}'")
        
        if result.liquid_response:
            print(f"🌊 Liquid: {result.liquid_response}")
        
        if result.lfm_response:
            print(f"🤖 LFM: {result.lfm_response}")
        
        print(f"💬 Final: {result.final_response}")
        print(f"⏱️  {result.processing_time_ms:.1f}ms | Confidence: {result.confidence:.2f}")
        print("-" * 60)
    
    print("\n🎯 HYBRID-VORTEIL:")
    print("✅ Municipal-Fragen: Schnelle, präzise Liquid-Antworten")
    print("✅ Allgemeine Fragen: Umfassendes LFM_3B Wissen")  
    print("✅ Hybrid-Fragen: Intelligente Kombination beider")
    print("✅ Automatisches Routing basierend auf Keywords")
    print("✅ Nutzt dein trainiertes Model + Liquid-Optimierung")
    
    print("\n📊 Architektur-Zusammenfassung:")
    print("┌─ Query Router: Entscheidet Municipal/General/Hybrid")
    print("├─ Liquid Layer: Schnelle Municipal-Facts + Kontext")
    print("├─ LFM_3B Base: Allgemeines Wissen aus deinem Training")
    print("└─ Smart Fusion: Kombiniert beide für beste Ergebnisse")

if __name__ == "__main__":
    demonstrate_hybrid_architecture()