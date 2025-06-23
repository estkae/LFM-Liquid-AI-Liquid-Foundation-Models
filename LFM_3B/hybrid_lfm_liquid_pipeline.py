#!/usr/bin/env python3
"""
Hybrid LFM + Liquid Foundation Model Pipeline
Nutzt LFM_3B als Basis für allgemeines Wissen + Liquid-Schicht für Municipal-Spezifika
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import sqlite3
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

# Import unsere bestehenden Komponenten
from liquid_pipeline import PatternType, PatternMatcher, EntityExtractor, LiquidAdapter
from municipal_moe_model import MunicipalMoEModel, MunicipalMoEConfig

logger = logging.getLogger(__name__)

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
    timestamp: str
    components_used: List[str]
    success: bool
    error_message: Optional[str] = None

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
            'wissenschaft': ['physik', 'chemie', 'biologie', 'mathematik'],
            'geschichte': ['weltkrieg', 'mittelalter', 'antike', 'revolution'],
            'geographie': ['hauptstadt', 'kontinent', 'ozean', 'berg'],
            'kultur': ['literatur', 'musik', 'kunst', 'film'],
            'technologie': ['computer', 'internet', 'ki', 'roboter'],
            'gesellschaft': ['politik', 'wirtschaft', 'philosophie', 'religion']
        }
    
    def route_query(self, query: str) -> Tuple[QueryType, float]:
        """Entscheidet welcher Ansatz verwendet werden soll"""
        query_lower = query.lower()
        
        # Score für Municipal-Relevanz
        municipal_score = 0.0
        for category, keywords in self.municipal_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    municipal_score += 1.0
        
        # Score für allgemeines Wissen
        general_score = 0.0
        for category, keywords in self.general_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    general_score += 1.0
        
        # Entscheidungslogik
        if municipal_score > general_score and municipal_score >= 1.0:
            return QueryType.MUNICIPAL_SPECIFIC, municipal_score / (municipal_score + general_score + 1)
        elif general_score > municipal_score and general_score >= 1.0:
            return QueryType.GENERAL_KNOWLEDGE, general_score / (municipal_score + general_score + 1)
        elif municipal_score > 0 and general_score > 0:
            return QueryType.HYBRID, 0.5
        else:
            # Default: Verwende LFM für unklare Fragen
            return QueryType.GENERAL_KNOWLEDGE, 0.3

class LFMWrapper:
    """Wrapper für LFM_3B Model"""
    
    def __init__(self, model_path: str = "./municipal_moe_balanced/best_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Lädt das LFM_3B Model"""
        try:
            # Versuche zuerst Municipal MoE Model zu laden
            self.model = MunicipalMoEModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded Municipal MoE model from {self.model_path}")
            
        except Exception as e:
            logger.warning(f"Could not load Municipal model: {e}")
            
            try:
                # Fallback: Standard GPT-2 für allgemeines Wissen
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("Loaded fallback GPT-2 model")
                
            except Exception as e2:
                logger.error(f"Could not load any model: {e2}")
                self.model = None
                self.tokenizer = None
    
    def generate_response(self, 
                         query: str, 
                         max_length: int = 150,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> Optional[str]:
        """Generiert Antwort mit LFM_3B"""
        
        if self.model is None or self.tokenizer is None:
            return None
        
        try:
            # Prompt formatieren
            prompt = f"Frage: {query}\nAntwort:"
            
            # Tokenisieren
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            
            # Generieren
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=prompt_length + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Dekodieren (nur den generierten Teil)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extrahiere nur die Antwort
            if "Antwort:" in full_response:
                answer = full_response.split("Antwort:")[-1].strip()
                
                # Bereinige die Antwort
                answer = self._clean_response(answer)
                
                return answer if answer else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating LFM response: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """Bereinigt die LFM-Antwort"""
        
        # Entferne unvollständige Sätze am Ende
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Entferne repetitive Teile
        words = response.split()
        if len(words) > 10:
            # Prüfe auf Wiederholungen
            for i in range(3, min(len(words)//2, 10)):
                if words[-i:] == words[-2*i:-i]:
                    response = ' '.join(words[:-i])
                    break
        
        # Kürze sehr lange Antworten
        if len(response) > 300:
            response = response[:300].rsplit(' ', 1)[0] + '...'
        
        return response.strip()

class HybridLiquidPipeline:
    """Hybrid Pipeline: LFM_3B + Liquid Municipal Layer"""
    
    def __init__(self, 
                 lfm_model_path: str = "./municipal_moe_balanced/best_model",
                 knowledge_db_path: str = "municipal_knowledge.db"):
        
        # Komponenten initialisieren
        self.query_router = QueryRouter()
        self.lfm_wrapper = LFMWrapper(lfm_model_path)
        
        # Liquid-Komponenten (nur für Municipal)
        self.pattern_matcher = PatternMatcher()
        self.entity_extractor = EntityExtractor()
        self.liquid_adapter = LiquidAdapter()
        
        # Municipal Knowledge Base
        self.municipal_knowledge = self._load_municipal_knowledge(knowledge_db_path)
        
        logger.info("Hybrid LFM + Liquid Pipeline initialized")
    
    def _load_municipal_knowledge(self, db_path: str) -> Dict[Tuple[str, str], str]:
        """Lädt Municipal-spezifisches Wissen"""
        knowledge = {}
        
        # Basis Municipal-Wissen
        municipal_facts = [
            (("personalausweis", "cost"), "37,00 Euro (22,80 Euro für Personen unter 24 Jahren)"),
            (("reisepass", "cost"), "60,00 Euro (92,00 Euro im Expressverfahren)"),
            (("geburtsurkunde", "cost"), "12,00 Euro"),
            (("ummeldung", "deadline"), "Innerhalb von 14 Tagen nach Umzug"),
            (("personalausweis", "duration"), "4-6 Wochen (Express: 3 Werktage)"),
            (("baugenehmigung", "duration"), "2-6 Monate je nach Komplexität"),
            (("bürgerbüro", "hours"), "Mo-Fr: 8:00-16:00 Uhr, Do: 8:00-18:00 Uhr"),
            (("personalausweis", "documents"), "Altes Ausweisdokument, biometrisches Passfoto, Meldebescheinigung"),
            (("ummeldung", "location"), "Bürgerbüro oder Bürgeramt Ihrer Stadt"),
        ]
        
        for key, value in municipal_facts:
            knowledge[key] = value
        
        return knowledge
    
    def process_query(self, 
                     query: str, 
                     context: Optional[Dict[str, float]] = None) -> HybridQueryResult:
        """Hauptverarbeitung mit Hybrid-Ansatz"""
        
        start_time = datetime.now()
        query_id = self._generate_query_id(query)
        
        if context is None:
            context = {"formality": 0.5, "urgency": 0.3, "language_level": 1.0}
        
        # 1. Query Routing - Entscheide welcher Ansatz
        query_type, routing_confidence = self.query_router.route_query(query)
        
        components_used = []
        liquid_response = None
        lfm_response = None
        municipal_pattern = None
        municipal_entity = None
        
        try:
            if query_type == QueryType.MUNICIPAL_SPECIFIC:
                # Verwende Liquid für Municipal-spezifische Fragen
                components_used.append("Liquid Municipal")
                
                # Pattern & Entity Recognition
                pattern_type, pattern_confidence = self.pattern_matcher.match(query)
                entity, entity_confidence = self.entity_extractor.extract(query)
                
                municipal_pattern = pattern_type.value if pattern_type else None
                municipal_entity = entity
                
                # Municipal Knowledge Lookup
                if pattern_type and entity:
                    key = (entity, pattern_type.value)
                    if key in self.municipal_knowledge:
                        base_response = self.municipal_knowledge[key]
                        liquid_response = self.liquid_adapter.adapt(
                            base_response, entity, pattern_type, context
                        )
                
                # Fallback zu LFM wenn kein Municipal-Wissen
                if not liquid_response:
                    components_used.append("LFM Fallback")
                    lfm_response = self.lfm_wrapper.generate_response(query)
                
                final_response = liquid_response or lfm_response or "Keine spezifischen Informationen verfügbar."
                confidence = routing_confidence
                
            elif query_type == QueryType.GENERAL_KNOWLEDGE:
                # Verwende LFM für allgemeine Fragen
                components_used.append("LFM General")
                
                lfm_response = self.lfm_wrapper.generate_response(query)
                final_response = lfm_response or "Ich kann diese Frage leider nicht beantworten."
                confidence = routing_confidence
                
            else:  # HYBRID
                # Verwende beide Ansätze und kombiniere
                components_used.extend(["Liquid Municipal", "LFM General"])
                
                # Liquid-Versuch
                pattern_type, _ = self.pattern_matcher.match(query)
                entity, _ = self.entity_extractor.extract(query)
                
                if pattern_type and entity:
                    key = (entity, pattern_type.value)
                    if key in self.municipal_knowledge:
                        base_response = self.municipal_knowledge[key]
                        liquid_response = self.liquid_adapter.adapt(
                            base_response, entity, pattern_type, context
                        )
                
                # LFM-Versuch
                lfm_response = self.lfm_wrapper.generate_response(query)
                
                # Kombiniere beide Antworten
                final_response = self._combine_responses(liquid_response, lfm_response, query)
                confidence = 0.7  # Mittlere Confidence für Hybrid
            
            success = bool(final_response and final_response.strip())
            error_message = None
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            final_response = "Es ist ein Fehler bei der Verarbeitung aufgetreten."
            success = False
            error_message = str(e)
            confidence = 0.0
        
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
            timestamp=datetime.now().isoformat(),
            components_used=components_used,
            success=success,
            error_message=error_message
        )
    
    def _combine_responses(self, 
                          liquid_response: Optional[str], 
                          lfm_response: Optional[str],
                          original_query: str) -> str:
        """Kombiniert Liquid- und LFM-Antworten intelligente"""
        
        if liquid_response and lfm_response:
            # Beide Antworten verfügbar - kombiniere sie
            if len(liquid_response) > len(lfm_response):
                return f"{liquid_response} {lfm_response}"
            else:
                return f"{lfm_response} Zusätzlich: {liquid_response}"
        
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
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über geladene Modelle zurück"""
        return {
            "lfm_model_loaded": self.lfm_wrapper.model is not None,
            "lfm_model_path": self.lfm_wrapper.model_path,
            "municipal_knowledge_entries": len(self.municipal_knowledge),
            "device": str(self.lfm_wrapper.device),
            "available_patterns": [p.value for p in PatternType],
            "available_entities": list(self.entity_extractor.entities.keys())
        }

def demo_hybrid_pipeline():
    """Demo der Hybrid Pipeline"""
    
    print("🌊🤖 Hybrid LFM + Liquid Pipeline Demo")
    print("=" * 60)
    
    # Initialisiere Pipeline
    pipeline = HybridLiquidPipeline()
    
    # Zeige Model-Info
    model_info = pipeline.get_model_info()
    print("\n📊 Model Information:")
    print(f"LFM Model loaded: {model_info['lfm_model_loaded']}")
    print(f"Device: {model_info['device']}")
    print(f"Municipal knowledge entries: {model_info['municipal_knowledge_entries']}")
    
    # Test-Anfragen verschiedener Typen
    test_queries = [
        # Municipal-spezifisch (sollte Liquid verwenden)
        ("Was kostet ein Personalausweis?", "Municipal-spezifisch"),
        ("Wo kann ich mich ummelden?", "Municipal-spezifisch"),
        ("Wie lange dauert eine Baugenehmigung?", "Municipal-spezifisch"),
        
        # Allgemeines Wissen (sollte LFM verwenden)
        ("Wer war Albert Einstein?", "Allgemeines Wissen"),
        ("Was ist die Hauptstadt von Frankreich?", "Allgemeines Wissen"),
        ("Wie funktioniert Photosynthese?", "Allgemeines Wissen"),
        
        # Hybrid (könnte beide verwenden)
        ("Brauche ich einen Personalausweis für Reisen nach Frankreich?", "Hybrid"),
        ("Was sind die Öffnungszeiten und wer war der Bürgermeister?", "Hybrid")
    ]
    
    print("\n🔍 Test verschiedener Query-Typen:\n")
    
    for query, expected_type in test_queries:
        result = pipeline.process_query(query)
        
        print(f"❓ {query}")
        print(f"📋 Erwartet: {expected_type} | Erkannt: {result.query_type.value}")
        print(f"🔧 Komponenten: {', '.join(result.components_used)}")
        
        if result.municipal_pattern:
            print(f"🏛️  Municipal: {result.municipal_pattern} + {result.municipal_entity}")
        
        print(f"💬 Antwort: {result.final_response}")
        print(f"⏱️  {result.processing_time_ms:.2f}ms | Confidence: {result.confidence:.2f}")
        print("-" * 60)
    
    print("\n✨ DAS IST DER HYBRID-VORTEIL:")
    print("- Municipal-Fragen: Schnelle Liquid-Antworten (Fakten + Kontext)")
    print("- Allgemeine Fragen: LFM_3B Wissensbasis")
    print("- Hybrid-Fragen: Beste aus beiden Welten")
    print("- Automatisches Routing basierend auf Anfrage-Typ")

if __name__ == "__main__":
    demo_hybrid_pipeline()