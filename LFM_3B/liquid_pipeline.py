#!/usr/bin/env python3
"""
Liquid Foundation Model Pipeline - Vollständige Produktions-Pipeline
"""

import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import hashlib

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Frage-Pattern Typen"""
    COST = "cost"
    PROCESS = "process"
    LOCATION = "location"
    DOCUMENTS = "documents"
    DURATION = "duration"
    DEADLINE = "deadline"
    REQUIREMENTS = "requirements"
    FUNCTION = "function"
    HOURS = "hours"
    CONTACT = "contact"

@dataclass
class QueryResult:
    """Struktur für Pipeline-Ergebnisse"""
    query_id: str
    original_query: str
    pattern_type: Optional[str]
    entity: Optional[str]
    confidence: float
    base_response: Optional[str]
    adapted_response: str
    context: Dict[str, float]
    processing_time_ms: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class KnowledgeBase:
    """Verwaltung der statischen Wissensbasis"""
    
    def __init__(self, db_path: str = "municipal_knowledge.db"):
        self.db_path = db_path
        self._init_database()
        self._load_default_knowledge()
    
    def _init_database(self):
        """Initialisiert SQLite Datenbank"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Knowledge Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                entity TEXT,
                pattern_type TEXT,
                value TEXT,
                metadata TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (entity, pattern_type)
            )
        """)
        
        # Query Log Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                query_id TEXT PRIMARY KEY,
                original_query TEXT,
                pattern_type TEXT,
                entity TEXT,
                confidence REAL,
                response TEXT,
                context TEXT,
                processing_time_ms REAL,
                timestamp TIMESTAMP,
                success BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_knowledge(self):
        """Lädt Standard-Wissensbasis"""
        default_knowledge = [
            # Kosten
            ("personalausweis", "cost", "37,00 Euro (22,80 Euro für Personen unter 24 Jahren)"),
            ("reisepass", "cost", "60,00 Euro (92,00 Euro im Expressverfahren)"),
            ("führerschein", "cost", "43,40 Euro"),
            ("geburtsurkunde", "cost", "12,00 Euro"),
            ("heiratsurkunde", "cost", "12,00 Euro"),
            ("meldebescheinigung", "cost", "5,00 Euro"),
            
            # Prozesse
            ("personalausweis", "process", "1. Termin online buchen 2. Biometrisches Foto mitbringen 3. Altes Dokument vorlegen 4. Antrag unterschreiben"),
            ("ummeldung", "process", "1. Meldeformular ausfüllen 2. Personalausweis mitbringen 3. Wohnungsgeberbestätigung vorlegen"),
            
            # Orte
            ("personalausweis", "location", "Bürgerbüro, Bürgeramt oder Rathaus Ihrer Stadt"),
            ("ummeldung", "location", "Einwohnermeldeamt oder Bürgerbüro"),
            ("führerschein", "location", "Führerscheinstelle des Straßenverkehrsamts"),
            
            # Dokumente
            ("personalausweis", "documents", "Altes Ausweisdokument, biometrisches Passfoto, Meldebescheinigung"),
            ("ummeldung", "documents", "Personalausweis/Reisepass, Wohnungsgeberbestätigung"),
            ("geburtsurkunde", "documents", "Personalausweis, ggf. Abstammungsurkunde"),
            
            # Dauer
            ("personalausweis", "duration", "4-6 Wochen (Express: 3 Werktage)"),
            ("reisepass", "duration", "3-4 Wochen (Express: 3 Werktage)"),
            ("baugenehmigung", "duration", "2-6 Monate je nach Komplexität"),
            
            # Fristen
            ("ummeldung", "deadline", "Innerhalb von 14 Tagen nach Umzug"),
            ("personalausweis", "deadline", "Rechtzeitig vor Ablauf beantragen"),
            
            # Öffnungszeiten
            ("bürgerbüro", "hours", "Mo-Fr: 8:00-16:00 Uhr, Do: 8:00-18:00 Uhr, Sa: 9:00-12:00 Uhr"),
            
            # Kontakt
            ("bürgerbüro", "contact", "Tel: 0221/221-0, E-Mail: buergerbuero@stadt.de"),
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for entity, pattern, value in default_knowledge:
            cursor.execute("""
                INSERT OR IGNORE INTO knowledge 
                (entity, pattern_type, value, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (entity, pattern, value, "{}"))
        
        conn.commit()
        conn.close()
    
    def lookup(self, entity: str, pattern_type: str) -> Optional[str]:
        """Sucht Wissen in der Datenbank"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT value FROM knowledge 
            WHERE entity = ? AND pattern_type = ?
        """, (entity, pattern_type))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def add_knowledge(self, entity: str, pattern_type: str, value: str, metadata: Dict = None):
        """Fügt neues Wissen hinzu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge 
            (entity, pattern_type, value, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        """, (entity, pattern_type, value, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added knowledge: {entity} - {pattern_type} = {value}")

class PatternMatcher:
    """Pattern Recognition Engine"""
    
    def __init__(self):
        self.patterns = {
            PatternType.COST: [
                (r"was kostet", 1.0),
                (r"wie teuer", 0.9),
                (r"gebühr", 0.8),
                (r"preis", 0.7),
                (r"wie viel.*bezahl", 0.9),
                (r"kosten für", 0.95)
            ],
            PatternType.PROCESS: [
                (r"wie beantrage ich", 1.0),
                (r"wo beantrage", 0.9),
                (r"antrag stellen", 0.9),
                (r"wie kann ich.*beantragen", 0.95),
                (r"beantragung von", 0.85)
            ],
            PatternType.LOCATION: [
                (r"wo kann ich", 1.0),
                (r"wo finde ich", 0.9),
                (r"wo ist", 0.8),
                (r"wohin muss ich", 0.95),
                (r"an welcher stelle", 0.85)
            ],
            PatternType.DOCUMENTS: [
                (r"welche unterlagen", 1.0),
                (r"was brauche ich für", 0.9),
                (r"welche dokumente", 0.95),
                (r"was muss ich mitbringen", 0.95),
                (r"erforderliche unterlagen", 0.9)
            ],
            PatternType.DURATION: [
                (r"wie lange dauert", 1.0),
                (r"wann.*fertig", 0.9),
                (r"bearbeitungszeit", 0.95),
                (r"wie schnell", 0.85),
                (r"dauer der", 0.9)
            ],
            PatternType.DEADLINE: [
                (r"wann muss ich", 1.0),
                (r"bis wann", 0.95),
                (r"frist für", 0.9),
                (r"innerhalb welcher zeit", 0.9),
                (r"deadline", 0.8)
            ],
            PatternType.HOURS: [
                (r"öffnungszeit", 1.0),
                (r"wann.*geöffnet", 0.95),
                (r"wann.*offen", 0.9),
                (r"sprechzeit", 0.9),
                (r"servicezeit", 0.85)
            ],
            PatternType.CONTACT: [
                (r"telefonnummer", 1.0),
                (r"e-?mail", 0.9),
                (r"kontakt", 0.85),
                (r"wie erreiche ich", 0.95),
                (r"ansprechpartner", 0.9)
            ]
        }
    
    def match(self, text: str) -> Tuple[Optional[PatternType], float]:
        """Findet das beste Pattern mit Confidence Score"""
        text_lower = text.lower()
        best_match = None
        best_score = 0.0
        
        for pattern_type, pattern_list in self.patterns.items():
            for pattern_regex, weight in pattern_list:
                if re.search(pattern_regex, text_lower):
                    # Score basiert auf Pattern-Länge und Gewicht
                    match = re.search(pattern_regex, text_lower)
                    coverage = len(match.group()) / len(text_lower)
                    score = weight * (0.7 + 0.3 * coverage)
                    
                    if score > best_score:
                        best_match = pattern_type
                        best_score = score
        
        return best_match, best_score

class EntityExtractor:
    """Entity Extraction Engine"""
    
    def __init__(self):
        self.entities = {
            "personalausweis": ["personalausweis", "perso", "ausweis", "personal-ausweis"],
            "reisepass": ["reisepass", "pass", "reise-pass"],
            "führerschein": ["führerschein", "fahrerlaubnis", "führerscheins"],
            "geburtsurkunde": ["geburtsurkunde", "geburts-urkunde", "geburt"],
            "heiratsurkunde": ["heiratsurkunde", "heirats-urkunde", "eheurkunde"],
            "ummeldung": ["ummeldung", "umzug", "wohnsitz", "ummelden", "umgemeldet"],
            "anmeldung": ["anmeldung", "erstanmeldung", "anmelden"],
            "baugenehmigung": ["baugenehmigung", "bauantrag", "bau-genehmigung"],
            "gewerbeanmeldung": ["gewerbeanmeldung", "gewerbe", "gewerbe-anmeldung"],
            "kindergeld": ["kindergeld", "kinder-geld"],
            "elterngeld": ["elterngeld", "eltern-geld"],
            "wohngeld": ["wohngeld", "wohn-geld"],
            "bürgerbüro": ["bürgerbüro", "bürgeramt", "rathaus", "bürger-büro"],
            "meldebescheinigung": ["meldebescheinigung", "melde-bescheinigung"],
            "hundesteuer": ["hundesteuer", "hunde-steuer", "hund"]
        }
        
        # Compile regex patterns für bessere Performance
        self.entity_patterns = {}
        for entity, keywords in self.entities.items():
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.entity_patterns[entity] = re.compile(pattern, re.IGNORECASE)
    
    def extract(self, text: str) -> Tuple[Optional[str], float]:
        """Extrahiert Entity mit Confidence Score"""
        best_entity = None
        best_score = 0.0
        
        for entity, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Score basiert auf Anzahl und Position der Matches
                match_score = len(matches) * 0.3
                
                # Bonus für frühe Position
                first_match = pattern.search(text)
                if first_match:
                    position_score = 1.0 - (first_match.start() / len(text))
                    match_score += position_score * 0.7
                
                if match_score > best_score:
                    best_entity = entity
                    best_score = min(match_score, 1.0)
        
        return best_entity, best_score

class LiquidAdapter:
    """Liquid Neural Network Simulator für Kontext-Anpassung"""
    
    def __init__(self):
        self.style_templates = {
            "formal": {
                "prefix": "Sehr geehrte/r Bürger/in, ",
                "cost": "die Gebühr für {entity} beträgt {value}.",
                "process": "für {entity} ist folgender Ablauf vorgesehen: {value}",
                "suffix": " Mit freundlichen Grüßen, Ihr Bürgeramt."
            },
            "informal": {
                "prefix": "",
                "cost": "{entity} kostet {value}.",
                "process": "{entity}: {value}",
                "suffix": ""
            },
            "urgent": {
                "prefix": "WICHTIG: ",
                "cost": "{entity} = {value}! Bitte beachten!",
                "process": "DRINGEND für {entity}: {value}",
                "suffix": " Bitte zeitnah erledigen!"
            },
            "simple": {
                "prefix": "",
                "cost": "{entity}: {value} bezahlen.",
                "process": "{entity} so machen: {value}",
                "suffix": ""
            }
        }
    
    def adapt(self, 
              base_response: str, 
              entity: str,
              pattern_type: PatternType,
              context: Dict[str, float]) -> str:
        """Passt Antwort an Kontext an"""
        
        # Bestimme Stil basierend auf Kontext
        style = self._determine_style(context)
        
        # Hole Template
        templates = self.style_templates[style]
        pattern_key = pattern_type.value if pattern_type else "default"
        
        # Formatiere Entity-Namen
        entity_display = entity.replace("_", " ").title() if entity else "Information"
        
        # Wähle Template
        if pattern_key in templates:
            template = templates[pattern_key]
        else:
            template = "{entity}: {value}"
        
        # Baue Antwort
        response = template.format(entity=entity_display, value=base_response)
        
        # Füge Prefix/Suffix hinzu
        if context.get("include_greeting", False):
            response = templates["prefix"] + response + templates["suffix"]
        
        # Weitere Anpassungen
        if context.get("language_level", 1.0) < 0.5:
            response = self._simplify_language(response)
        
        if context.get("add_help", False):
            response += " Weitere Fragen? Rufen Sie uns an: 0221/221-0"
        
        return response.strip()
    
    def _determine_style(self, context: Dict[str, float]) -> str:
        """Bestimmt Stil basierend auf Kontext"""
        if context.get("urgency", 0.0) > 0.7:
            return "urgent"
        elif context.get("formality", 0.5) > 0.7:
            return "formal"
        elif context.get("language_level", 1.0) < 0.5:
            return "simple"
        else:
            return "informal"
    
    def _simplify_language(self, text: str) -> str:
        """Vereinfacht Sprache"""
        simplifications = {
            "beträgt": "ist",
            "Gebühr": "Preis",
            "erforderlich": "nötig",
            "Dokument": "Papier",
            "vereinbaren": "machen",
            "beantragen": "holen"
        }
        
        for complex_word, simple_word in simplifications.items():
            text = text.replace(complex_word, simple_word)
        
        return text

class LiquidPipeline:
    """Hauptpipeline für Liquid Foundation Model"""
    
    def __init__(self, 
                 knowledge_db_path: str = "municipal_knowledge.db",
                 enable_logging: bool = True):
        
        self.knowledge_base = KnowledgeBase(knowledge_db_path)
        self.pattern_matcher = PatternMatcher()
        self.entity_extractor = EntityExtractor()
        self.liquid_adapter = LiquidAdapter()
        self.enable_logging = enable_logging
        
        logger.info("Liquid Pipeline initialized")
    
    def process(self, 
                query: str, 
                context: Optional[Dict[str, float]] = None) -> QueryResult:
        """Hauptverarbeitungsfunktion"""
        
        start_time = datetime.now()
        query_id = self._generate_query_id(query)
        
        # Default Context
        if context is None:
            context = self._create_default_context()
        
        try:
            # 1. Pattern Matching
            pattern_type, pattern_confidence = self.pattern_matcher.match(query)
            
            # 2. Entity Extraction
            entity, entity_confidence = self.entity_extractor.extract(query)
            
            # Combined Confidence
            confidence = (pattern_confidence + entity_confidence) / 2.0
            
            # 3. Knowledge Lookup
            base_response = None
            if pattern_type and entity:
                base_response = self.knowledge_base.lookup(entity, pattern_type.value)
            
            # 4. Response Generation
            if base_response:
                # Liquid Adaptation
                adapted_response = self.liquid_adapter.adapt(
                    base_response, entity, pattern_type, context
                )
                success = True
                error_message = None
            else:
                # Fallback
                adapted_response = self._generate_fallback(pattern_type, entity, confidence)
                success = False
                error_message = "No knowledge found"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = QueryResult(
                query_id=query_id,
                original_query=query,
                pattern_type=pattern_type.value if pattern_type else None,
                entity=entity,
                confidence=confidence,
                base_response=base_response,
                adapted_response=adapted_response,
                context=context,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                success=success,
                error_message=error_message,
                metadata={
                    "pattern_confidence": pattern_confidence,
                    "entity_confidence": entity_confidence
                }
            )
            
            # Log query
            if self.enable_logging:
                self._log_query(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResult(
                query_id=query_id,
                original_query=query,
                pattern_type=None,
                entity=None,
                confidence=0.0,
                base_response=None,
                adapted_response="Entschuldigung, es ist ein Fehler aufgetreten.",
                context=context,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def batch_process(self, queries: List[str], context: Optional[Dict] = None) -> List[QueryResult]:
        """Verarbeitet mehrere Anfragen"""
        results = []
        for query in queries:
            result = self.process(query, context)
            results.append(result)
        return results
    
    def add_knowledge(self, entity: str, pattern_type: str, value: str, metadata: Dict = None):
        """Fügt neues Wissen zur Pipeline hinzu"""
        self.knowledge_base.add_knowledge(entity, pattern_type, value, metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Holt Statistiken aus der Datenbank"""
        conn = sqlite3.connect(self.knowledge_base.db_path)
        cursor = conn.cursor()
        
        # Erfolgsrate
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(success) as successful,
                AVG(confidence) as avg_confidence,
                AVG(processing_time_ms) as avg_time
            FROM query_log
        """)
        
        stats = cursor.fetchone()
        
        # Top Patterns
        cursor.execute("""
            SELECT pattern_type, COUNT(*) as count
            FROM query_log
            WHERE pattern_type IS NOT NULL
            GROUP BY pattern_type
            ORDER BY count DESC
            LIMIT 5
        """)
        
        top_patterns = cursor.fetchall()
        
        # Top Entities
        cursor.execute("""
            SELECT entity, COUNT(*) as count
            FROM query_log
            WHERE entity IS NOT NULL
            GROUP BY entity
            ORDER BY count DESC
            LIMIT 5
        """)
        
        top_entities = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_queries": stats[0] or 0,
            "successful_queries": stats[1] or 0,
            "success_rate": (stats[1] / stats[0] * 100) if stats[0] else 0,
            "avg_confidence": stats[2] or 0,
            "avg_processing_time_ms": stats[3] or 0,
            "top_patterns": top_patterns,
            "top_entities": top_entities
        }
    
    def _generate_query_id(self, query: str) -> str:
        """Generiert eindeutige Query ID"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{query}{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _create_default_context(self) -> Dict[str, float]:
        """Erstellt Standard-Kontext"""
        import random
        
        # Simuliere Tageszeit-basierten Kontext
        hour = datetime.now().hour
        is_business_hours = 9 <= hour <= 17
        
        return {
            "formality": 0.5,
            "urgency": 0.3,
            "language_level": 1.0,
            "user_age": 0.5,
            "time_of_day": hour / 24.0,
            "is_business_hours": float(is_business_hours),
            "include_greeting": False,
            "add_help": random.random() > 0.8  # 20% Chance für Hilfe-Text
        }
    
    def _generate_fallback(self, 
                          pattern_type: Optional[PatternType], 
                          entity: Optional[str],
                          confidence: float) -> str:
        """Generiert Fallback-Antwort"""
        
        if confidence > 0.5:
            if pattern_type and not entity:
                return f"Ich verstehe, dass Sie nach {pattern_type.value} fragen, aber ich konnte nicht erkennen, worum es genau geht."
            elif entity and not pattern_type:
                return f"Sie fragen nach {entity}, aber ich bin nicht sicher, was genau Sie wissen möchten."
            else:
                return "Leider habe ich dazu keine spezifischen Informationen. Bitte wenden Sie sich an das Bürgerbüro."
        else:
            return "Entschuldigung, ich konnte Ihre Frage nicht verstehen. Können Sie sie anders formulieren?"
    
    def _log_query(self, result: QueryResult):
        """Loggt Query in Datenbank"""
        conn = sqlite3.connect(self.knowledge_base.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO query_log 
            (query_id, original_query, pattern_type, entity, confidence, 
             response, context, processing_time_ms, timestamp, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.query_id,
            result.original_query,
            result.pattern_type,
            result.entity,
            result.confidence,
            result.adapted_response,
            json.dumps(result.context),
            result.processing_time_ms,
            result.timestamp,
            result.success
        ))
        
        conn.commit()
        conn.close()

def demo_pipeline():
    """Demo der Liquid Pipeline"""
    
    print("🌊 Liquid Foundation Model Pipeline Demo")
    print("=" * 60)
    
    # Initialisiere Pipeline
    pipeline = LiquidPipeline()
    
    # Test-Anfragen
    test_queries = [
        ("Was kostet ein Personalausweis?", {"formality": 0.5, "urgency": 0.2}),
        ("Was kostet ein Personalausweis?", {"formality": 0.9, "urgency": 0.2}),
        ("Was kostet ein Personalausweis?", {"formality": 0.3, "urgency": 0.9}),
        ("Wie beantrage ich einen Reisepass?", {"language_level": 0.3}),
        ("Wo kann ich mich ummelden?", {"include_greeting": True}),
        ("Welche Unterlagen brauche ich für eine Geburtsurkunde?", None),
        ("Wie lange dauert eine Baugenehmigung?", {"add_help": True}),
        ("Öffnungszeiten Bürgerbüro", {"formality": 0.1}),
    ]
    
    print("\n📊 Verarbeite Anfragen:\n")
    
    for query, context in test_queries:
        result = pipeline.process(query, context)
        
        print(f"❓ {result.original_query}")
        if result.success:
            print(f"✅ Pattern: {result.pattern_type} | Entity: {result.entity}")
            print(f"   Confidence: {result.confidence:.2f}")
        else:
            print(f"⚠️  Fehler: {result.error_message}")
        print(f"💬 {result.adapted_response}")
        print(f"⏱️  {result.processing_time_ms:.2f}ms")
        print("-" * 60)
    
    # Füge neues Wissen hinzu
    print("\n➕ Füge neues Wissen hinzu:")
    pipeline.add_knowledge(
        entity="parkausweis",
        pattern_type="cost",
        value="30 Euro pro Jahr für Anwohner"
    )
    
    # Teste neues Wissen
    result = pipeline.process("Was kostet ein Parkausweis?")
    print(f"\n❓ {result.original_query}")
    print(f"💬 {result.adapted_response}")
    
    # Zeige Statistiken
    print("\n📈 Pipeline Statistiken:")
    stats = pipeline.get_statistics()
    print(f"Gesamt Anfragen: {stats['total_queries']}")
    print(f"Erfolgsrate: {stats['success_rate']:.1f}%")
    print(f"Durchschnittliche Verarbeitungszeit: {stats['avg_processing_time_ms']:.2f}ms")
    
    # Performance Test
    print("\n⚡ Performance Test (100 Anfragen):")
    import time
    
    start = time.time()
    for _ in range(100):
        pipeline.process("Was kostet ein Personalausweis?", {"formality": 0.5})
    
    elapsed = time.time() - start
    print(f"100 Anfragen in {elapsed:.2f}s = {100/elapsed:.0f} Anfragen/Sekunde")

if __name__ == "__main__":
    demo_pipeline()