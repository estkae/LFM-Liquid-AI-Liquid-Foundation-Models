#!/usr/bin/env python3
"""
PDF-Aufbereitung für Municipal MoE Training
Extrahiert und strukturiert PDF-Inhalte für das Training
"""

import pymupdf4llm
import json
import re
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import os

class PDFMunicipalProcessor:
    """Verarbeitet PDF-Dokumente für Municipal Training"""
    
    def __init__(self):
        self.municipal_keywords = [
            'personalausweis', 'reisepass', 'führerschein', 'anmeldung', 'ummeldung',
            'abmeldung', 'geburtsurkunde', 'heiratsurkunde', 'sterbeurkunde',
            'baugenehmigung', 'gewerbeanmeldung', 'gewerbeabmeldung', 'wohngeld',
            'kindergeld', 'elterngeld', 'bürgerbüro', 'standesamt', 'ordnungsamt',
            'sozialamt', 'jugendamt', 'finanzamt', 'meldeamt', 'bauamt'
        ]
        
        self.question_patterns = [
            r'was kostet',
            r'wie beantrage ich',
            r'wo kann ich',
            r'welche unterlagen',
            r'wie lange dauert',
            r'wann muss ich',
            r'was brauche ich',
            r'wie funktioniert'
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extrahiert Text aus PDF mit pymupdf4llm"""
        print(f"📄 Extrahiere Text aus: {pdf_path}")
        
        try:
            # Extrahiere Markdown-formatierter Text
            md_text = pymupdf4llm.to_markdown(pdf_path)
            return md_text
        except Exception as e:
            print(f"❌ Fehler beim Extrahieren von {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Bereinigt extrahierten Text"""
        
        # Entferne übermäßige Leerzeichen und Zeilenumbrüche
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Entferne Seitenzahlen
        text = re.sub(r'Seite \d+', '', text)
        text = re.sub(r'\d+\s*/\s*\d+', '', text)
        
        # Entferne Header/Footer patterns
        text = re.sub(r'^[-=]{3,}.*$', '', text, flags=re.MULTILINE)
        
        # Entferne URLs und E-Mail-Adressen (außer sie sind relevant)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def is_municipal_content(self, text: str) -> bool:
        """Prüft ob Text Municipal-relevante Inhalte hat"""
        text_lower = text.lower()
        
        # Mindestens 2 Municipal-Keywords
        keyword_count = sum(1 for keyword in self.municipal_keywords if keyword in text_lower)
        return keyword_count >= 2
    
    def extract_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Extrahiert Frage-Antwort-Paare aus Text"""
        qa_pairs = []
        
        # Suche nach FAQ-Strukturen
        faq_patterns = [
            r'(Frage:?\s*)(.*?)\n(Antwort:?\s*)(.*?)(?=\nFrage:|\nQ:|\n\n|$)',
            r'(Q:?\s*)(.*?)\n(A:?\s*)(.*?)(?=\nQ:|\nFrage:|\n\n|$)',
            r'(\d+\.\s*)(.*?\?)\s*(.*?)(?=\n\d+\.|\n\n|$)'
        ]
        
        for pattern in faq_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 4:
                    question = match.group(2).strip()
                    answer = match.group(4).strip()
                    
                    if len(question) > 10 and len(answer) > 20:
                        qa_pairs.append({
                            "question": question,
                            "answer": answer
                        })
        
        return qa_pairs
    
    def create_training_examples(self, text: str, pdf_name: str) -> List[Dict[str, str]]:
        """Erstellt Training-Beispiele aus Text"""
        examples = []
        
        # 1. Direkte QA-Paare extrahieren
        qa_pairs = self.extract_qa_pairs(text)
        for qa in qa_pairs:
            examples.append({
                "text": f"Frage: {qa['question']}\nAntwort: {qa['answer']}",
                "source": pdf_name,
                "type": "qa_pair"
            })
        
        # 2. Informative Abschnitte in QA umwandeln
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            
            if len(para) < 50 or not self.is_municipal_content(para):
                continue
            
            # Versuche automatisch Fragen zu generieren
            generated_questions = self.generate_questions_from_text(para)
            for question in generated_questions:
                examples.append({
                    "text": f"Frage: {question}\nAntwort: {para}",
                    "source": pdf_name,
                    "type": "generated_qa"
                })
        
        return examples
    
    def generate_questions_from_text(self, text: str) -> List[str]:
        """Generiert Fragen basierend auf Textinhalt"""
        questions = []
        text_lower = text.lower()
        
        # Regelbasierte Fragengenerierung
        if 'kosten' in text_lower or 'gebühr' in text_lower or 'preis' in text_lower:
            for keyword in self.municipal_keywords:
                if keyword in text_lower:
                    questions.append(f"Was kostet {keyword.replace('_', ' ')}?")
                    break
        
        if 'beantragen' in text_lower or 'antrag' in text_lower:
            for keyword in self.municipal_keywords:
                if keyword in text_lower:
                    questions.append(f"Wie beantrage ich {keyword.replace('_', ' ')}?")
                    break
        
        if 'unterlagen' in text_lower or 'dokumente' in text_lower:
            questions.append("Welche Unterlagen brauche ich?")
        
        if 'öffnungszeiten' in text_lower or 'sprechzeiten' in text_lower:
            questions.append("Wie sind die Öffnungszeiten?")
        
        if 'adresse' in text_lower or 'wo' in text_lower:
            questions.append("Wo finde ich das Amt?")
        
        return questions[:2]  # Max 2 Fragen pro Absatz
    
    def process_pdf_directory(self, pdf_dir: str, output_file: str = "pdf_training_data.jsonl"):
        """Verarbeitet alle PDFs in einem Verzeichnis"""
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            print(f"❌ Verzeichnis nicht gefunden: {pdf_dir}")
            return
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ Keine PDF-Dateien in {pdf_dir} gefunden")
            return
        
        print(f"📚 Verarbeite {len(pdf_files)} PDF-Dateien...")
        
        all_examples = []
        
        for pdf_file in pdf_files:
            print(f"\n📄 Verarbeite: {pdf_file.name}")
            
            # Text extrahieren
            raw_text = self.extract_from_pdf(str(pdf_file))
            if not raw_text:
                continue
            
            # Text bereinigen
            clean_text = self.clean_text(raw_text)
            
            # Prüfe Municipal-Relevanz
            if not self.is_municipal_content(clean_text):
                print(f"⚠️  Keine Municipal-Inhalte gefunden in {pdf_file.name}")
                continue
            
            # Training-Beispiele erstellen
            examples = self.create_training_examples(clean_text, pdf_file.name)
            all_examples.extend(examples)
            
            print(f"✅ {len(examples)} Beispiele aus {pdf_file.name} extrahiert")
        
        # Speichere alle Beispiele
        if all_examples:
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in all_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            print(f"\n🎉 {len(all_examples)} Training-Beispiele gespeichert in {output_file}")
            
            # Statistiken
            qa_pairs = sum(1 for ex in all_examples if ex['type'] == 'qa_pair')
            generated = sum(1 for ex in all_examples if ex['type'] == 'generated_qa')
            
            print(f"📊 Statistiken:")
            print(f"   - Direkte QA-Paare: {qa_pairs}")
            print(f"   - Generierte QAs: {generated}")
            print(f"   - Gesamt: {len(all_examples)}")
        else:
            print("❌ Keine verwertbaren Beispiele gefunden")
    
    def process_single_pdf(self, pdf_path: str, output_file: str = None):
        """Verarbeitet eine einzelne PDF-Datei"""
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"❌ PDF nicht gefunden: {pdf_path}")
            return
        
        if not output_file:
            output_file = f"{pdf_file.stem}_training_data.jsonl"
        
        print(f"📄 Verarbeite einzelne PDF: {pdf_file.name}")
        
        # Text extrahieren und verarbeiten
        raw_text = self.extract_from_pdf(pdf_path)
        if not raw_text:
            return
        
        clean_text = self.clean_text(raw_text)
        
        if not self.is_municipal_content(clean_text):
            print(f"⚠️  Keine Municipal-Inhalte gefunden")
            return
        
        examples = self.create_training_examples(clean_text, pdf_file.name)
        
        if examples:
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            print(f"✅ {len(examples)} Beispiele gespeichert in {output_file}")
        else:
            print("❌ Keine verwertbaren Beispiele gefunden")


def main():
    parser = argparse.ArgumentParser(description="PDF zu Municipal Training Data Konverter")
    parser.add_argument("--pdf-dir", type=str, help="Verzeichnis mit PDF-Dateien")
    parser.add_argument("--pdf-file", type=str, help="Einzelne PDF-Datei")
    parser.add_argument("--output", type=str, default="pdf_training_data.jsonl", 
                       help="Output-Datei für Training-Daten")
    
    args = parser.parse_args()
    
    if not args.pdf_dir and not args.pdf_file:
        print("❌ Bitte --pdf-dir oder --pdf-file angeben")
        return
    
    processor = PDFMunicipalProcessor()
    
    if args.pdf_dir:
        processor.process_pdf_directory(args.pdf_dir, args.output)
    elif args.pdf_file:
        processor.process_single_pdf(args.pdf_file, args.output)


if __name__ == "__main__":
    main()