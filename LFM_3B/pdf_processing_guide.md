# 📄 PDF-Aufbereitung für Municipal MoE Training

## 🎯 Ziel
PDFs mit deutschen Verwaltungsinformationen in strukturierte Training-Daten für dein Municipal MoE Model umwandeln.

## 📋 Schritt-für-Schritt Anleitung

### 1. **Vorbereitung**
```bash
# Installiere erforderliche Pakete
pip install pymupdf4llm

# Erstelle Verzeichnis für PDFs
mkdir municipal_pdfs
```

### 2. **PDF-Sammlung**
Sammle relevante PDFs wie:
- **Bürgerservice-Broschüren** (Personalausweis, Reisepass, etc.)
- **Antragsformulare** mit Erklärungen
- **FAQ-Dokumente** von Städten/Gemeinden
- **Gebührenordnungen**
- **Verwaltungshandbücher**

### 3. **PDF-Verarbeitung**

#### Einzelne PDF verarbeiten:
```bash
python3 pdf_to_training_data.py --pdf-file "./municipal_pdfs/buergerservice.pdf"
```

#### Ganzes Verzeichnis verarbeiten:
```bash
python3 pdf_to_training_data.py --pdf-dir "./municipal_pdfs" --output "pdf_training_data.jsonl"
```

### 4. **Qualitätsprüfung**
```bash
# Prüfe die extrahierten Daten
head -5 pdf_training_data.jsonl

# Zähle Beispiele
wc -l pdf_training_data.jsonl
```

### 5. **Daten kombinieren**
```bash
# Kombiniere PDF-Daten mit bestehenden generierten Daten
python3 combine_training_data.py \
    --pdf-data "pdf_training_data.jsonl" \
    --existing-data "massive_municipal_training_data.jsonl" \
    --output "combined_municipal_training_data.jsonl" \
    --balance-ratio 0.4
```

### 6. **Training mit kombinierten Daten**
```bash
# Trainiere Model mit kombinierten Daten
python3 train_municipal_moe_improved.py \
    --data-file "combined_municipal_training_data.jsonl" \
    --output-dir "./municipal_moe_pdf_trained" \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 5e-5
```

## 🔧 Was das Script macht

### **Text-Extraktion**
- Verwendet `pymupdf4llm` für saubere Markdown-Extraktion
- Behält Struktur und Formatierung bei
- Entfernt Seitenzahlen und Headers/Footers

### **Inhalts-Filterung** 
- Prüft auf Municipal-Keywords (personalausweis, baugenehmigung, etc.)
- Filtert irrelevante Inhalte heraus
- Behält nur verwaltungsrelevante Texte

### **QA-Generierung**
```python
# Automatische Fragenerstellung:
"Was kostet eine Geburtsurkunde?" → "Geburtsurkunde: 12 Euro"
"Wie beantrage ich einen Personalausweis?" → "Personalausweis: Antrag im Bürgerbüro..."
```

### **Datenstruktur**
```json
{
  "text": "Frage: Was kostet eine Geburtsurkunde?\nAntwort: Eine Geburtsurkunde kostet 12 Euro.",
  "source": "buergerservice.pdf",
  "type": "qa_pair"
}
```

## 📊 Beispiel-Workflow

```bash
# 1. PDFs sammeln
mkdir municipal_pdfs
# Kopiere PDFs in dieses Verzeichnis

# 2. PDFs verarbeiten
python3 pdf_to_training_data.py --pdf-dir "./municipal_pdfs"

# 3. Daten kombinieren (30% PDF, 70% generiert)
python3 combine_training_data.py --balance-ratio 0.3

# 4. Training starten
python3 train_municipal_moe_improved.py \
    --data-file "combined_municipal_training_data.jsonl" \
    --output-dir "./municipal_moe_pdf_enhanced"
```

## 💡 Tipps für bessere Ergebnisse

### **PDF-Qualität**
- ✅ **Gute PDFs**: Direkte Texte, strukturierte FAQs
- ❌ **Schlechte PDFs**: Gescannte Bilder, komplexe Layouts

### **Content-Typen**
- 🏆 **Beste**: FAQ-Dokumente, Antragsanleitungen
- 👍 **Gut**: Informationsbroschüren, Gebührenordnungen  
- 🤔 **OK**: Formulare mit Erklärungen

### **Balance-Ratio**
```python
# Empfohlene Ratios:
--balance-ratio 0.2  # 20% PDF, 80% generiert (sicher)
--balance-ratio 0.4  # 40% PDF, 60% generiert (ausgewogen)
--balance-ratio 0.6  # 60% PDF, 40% generiert (PDF-fokussiert)
```

## 🚨 Häufige Probleme

### **Keine Municipal-Inhalte erkannt**
```bash
# Prüfe Keywords im PDF
python3 -c "
import pymupdf4llm
text = pymupdf4llm.to_markdown('deine_datei.pdf')
print('personalausweis' in text.lower())
print('baugenehmigung' in text.lower())
"
```

### **Schlechte Text-Extraktion**
- PDF könnte gescannt sein → OCR verwenden
- Komplexes Layout → Manuell bereinigen

### **Zu wenige QA-Paare**
- Mehr FAQ-strukturierte PDFs verwenden
- Manual nachbearbeiten mit Texteditor

## 📈 Monitoring

```bash
# Prüfe Datenqualität
python3 -c "
import json
with open('pdf_training_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    
print(f'Gesamt: {len(data)} Beispiele')
qa_pairs = sum(1 for d in data if d['type'] == 'qa_pair')
print(f'Direkte QA: {qa_pairs}')
print(f'Generierte QA: {len(data) - qa_pairs}')
"
```

Die kombinierten Daten werden dein Model mit echten Verwaltungsinformationen trainieren und die Antwortqualität deutlich verbessern!