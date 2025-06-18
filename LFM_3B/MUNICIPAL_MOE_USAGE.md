# Municipal MoE Model - Nutzungsanleitung

## 🏛️ Übersicht
Das Municipal MoE (Mixture of Experts) Modell ist speziell für deutsche Kommunalverwaltungen entwickelt und verfügt über 8 spezialisierte Experten für verschiedene Ämter.

## 🚀 Erste Schritte

### 1. Basis-Modell erstellen (falls noch nicht vorhanden)
```bash
cd /notebooks/LFM-Liquid-AI-Liquid-Foundation-Models/LFM_3B
python3 municipal_moe_model.py
```
Dies erstellt das Basis-Modell unter `./municipal_moe_base/`

### 2. Modell mit Tokenizer testen
```bash
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_base --prompt "Ich möchte meinen Personalausweis verlängern"
```

### 3. Municipal Demo ausführen
```bash
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_base --municipal-demo
```

Die Demo zeigt Beispiele für:
- 📋 **Einwohnermeldeamt**: Wohnsitz ummelden
- 🏗️ **Bauamt**: Baugenehmigung beantragen  
- 📄 **Standesamt**: Urkunden beantragen
- 🚨 **Ordnungsamt**: Beschwerden melden

### 4. Interaktiver Chat
```bash
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_base --chat
```

Im Chat-Modus können Sie:
- Fragen zu kommunalen Dienstleistungen stellen
- Informationen über Behördengänge erhalten
- Anträge und Verfahren erklärt bekommen
- Mit 'exit' beenden

### 5. Expert-Routing analysieren
```bash
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_base --analyze-routing "Lärmbelästigung melden"
```

Zeigt, welche Experten für eine bestimmte Anfrage aktiviert werden.

## 🏢 Verfügbare Experten

| Expert ID | Zuständigkeit | Beispielthemen |
|-----------|---------------|----------------|
| 0 | Einwohnermeldeamt | An-/Ummeldung, Meldebestätigung |
| 1 | Bauamt | Baugenehmigungen, Bauanzeigen |
| 2 | Ordnungsamt | Ordnungswidrigkeiten, Genehmigungen |
| 3 | Stadtkasse | Gebühren, Steuern, Zahlungen |
| 4 | Sozialamt | Sozialleistungen, Unterstützung |
| 5 | Standesamt | Geburts-/Sterbeurkunden, Eheschließung |
| 6 | Jugendamt | Familienleistungen, Kinderbetreuung |
| 7 | General | Allgemeine Verwaltungsfragen |

## 📝 Weitere Optionen

### Parameter für Textgenerierung
- `--max-length`: Maximale Länge der generierten Antwort (Standard: 100)
- `--temperature`: Kreativität der Antworten (0.1-1.0, Standard: 0.8)

### Beispiele
```bash
# Kurze, präzise Antworten
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_base \
  --prompt "Öffnungszeiten Bürgeramt" --max-length 50 --temperature 0.3

# Ausführliche Erklärungen
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_base \
  --prompt "Wie beantrage ich einen Reisepass?" --max-length 200 --temperature 0.7
```

## 🔧 Training mit deutschen Verwaltungsdaten

Das Basis-Modell muss erst mit deutschen Texten trainiert werden:

### 1. Trainingsdaten erstellen
```bash
python3 train_municipal_moe.py --create-data
```
Erstellt `municipal_training_data.jsonl` mit ~75 deutschen Verwaltungsbeispielen.

### 2. Modell trainieren
```bash
python3 train_municipal_moe.py --model-path ./municipal_moe_base \
  --data-file municipal_training_data.jsonl \
  --output-dir ./municipal_moe_trained \
  --epochs 3 --batch-size 4
```

### 3. Trainiertes Modell verwenden
```bash
# Nach dem Training das trainierte Modell nutzen:
python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_trained --chat
```

### Training-Parameter
- `--epochs`: Anzahl Trainingsdurchläufe (Standard: 3)
- `--batch-size`: Batch-Größe (Standard: 4, bei wenig GPU-Speicher reduzieren)
- `--learning-rate`: Lernrate (Standard: 5e-5)
- `--max-length`: Maximale Textlänge (Standard: 256)

## 📊 Modell-Details
- **Basis**: GPT-2 Tokenizer
- **Architektur**: 6 Transformer-Layer mit MoE
- **Experten**: 8 spezialisierte Netzwerke
- **Aktivierung**: Top-2 Experten pro Token
- **Parameter**: ~47M gesamt