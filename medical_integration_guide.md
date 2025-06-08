# Medical Expert Integration Guide für LFM

## Schritt-für-Schritt Anleitung

### 1. Medizinische Daten vorbereiten

```bash
# Schritt 1.1: Rohdaten sammeln
# Erstelle einen Ordner für deine medizinischen Daten
mkdir -p data/medical
cd data/medical

# Schritt 1.2: Daten vorbereiten mit dem prepare_medical_data.py Script
python prepare_medical_data.py \
    --input_dir ./raw_data \
    --output_file ../train_medical.jsonl \
    --specialties cardiology,neurology,radiology \
    --min_confidence 0.8
```

**Datenformat (JSONL):**
```json
{
  "text": "Patient presents with chest pain...",
  "specialty": "cardiology",
  "urgency": "high",
  "metadata": {
    "source": "clinical_notes",
    "validated": true
  }
}
```

### 2. Basis-LFM Model vorbereiten

```bash
# Schritt 2.1: LFM-3B Model laden
python -c "
from lfm.model import LFModel
from lfm.config import LFMConfig

# Basis-Konfiguration für LFM-3B
config = LFMConfig(
    hidden_dim=3072,
    num_layers=32,
    num_experts=8,  # Standard MoE
    model_size='3B'
)

# Model initialisieren
model = LFModel(config)
print(f'Model loaded: {model.get_num_params():,} parameters')
"

# Schritt 2.2: Checkpoint herunterladen (falls vorhanden)
# wget https://your-model-repository/lfm-3b-base.pt
```

### 3. Medical MoE konfigurieren

```python
# config_medical_model.py
from lfm.config import LFMConfig
from lfm.medical_moe import MedicalMoE

# Medical-spezifische Konfiguration
medical_config = LFMConfig(
    hidden_dim=3072,
    num_layers=32,
    num_experts=12,  # 12 medizinische Experten
    num_experts_per_token=2,  # Kann auf 4 erhöht werden bei Urgency
    model_size='3B',
    # Medical-spezifische Parameter
    medical_specialties=[
        'cardiology', 'neurology', 'oncology', 'radiology',
        'pathology', 'pharmacology', 'emergency', 'pediatrics',
        'surgery', 'psychiatry', 'internal_medicine', 'general'
    ],
    use_safety_gates=True,
    confidence_threshold=0.85,
    evidence_extraction=True
)
```

### 4. Fine-tuning durchführen

```bash
# Schritt 4.1: Training starten
python train_medical_lfm.py \
    --model_size 3B \
    --data_path data/train_medical.jsonl \
    --output_dir ./medical_checkpoints \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --warmup_steps 1000 \
    --specialties all \
    --use_lora True \
    --lora_rank 64
```

**Training-Parameter erklärt:**
- `--use_lora`: Efficient fine-tuning mit LoRA (Low-Rank Adaptation)
- `--specialties`: Welche medizinischen Fachgebiete trainiert werden sollen
- `--batch_size`: Reduziert für GPU-Speicher (bei 3B Model)

### 5. Experten integrieren und testen

```python
# test_medical_integration.py
import torch
from lfm.model_v2 import LFModel
from lfm.medical_moe import MedicalMoE

# Model mit Medical MoE laden
model = LFModel.from_pretrained(
    "medical_checkpoints/best_model",
    use_medical_moe=True
)

# Test-Beispiel
test_input = """
Patient: 45-year-old male
Symptoms: Severe chest pain, shortness of breath, sweating
Duration: 30 minutes
Medical History: Hypertension, smoking
"""

# Inference mit Medical MoE
with torch.no_grad():
    output = model.generate(
        test_input,
        max_length=200,
        return_expert_info=True  # Zeigt welche Experten aktiviert wurden
    )
    
print("Diagnose:", output['text'])
print("Aktivierte Experten:", output['expert_info']['active_experts'])
print("Konfidenz:", output['expert_info']['confidence'])
print("Urgency Score:", output['expert_info']['urgency'])
```

### 6. Evaluierung

```bash
# Schritt 6.1: Medical Safety Compliance prüfen
python medical_safety_compliance.py \
    --model_path ./medical_checkpoints/best_model \
    --test_suite comprehensive \
    --output_report ./safety_report.json

# Schritt 6.2: Performance evaluieren
python evaluate_medical_model.py \
    --model_path ./medical_checkpoints/best_model \
    --test_data ./data/test_medical.jsonl \
    --metrics accuracy,f1,expert_usage,safety_violations
```

### 7. Deployment

```python
# deploy_medical_model.py
from gradio_app import create_medical_interface

# Gradio App für Medical Model starten
app = create_medical_interface(
    model_path="./medical_checkpoints/best_model",
    enable_safety_checks=True,
    log_interactions=True,
    specialties=['cardiology', 'emergency', 'internal_medicine']
)

app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Für öffentlichen Zugriff
)
```

## Wichtige Überlegungen

### Datenqualität
- Medizinische Daten müssen validiert und anonymisiert sein
- Mindestens 10k Beispiele pro Fachgebiet empfohlen
- Balance zwischen Fachgebieten wichtig

### Training-Tipps
1. **Schrittweises Vorgehen**: Erst einzelne Fachgebiete, dann alle
2. **Monitoring**: Expert usage und load balancing überwachen
3. **Safety First**: Immer Safety Gates aktiviert lassen
4. **Evaluation**: Medizinische Experten zur Validierung einbeziehen

### Resource Requirements
- **GPU**: Minimum 24GB VRAM für 3B Model Fine-tuning
- **Training Zeit**: ~24-48 Stunden auf A100
- **Speicher**: ~15GB für Model + Checkpoints

### Nächste Schritte nach Integration
1. A/B Testing mit Standard LFM vs Medical LFM
2. Feedback von medizinischem Personal einholen
3. Kontinuierliches Learning mit neuen Fällen
4. Spezialisierung auf bestimmte Fachgebiete vertiefen