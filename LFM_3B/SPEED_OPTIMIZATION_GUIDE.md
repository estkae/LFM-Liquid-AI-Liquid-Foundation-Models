# 🚀 Speed Optimization Guide für Municipal MoE Training

## ⚡ Optimierte Parameter

### Standard (Deine Parameter):
```bash
--epochs 3 --batch-size 4 --learning-rate 5e-5
```
**Geschätzte Zeit: 4-6 Stunden**

### Speed-Optimiert:
```bash
--epochs 2 --batch-size 16 --learning-rate 1e-4 --gradient-accumulation-steps 8 --max-length 128
```
**Geschätzte Zeit: 1-2 Stunden** (3x schneller!)

## 🔧 Optimierungen im Detail

### 1. **Batch Size erhöhen**
- **Von:** `--batch-size 4`
- **Zu:** `--batch-size 16`
- **Effekt:** 4x weniger Iterationen

### 2. **Gradient Accumulation**
- **Standard:** 4 steps × 4 batch = 16 effective
- **Optimiert:** 8 steps × 16 batch = 128 effective
- **Effekt:** Größere effektive Batches = bessere GPU-Auslastung

### 3. **Sequenzlänge reduzieren**
- **Von:** `--max-length 256`
- **Zu:** `--max-length 128`
- **Effekt:** 2x schnellere Attention-Berechnung

### 4. **Learning Rate anpassen**
- **Von:** `5e-5`
- **Zu:** `1e-4`
- **Effekt:** Schnellere Konvergenz bei größeren Batches

### 5. **Weniger Epochen**
- **Von:** 3 Epochen
- **Zu:** 2 Epochen
- **Effekt:** 33% weniger Training

## 🎯 GPU-spezifische Optimierungen

### Für verschiedene GPUs:

#### RTX 3090/4090 (24GB VRAM):
```bash
--batch-size 32 --gradient-accumulation-steps 4 --max-length 128
```

#### RTX 3070/4070 (8-12GB VRAM):
```bash
--batch-size 8 --gradient-accumulation-steps 16 --max-length 128
```

#### V100/A100 (16-40GB VRAM):
```bash
--batch-size 64 --gradient-accumulation-steps 2 --max-length 256
```

## 📊 Performance-Vergleich

| Einstellung | Batch | Grad Accum | Seq Len | Epochen | Zeit |
|------------|-------|------------|---------|---------|------|
| Original | 4 | 4 | 256 | 3 | 4-6h |
| Optimiert | 16 | 8 | 128 | 2 | 1-2h |
| Ultra-Fast | 32 | 4 | 64 | 1 | 30-45min |

## 💡 Weitere Speed-Tricks

### 1. **Mixed Precision Training** (in Code bereits aktiviert)
```python
with torch.cuda.amp.autocast():
    outputs = model(...)
```

### 2. **DataLoader Optimierung** (bereits hinzugefügt)
```python
num_workers=4, pin_memory=True, prefetch_factor=2
```

### 3. **Compile Model** (PyTorch 2.0+)
```python
model = torch.compile(model)  # 10-30% speedup
```

### 4. **Flash Attention** (wenn verfügbar)
```bash
pip install flash-attn
# Automatisch 2x schneller für Attention
```

## ⚠️ Trade-offs

### Speed vs. Qualität:
- ✅ **Schneller:** Weniger Epochen, kürzere Sequenzen
- ❌ **Qualität:** Möglicherweise schlechtere Ergebnisse

### Empfehlung für beste Balance:
```bash
python3 train_municipal_moe_improved.py \
    --model-path ./municipal_moe_base \
    --data-file massive_municipal_training_data.jsonl \
    --output-dir ./municipal_moe_balanced \
    --epochs 2 \
    --batch-size 8 \
    --learning-rate 8e-5 \
    --gradient-accumulation-steps 8 \
    --max-length 192
```
**Zeit: 2-3 Stunden, gute Qualität**

## 🏃 Quick Start

```bash
# Maximale Geschwindigkeit (Qualität kann leiden):
sh train_speed_optimized.sh

# Oder direkt mit allen Optimierungen:
export CUDA_LAUNCH_BLOCKING=0
python3 train_municipal_moe_improved.py \
    --batch-size 16 \
    --gradient-accumulation-steps 8 \
    --max-length 128 \
    --epochs 2
```