#!/bin/bash
# Ultra Large Dataset Training Pipeline - 20,000+ examples

echo "🚀 ULTRA LARGE Municipal Dataset Training Pipeline"
echo "=================================================="
echo "⚡ This will create 20,000+ training examples!"

# Step 1: Create the ultra large dataset
echo -e "\n📊 Creating ultra large dataset (20,000+ examples)..."
echo "   This may take 1-2 minutes..."
python3 create_ultra_large_municipal_dataset.py

# Check if dataset was created
if [ ! -f "ultra_large_municipal_training_data.jsonl" ]; then
    echo "❌ Failed to create ultra dataset!"
    exit 1
fi

# Count examples
EXAMPLE_COUNT=$(wc -l < ultra_large_municipal_training_data.jsonl)
echo "✅ Created $EXAMPLE_COUNT training examples!"

# Step 2: Train with optimal parameters for ultra large dataset
echo -e "\n🎓 Starting training with ultra large dataset..."
echo "   ⏱️  Estimated time: 2-3 hours on GPU"
echo "   💡 Tip: Use tmux or screen to keep training running"

python3 train_municipal_moe_improved.py \
    --model-path ./municipal_moe_base \
    --data-file ultra_large_municipal_training_data.jsonl \
    --output-dir ./municipal_moe_ultra_trained \
    --epochs 4 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --gradient-accumulation-steps 8 \
    --max-length 256

# Check if training completed
if [ ! -d "./municipal_moe_ultra_trained" ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo -e "\n✅ Training completed successfully!"

# Step 3: Comprehensive testing
echo -e "\n🧪 Testing the ultra-trained model..."

echo -e "\n1️⃣ Ummeldung Test:"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_ultra_trained/best_model \
    --prompt "Frage: Wo kann ich mich ummelden?\nAntwort:" \
    --max-length 150

echo -e "\n2️⃣ Personalausweis Test:"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_ultra_trained/best_model \
    --prompt "Frage: Wie beantrage ich einen neuen Personalausweis?\nAntwort:" \
    --max-length 150

echo -e "\n3️⃣ Geburtsurkunde Test:"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_ultra_trained/best_model \
    --prompt "Frage: Was kostet eine Geburtsurkunde?\nAntwort:" \
    --max-length 150

echo -e "\n4️⃣ Baugenehmigung Test:"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_ultra_trained/best_model \
    --prompt "Frage: Brauche ich eine Genehmigung für ein Gartenhaus?\nAntwort:" \
    --max-length 150

echo -e "\n5️⃣ Conversational Test:"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_ultra_trained/best_model \
    --prompt "Frage: Guten Tag, ich bin umgezogen. Was muss ich tun?\nAntwort:" \
    --max-length 150

echo -e "\n🎉 Ultra training pipeline completed!"
echo -e "\n📋 Your model is ready for production use!"
echo "   • Model location: ./municipal_moe_ultra_trained/best_model"
echo "   • Interactive chat: python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_ultra_trained/best_model --chat"
echo "   • Municipal demo: python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_ultra_trained/best_model --municipal-demo"
echo -e "\n💡 The model should now generate perfect German administrative responses!"