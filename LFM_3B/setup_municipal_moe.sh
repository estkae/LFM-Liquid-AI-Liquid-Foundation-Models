#!/bin/bash
# Setup script for Municipal MoE Model

echo "🏛️ Municipal MoE Model Setup"
echo "=========================="

# Step 1: Create base model if it doesn't exist
if [ ! -d "./municipal_moe_base" ]; then
    echo "📦 Creating base model..."
    python3 municipal_moe_model.py
else
    echo "✅ Base model already exists"
fi

# Step 2: Create training data if it doesn't exist
if [ ! -f "./municipal_training_data.jsonl" ]; then
    echo "📝 Creating training data..."
    python3 train_municipal_moe.py --create-data
else
    echo "✅ Training data already exists"
fi

# Step 3: Train the model
if [ ! -d "./municipal_moe_trained" ]; then
    echo "🎓 Training model (this will take 30-60 minutes)..."
    python3 train_municipal_moe.py \
        --model-path ./municipal_moe_base \
        --data-file municipal_training_data.jsonl \
        --output-dir ./municipal_moe_trained \
        --epochs 3 \
        --batch-size 4
else
    echo "✅ Trained model already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Now you can use the trained model:"
echo "python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_trained --prompt \"Personalausweis beantragen\""
echo ""
echo "Or start interactive chat:"
echo "python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_trained --chat"