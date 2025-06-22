#!/bin/sh
# Quick script to use the best trained model

echo "🏆 Using Best Municipal MoE Model"
echo "================================"

# Find best models
echo "🔍 Searching for trained models..."
python3 find_best_model.py

echo ""
echo "📋 Quick Usage Commands:"
echo ""

# Common best model paths
MODELS="
./municipal_moe_massive_trained/best_model
./municipal_moe_speed_trained/best_model
./municipal_moe_ultra_trained/best_model
./municipal_moe_trained_v2/best_model
./municipal_moe_large_trained/best_model
"

# Find first existing model
BEST_MODEL=""
for model in $MODELS; do
    if [ -d "$model" ]; then
        BEST_MODEL="$model"
        break
    fi
done

if [ -n "$BEST_MODEL" ]; then
    echo "✅ Found model: $BEST_MODEL"
    echo ""
    echo "1️⃣ Test with a question:"
    echo "python3 municipal_tokenizer_integration.py --model-path $BEST_MODEL --prompt \"Frage: Was kostet eine Geburtsurkunde?\\nAntwort:\""
    echo ""
    echo "2️⃣ Interactive chat:"
    echo "python3 municipal_tokenizer_integration.py --model-path $BEST_MODEL --chat"
    echo ""
    echo "3️⃣ Run demo:"
    echo "python3 municipal_tokenizer_integration.py --model-path $BEST_MODEL --municipal-demo"
    echo ""
    echo "4️⃣ Analyze expert routing:"
    echo "python3 municipal_tokenizer_integration.py --model-path $BEST_MODEL --analyze-routing \"Baugenehmigung beantragen\""
else
    echo "❌ No trained models found!"
    echo ""
    echo "Train a model first with:"
    echo "  sh train_speed_optimized.sh"
fi