#!/bin/bash
# Optimized training pipeline for super large municipal dataset

echo "🚀 Super Large Municipal Dataset Training Pipeline"
echo "================================================="

# Step 1: Create the super large dataset
echo "📊 Creating super large dataset (2000+ examples)..."
python3 create_super_large_municipal_dataset.py

# Check if dataset was created successfully
if [ ! -f "super_large_municipal_training_data.jsonl" ]; then
    echo "❌ Failed to create dataset!"
    exit 1
fi

echo "✅ Super large dataset created!"

# Step 2: Train with optimized parameters for large dataset
echo "🎓 Starting training with super large dataset..."
echo "   This will take 60-90 minutes depending on your GPU..."

python3 train_municipal_moe_improved.py \
    --model-path ./municipal_moe_base \
    --data-file super_large_municipal_training_data.jsonl \
    --output-dir ./municipal_moe_super_trained \
    --epochs 6 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --gradient-accumulation-steps 8 \
    --max-length 256

# Check if training completed successfully
if [ ! -d "./municipal_moe_super_trained" ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ Training completed successfully!"

# Step 3: Test the super trained model
echo "🧪 Testing the super trained model..."

echo "Test 1: Standard question"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_super_trained/best_model \
    --prompt "Frage: Wie beantrage ich einen Personalausweis?\nAntwort:"

echo -e "\nTest 2: Ummeldung question"  
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_super_trained/best_model \
    --prompt "Frage: Wo kann ich mich ummelden?\nAntwort:"

echo -e "\nTest 3: Bauamt question"
python3 municipal_tokenizer_integration.py \
    --model-path ./municipal_moe_super_trained/best_model \
    --prompt "Frage: Brauche ich eine Baugenehmigung für einen Wintergarten?\nAntwort:"

echo -e "\n🎉 Super training pipeline completed!"
echo -e "\n📋 Next steps:"
echo "   • Test more examples with different prompts"
echo "   • Try the municipal demo: python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_super_trained/best_model --municipal-demo"
echo "   • Use interactive chat: python3 municipal_tokenizer_integration.py --model-path ./municipal_moe_super_trained/best_model --chat"