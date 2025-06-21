#!/bin/sh
# Simple wrapper script to run the training pipeline

echo "🚀 Starting Municipal MoE Training Pipeline"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "create_massive_municipal_dataset.py" ]; then
    echo "❌ Error: create_massive_municipal_dataset.py not found"
    echo "Please run this script from the LFM_3B directory"
    exit 1
fi

# Check if we have the base model
if [ ! -d "municipal_moe_base" ]; then
    echo "📦 Creating base MoE model..."
    python3 municipal_moe_model.py
fi

# Start the massive training pipeline
echo "🎯 Starting MASSIVE dataset training (500,000+ examples)"
echo "⏱️  Estimated total time: 4-6 hours"
echo "💾 Disk will auto-expand to 2TB if needed"
echo ""
echo "Press Ctrl+C to cancel, or any key to continue..."
read dummy

sh train_massive_dataset_linux.sh