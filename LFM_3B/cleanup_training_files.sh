#!/bin/bash
# Cleanup script to free disk space

echo "🧹 Municipal MoE Training Cleanup"
echo "================================="

# Check current disk usage
echo "📊 Current disk usage:"
df -h | grep -E "Filesystem|overlay|ubuntu"
echo ""

# Find large files in training directories
echo "🔍 Finding large training files..."

# List large files
echo "📁 Large files (>100MB):"
find . -name "*.jsonl" -size +100M -exec ls -lh {} \; 2>/dev/null | head -10
find . -name "*.bin" -size +100M -exec ls -lh {} \; 2>/dev/null | head -10
find . -name "*.pt" -size +100M -exec ls -lh {} \; 2>/dev/null | head -10

echo ""
echo "🗑️ Files that can be safely deleted:"

# Training data files (can be regenerated)
echo ""
echo "1️⃣ Training data files (can be regenerated):"
ls -lh *municipal_training_data.jsonl 2>/dev/null || echo "   No training data files found"

# Checkpoints (keep only best_model)
echo ""
echo "2️⃣ Checkpoint directories (keeping only best_model):"
for dir in municipal_moe_*/checkpoint-*; do
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null
    fi
done

# Old training outputs
echo ""
echo "3️⃣ Old training outputs:"
ls -lhd municipal_moe_trained* 2>/dev/null | grep -v "best_model" || echo "   No old outputs found"

# Sample files
echo ""
echo "4️⃣ Sample files:"
ls -lh sample_*.txt 2>/dev/null || echo "   No sample files found"

echo ""
echo "📋 Cleanup Options:"
echo ""
echo "A) Quick cleanup (delete training data only):"
echo "   rm -f *municipal_training_data.jsonl sample_*.txt"
echo ""
echo "B) Medium cleanup (delete data + old checkpoints):"
echo "   rm -f *municipal_training_data.jsonl sample_*.txt"
echo "   rm -rf municipal_moe_*/checkpoint-* (keeps best_model)"
echo ""
echo "C) Full cleanup (delete everything except code):"
echo "   rm -rf municipal_moe_* *municipal_training_data.jsonl sample_*.txt"
echo ""
echo "D) Emergency cleanup (delete MASSIVE dataset - 200MB+):"
echo "   rm -f massive_municipal_training_data.jsonl"
echo "   rm -f ultra_large_municipal_training_data.jsonl"
echo "   rm -f super_large_municipal_training_data.jsonl"
echo ""

# Function to perform cleanup
cleanup_quick() {
    echo "🧹 Performing quick cleanup..."
    rm -f *municipal_training_data.jsonl sample_*.txt
    echo "✅ Deleted training data files"
}

cleanup_medium() {
    echo "🧹 Performing medium cleanup..."
    rm -f *municipal_training_data.jsonl sample_*.txt
    
    # Remove checkpoints but keep best_model
    for dir in municipal_moe_*/checkpoint-*; do
        if [ -d "$dir" ]; then
            echo "   Deleting $dir..."
            rm -rf "$dir"
        fi
    done
    echo "✅ Deleted training data and checkpoints"
}

cleanup_massive() {
    echo "🧹 Performing MASSIVE dataset cleanup..."
    rm -f massive_municipal_training_data.jsonl
    rm -f ultra_large_municipal_training_data.jsonl  
    rm -f super_large_municipal_training_data.jsonl
    rm -f large_municipal_training_data.jsonl
    echo "✅ Deleted all large dataset files"
}

# Interactive menu
echo "💡 Choose cleanup option:"
echo "   1) Quick cleanup (training data only)"
echo "   2) Medium cleanup (data + checkpoints)"
echo "   3) Delete MASSIVE datasets only"
echo "   4) Exit without cleanup"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        cleanup_quick
        ;;
    2)
        cleanup_medium
        ;;
    3)
        cleanup_massive
        ;;
    4)
        echo "❌ Cleanup cancelled"
        ;;
    *)
        echo "❌ Invalid choice"
        ;;
esac

echo ""
echo "📊 Disk usage after cleanup:"
df -h | grep -E "Filesystem|overlay|ubuntu"