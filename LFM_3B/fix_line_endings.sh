#!/bin/sh
# Fix Windows line endings in all shell scripts

echo "🔧 Fixing Windows line endings in shell scripts..."

# Fix all shell scripts
for file in *.sh; do
    if [ -f "$file" ]; then
        echo "Fixing $file..."
        sed -i 's/\r$//' "$file"
    fi
done

# Also fix Python files if needed
for file in *.py; do
    if [ -f "$file" ]; then
        # Check if file has Windows line endings
        if grep -q $'\r' "$file" 2>/dev/null; then
            echo "Fixing $file..."
            sed -i 's/\r$//' "$file"
        fi
    fi
done

echo "✅ Fixed line endings"
echo ""
echo "Now you can run:"
echo "sh train_massive_dataset_linux.sh"