#!/bin/bash

# Titanic Random Forest Setup and Run Script
# This script sets up the virtual environment, installs dependencies, and runs the application

set -e  # Exit on any error

echo "🚢 Titanic Random Forest - Setup and Run Script"
echo "==============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
echo "📥 Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "✅ All packages installed successfully"

# Check if Kaggle credentials exist
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "⚠️  Warning: Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "   If you need to download data from Kaggle, please:"
    echo "   1. Go to https://www.kaggle.com/account"
    echo "   2. Create a new API token"
    echo "   3. Place the kaggle.json file in ~/.kaggle/"
    echo "   4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
else
    echo "✅ Kaggle credentials found"
fi

# Check if titanic data directory exists
if [ -d "titanic" ] && [ -f "titanic/train.csv" ]; then
    echo "✅ Titanic dataset already downloaded"
else
    echo "📥 Titanic dataset will be downloaded automatically when running the script"
fi

# Run the application
echo "🚀 Running the Titanic Random Forest model..."
echo "==============================================="
python main.py

echo ""
echo "✅ Script completed successfully!"
echo "🎯 Check the output above to see the Decision Tree model performance:"
echo "   - Mean Absolute Error on validation set"
echo "   - Model trained with max_leaf_nodes=55"
echo ""
echo "💡 The model uses features: Age, SibSp, Parch, LogFare, Pclass, Sex, Embarked"
