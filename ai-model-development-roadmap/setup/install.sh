#!/bin/bash

# AI Model Development Roadmap - Setup Script
# This script installs all required dependencies for the 12-week journey

set -e  # Exit on error

echo "=========================================="
echo "AI Model Development Roadmap Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo "❌ Error: Python 3.8+ required. Found Python $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION found"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "✓ CUDA $CUDA_VERSION found"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "✓ GPU: $GPU_NAME"
else
    echo "⚠ CUDA not found. CPU-only mode will be used."
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio
fi

# Install core libraries
echo ""
echo "Installing core AI libraries..."
pip install transformers>=4.35.0
pip install diffusers>=0.21.0
pip install accelerate>=0.24.0
pip install datasets>=2.14.0
pip install tokenizers>=0.14.0

# Install training & optimization
echo ""
echo "Installing training & optimization libraries..."
pip install deepspeed>=0.11.0
pip install bitsandbytes>=0.41.0
pip install peft>=0.6.0

# Flash attention (may fail on some systems)
echo ""
echo "Installing Flash Attention (optional)..."
pip install flash-attn --no-build-isolation || echo "⚠ Flash Attention installation failed (optional)"

# Install monitoring & logging
echo ""
echo "Installing monitoring tools..."
pip install wandb>=0.15.0
pip install tensorboard>=2.14.0

# Install production tools
echo ""
echo "Installing production tools..."
pip install onnx>=1.14.0
pip install onnxruntime>=1.16.0
pip install fastapi>=0.104.0
pip install uvicorn>=0.24.0

# Install additional utilities
echo ""
echo "Installing utilities..."
pip install numpy>=1.24.0
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.66.0
pip install jupyterlab>=4.0.0
pip install ipywidgets>=8.1.0

# Install safetensors
pip install safetensors>=0.4.0

# Install HuggingFace Hub
pip install huggingface-hub>=0.19.0

# Create requirements.txt
echo ""
echo "Creating requirements.txt..."
pip freeze > requirements.txt

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Start with Phase 0:"
echo "   cd phase0-huggingface-ecosystem"
echo "3. Happy learning! 🚀"
echo ""
