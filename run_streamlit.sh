#!/bin/bash
# Simple script to run the enhanced Streamlit interface for Higgs Audio

set -e

# Configuration
VENV_NAME=".venv"
STREAMLIT_PORT=8501
HOST="0.0.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}⚡ Faster Higgs Audio v2 - Streamlit Interface${NC}"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_NAME${NC}"
    echo -e "${YELLOW}Please set up the environment first:${NC}"
    echo -e "${YELLOW}  uv venv --python 3.10 && source .venv/bin/activate${NC}"
    echo -e "${YELLOW}  uv pip install -r requirements.txt -e . bitsandbytes streamlit loguru${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source $VENV_NAME/bin/activate

# Check if enhanced interface exists
if [ ! -f "streamlit_interface.py" ]; then
    echo -e "${RED}❌ Enhanced Streamlit interface not found!${NC}"
    exit 1
fi

# Display system info
echo -e "${BLUE}📊 System Information:${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ CUDA detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  CUDA not detected. Will use CPU mode${NC}"
fi

# Test dependencies
echo -e "${BLUE}🧪 Testing dependencies...${NC}"
python -c "
import sys
try:
    import streamlit as st
    import torch
    from pathlib import Path
    sys.path.insert(0, '.')
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        print('✅ Local model mode available')
    except ImportError:
        print('⚠️ Local model mode disabled, API-only')
    print('✅ Basic dependencies OK')
except Exception as e:
    print(f'❌ Dependency error: {e}')
    exit(1)
"

# Check if port is in use
if lsof -Pi :$STREAMLIT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port $STREAMLIT_PORT is already in use${NC}"
    echo -e "${YELLOW}Killing existing process...${NC}"
    kill $(lsof -ti:$STREAMLIT_PORT) 2>/dev/null || true
    sleep 2
fi

echo -e "${GREEN}🎨 Enhanced Streamlit Interface Features:${NC}"
echo -e "  - 🔧 Local model loading with 4-bit/8-bit quantization"
echo -e "  - 📡 API server mode compatibility"
echo -e "  - 🎛️ Comprehensive parameter control (temperature, top-k, top-p, etc.)"
echo -e "  - 👥 Voice cloning with preset and custom voices"
echo -e "  - 🎭 Multi-speaker generation with [SPEAKER0], [SPEAKER1] tags"
echo -e "  - 📊 Generation history and audio validation"
echo -e "  - 💾 Persistent model loading for faster generation"
echo -e "  - 🎵 Real-time audio validation with RMS levels"

# Start Streamlit interface
echo -e "\n${BLUE}🚀 Starting Enhanced Streamlit Interface...${NC}"
echo -e "Host: $HOST"
echo -e "Port: $STREAMLIT_PORT"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    kill $(lsof -ti:$STREAMLIT_PORT) 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Set environment variable to disable usage stats
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit
streamlit run streamlit_interface.py \
    --server.port $STREAMLIT_PORT \
    --server.address $HOST \
    --server.headless true \
    --server.runOnSave false \
    --server.allowRunOnSave false \
    --browser.gatherUsageStats false \
    --theme.base "light" \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6" &

STREAMLIT_PID=$!

# Wait for Streamlit to start
echo -e "${BLUE}⏳ Waiting for Streamlit to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$STREAMLIT_PORT >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Streamlit interface is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Streamlit failed to start within 30 seconds${NC}"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Display access information
echo -e "\n${GREEN}🎉 Enhanced Interface is ready!${NC}"
echo "============================================"
echo -e "${BLUE}🌐 Access URLs:${NC}"
echo "  - Local: http://localhost:$STREAMLIT_PORT"

# Get local IP for network access
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "  - Network: http://$LOCAL_IP:$STREAMLIT_PORT"

echo ""
echo -e "${GREEN}💡 Usage Tips:${NC}"
echo "  - Use 'Local Model' mode to load models directly with quantization"
echo "  - Start with 4-bit quantization for 8GB GPUs"
echo "  - Try 8-bit quantization for better quality on larger GPUs"
echo "  - Use voice presets like 'belinda' or 'broom_salesman' for cloning"
echo "  - Enable 'Static KV Cache' for faster generation (uses more memory)"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the interface${NC}"

# Wait for user interrupt
wait $STREAMLIT_PID