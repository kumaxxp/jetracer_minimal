#!/bin/bash

# Phase 1 Week 1 Implementation Verification Script
# Adapted for virtual environment (.venv)

echo "=========================================="
echo "Phase 1 Week 1 - Implementation Check"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass=0
check_fail=0

# Function to check file existence
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((check_pass++))
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        ((check_fail++))
    fi
}

# Function to check directory existence
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        ((check_pass++))
    else
        echo -e "${RED}✗${NC} $1/ (missing)"
        ((check_fail++))
    fi
}

echo "Checking directory structure..."
echo ""

# Check main directories
check_dir "data"
check_dir "data/raw_images"
check_dir "data/annotations"
check_dir "data/datasets"
check_dir "data/label_studio"
check_dir "models"
check_dir "models/custom_trained"
check_dir "scripts"
check_dir "tests"
check_dir "docs"
check_dir "configs"

echo ""
echo "Checking core files..."
echo ""

# Check core Python files
check_file "main.py"
check_file "camera.py"
check_file "segmentation.py"
check_file "monitor.py"
check_file "web_ui.py"

echo ""
echo "Checking configuration files..."
echo ""

# Check config files
check_file "configs/config.yaml"
check_file "requirements.txt"
check_file ".gitignore"

echo ""
echo "Checking documentation..."
echo ""

# Check documentation
check_file "README.md"
check_file "docs/USAGE_DataCollection.md"

echo ""
echo "Checking tests..."
echo ""

# Check tests
check_file "tests/test_data_collection.py"

echo ""
echo "Checking for data collection features in web_ui.py..."
echo ""

# Check for key methods in web_ui.py
if grep -q "def setup_data_collection_panel" web_ui.py; then
    echo -e "${GREEN}✓${NC} setup_data_collection_panel() method"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} setup_data_collection_panel() method (missing)"
    ((check_fail++))
fi

if grep -q "def start_collection_session" web_ui.py; then
    echo -e "${GREEN}✓${NC} start_collection_session() method"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} start_collection_session() method (missing)"
    ((check_fail++))
fi

if grep -q "def capture_frame" web_ui.py; then
    echo -e "${GREEN}✓${NC} capture_frame() method"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} capture_frame() method (missing)"
    ((check_fail++))
fi

if grep -q "def end_collection_session" web_ui.py; then
    echo -e "${GREEN}✓${NC} end_collection_session() method"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} end_collection_session() method (missing)"
    ((check_fail++))
fi

echo ""
echo "Checking Python environment..."
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment (.venv) exists"
    ((check_pass++))
else
    echo -e "${YELLOW}⚠${NC} Virtual environment (.venv) not found (optional)"
    echo "  Note: Will use system Python if .venv not present"
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Test imports
python3 -c "import nicegui" 2>/dev/null
if [ $? -eq 0 ]; then
    VERSION=$(python3 -c "import nicegui; print(nicegui.__version__)")
    echo -e "${GREEN}✓${NC} NiceGUI installed (version: $VERSION)"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} NiceGUI not installed"
    ((check_fail++))
fi

python3 -c "import cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
    echo -e "${GREEN}✓${NC} OpenCV installed (version: $VERSION)"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} OpenCV not installed"
    ((check_fail++))
fi

python3 -c "import numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    echo -e "${GREEN}✓${NC} NumPy installed (version: $VERSION)"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} NumPy not installed"
    ((check_fail++))
fi

python3 -c "import yaml" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyYAML installed"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} PyYAML not installed"
    ((check_fail++))
fi

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo -e "Passed: ${GREEN}${check_pass}${NC}"
echo -e "Failed: ${RED}${check_fail}${NC}"
echo ""

if [ $check_fail -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start application: ./start_jetracer.sh"
    echo "2. Access UI: http://$(hostname -I | awk '{print $1}'):8080"
    echo "3. Connect camera and test data collection"
    echo ""
    exit 0
else
    echo -e "${RED}✗✗✗ Some checks failed. Please review.${NC}"
    echo ""
    exit 1
fi