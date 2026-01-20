#!/bin/bash

# Bio-Oracle Environment Setup Script

set -e  # Exit on error

echo "Initializing Bio-Oracle Environment..."

# 1. Create Python 3.11 Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate and Upgrade Pip
source .venv/bin/activate
pip install --upgrade pip

# 3. Install Core Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Install Optional Development Tools
pip install black pytest

echo "Setup Complete! Activate with: source .venv/bin/activate"
