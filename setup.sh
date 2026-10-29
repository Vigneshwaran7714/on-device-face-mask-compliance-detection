#!/usr/bin/env bash
# One-time setup (macOS / Linux):  bash setup.sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete."
