#!/usr/bin/env bash
# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH to include src directory
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Launch Isaac Sim with the validation script (adjust path to isaac-sim executable if needed)
isaac-sim.sh --headless --script src/isaac_validation.py
