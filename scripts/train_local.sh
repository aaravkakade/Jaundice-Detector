#!/bin/bash
# Script to train the jaundice detection model locally

echo "=========================================="
echo "Jaundice Detection - Local Training"
echo "=========================================="

# Check if data is split
if [ ! -d "data/processed/train" ]; then
    echo "Error: Processed data not found!"
    echo "Please run 'python scripts/split_data.py' first to split your data."
    exit 1
fi

# Run training
echo "Starting training..."
python -m src.training.train

echo ""
echo "Training complete! Check models/ directory for saved checkpoints."
