# Quick Setup Guide

## For New Users (Clone and Run)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Jaundice-Detector-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (required)
   ```bash
   # If you have images in data/raw/, split them first:
   python scripts/split_data.py
   
   # Then train:
   python train.py
   ```
   
   **Note:** You need training images in `data/raw/jaundice/` and `data/raw/normal/` directories.

4. **Run the UI**
   ```bash
   python run_app.py
   ```

## What's Included

- ✅ Complete Streamlit UI (`app/app.py`)
- ✅ Inference module (`src/inference/predictor.py`)
- ✅ Training scripts and utilities
- ✅ All source code and configuration files

**Note:** Model files are not included (too large for GitHub). You'll need to train the model first.

## Requirements

- Python 3.11+
- PyTorch
- Streamlit
- PIL/Pillow
- See `requirements.txt` for full list

## Troubleshooting

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**"Model not found" error:**
- You need to train the model first: `python train.py`
- Make sure you have images in `data/raw/jaundice/` and `data/raw/normal/`
- See `TRAINING_GUIDE.md` for detailed instructions

**Streamlit compatibility issues:**
- Make sure you have Streamlit installed: `pip install streamlit`

## Next Steps

- Upload images through the UI to test jaundice detection
- See `TRAINING_GUIDE.md` if you want to retrain with your own data
- See `UI_GUIDE.md` for detailed UI usage instructions
