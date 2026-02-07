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

3. **Run the UI**
   ```bash
   python run_app.py
   ```

That's it! The trained model is included, so you can start using the jaundice detection UI immediately.

## What's Included

- ✅ Trained model (`models/best_model.pth`) - Ready to use
- ✅ Complete Streamlit UI (`app/app.py`)
- ✅ Inference module (`src/inference/predictor.py`)
- ✅ All source code and configuration files

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
- The model should be included in the repo. If missing, you'll need to train it first (see `TRAINING_GUIDE.md`)

**Streamlit compatibility issues:**
- Make sure you have Streamlit installed: `pip install streamlit`

## Next Steps

- Upload images through the UI to test jaundice detection
- See `TRAINING_GUIDE.md` if you want to retrain with your own data
- See `UI_GUIDE.md` for detailed UI usage instructions
