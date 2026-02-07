# Jaundice Detection UI Guide

## Overview
A Streamlit web application that allows you to upload images and detect jaundice using the trained deep learning model.

## Running the Application

### Option 1: Using the run script (Recommended)
```bash
python run_app.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run app/app.py
```

### Option 3: Using the module path
```bash
python -m streamlit run app/app.py
```

## What You'll See

1. **Upload Interface**: Drag and drop or browse to upload an image
2. **Image Preview**: Your uploaded image will be displayed
3. **Prediction Results**: 
   - Class prediction (Normal or Jaundice)
   - Confidence score (0-100%)
   - Probability breakdown for both classes
4. **Detailed View**: Expandable section with raw probabilities

## Features

- ✅ Real-time image analysis
- ✅ Confidence scores and probability breakdowns
- ✅ Clean, user-friendly interface
- ✅ Color-coded results (green for normal, red for jaundice)
- ✅ Responsive design

## Requirements

Make sure you have:
- Trained model at `models/best_model.pth`
- All dependencies installed: `pip install -r requirements.txt`

## Troubleshooting

### "Model not found" error
- Make sure you've trained the model first: `python train.py`
- Check that `models/best_model.pth` exists

### Import errors
- Make sure you're running from the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Image upload issues
- Supported formats: JPG, JPEG, PNG, BMP
- Try a different image if one fails to load

## Notes

⚠️ **Important**: This tool is for demonstration purposes only. Always consult a healthcare professional for medical diagnosis.
