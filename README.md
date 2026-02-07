# Jaundice Detection from Images

A hackathon project for detecting jaundice from medical images using deep learning.

## Overview

This project aims to build a machine learning model that can detect jaundice (yellowing of the skin/eyes) from images. The model will be trained using PyTorch and deployed via a Streamlit web interface for easy demonstration.

## Tech Stack

- **Python 3.11**: Core programming language
- **PyTorch + torchvision**: Deep learning framework for model training and inference
- **Streamlit**: Fast web UI framework for the demo interface
- **Utilities**: numpy, pillow, opencv-python (for image processing)

## Project Structure

```
.
├── data/              # Dataset storage
│   ├── raw/          # Unmodified dataset downloads
│   └── processed/    # Preprocessed data (train/val/test splits)
├── models/           # Saved model checkpoints
├── src/              # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── training/     # Model training scripts
│   ├── inference/    # Model inference utilities
│   └── ui/           # Streamlit UI components
├── scripts/          # Utility scripts (data download, splitting, training)
├── app/              # Main Streamlit application
└── tests/            # Unit and integration tests
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Jaundice-Detector-2
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (required - model files are too large for GitHub)
   ```bash
   # First, split your data (if you have images in data/raw/)
   python scripts/split_data.py
   
   # Then train the model
   python train.py
   ```
   
   **Note:** If you don't have training data, you'll need to add images to `data/raw/jaundice/` and `data/raw/normal/` first.

5. **Run the application**
   ```bash
   python run_app.py
   ```

## Quick Start Workflow

1. **Place your images manually** in `data/raw/`:
   ```
   data/raw/
   ├── jaundice/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── normal/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

2. **Split the data**:
   ```bash
   python scripts/split_data.py
   ```

3. **Train the model**:
   ```bash
   python -m src.training.train
   ```

4. **Check results**: Trained models are saved in `models/` directory

## Running the Application

The Streamlit UI is ready to use! Run it with:

```bash
python run_app.py
```

Or directly:
```bash
streamlit run app/app.py
```

The app will open in your browser where you can upload images and get jaundice predictions.

**Note:** The trained model (`models/best_model.pth`) is included in the repository, so you can run the UI immediately without training.

## Training Your Own Model

If you want to retrain the model with your own data:

1. **Place your images** in `data/raw/`:
   ```
   data/raw/
   ├── jaundice/  (images showing jaundice)
   └── normal/    (images without jaundice)
   ```

2. **Split the data**:
   ```bash
   python scripts/split_data.py
   ```

3. **Train the model**:
   ```bash
   python train.py
   ```

See `TRAINING_GUIDE.md` for detailed training instructions.

## Configuration

Edit `src/config.py` to customize:
- Training hyperparameters (batch size, learning rate, epochs)
- Model architecture (resnet18, resnet34, resnet50)
- Data split ratios
- Early stopping patience

## Development

- **Training**: Use `python -m src.training.train` or `./scripts/train_local.sh`
- **Data splitting**: Run `python scripts/split_data.py` after placing images
- **Testing**: Add tests in `tests/` directory

## License

MIT License - see LICENSE file for details.

## Contributing

This is a hackathon project. Contributions and improvements welcome!
