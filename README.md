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

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository-url>
   cd Jaundice-Detector
   ```

2. **Create a virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Prepare and train the model**
   ```bash
   # Step 1: Manually place images in data/raw/ organized by class
   # Structure should be:
   #   data/raw/jaundice/  (images showing jaundice)
   #   data/raw/normal/     (images without jaundice)
   
   # Step 2: Split data into train/val/test sets
   python scripts/split_data.py
   
   # Step 3: Train the model
   python -m src.training.train
   # Or use the script:
   # ./scripts/train_local.sh
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

## Next Steps Checklist

- [x] Data preprocessing pipeline (`src/data/`)
- [x] Model architecture and training (`src/training/`)
- [ ] Implement inference utilities (`src/inference/`)
- [ ] Build Streamlit UI (`app/app.py`)
- [ ] Test end-to-end workflow
- [ ] Prepare demo presentation

## Running the Application

Once implemented, run the Streamlit app:
```bash
streamlit run app/app.py
```

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
