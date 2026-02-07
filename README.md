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

5. **Prepare data**
   - Place raw dataset in `data/raw/`
   - Run preprocessing scripts to generate `data/processed/`

## Next Steps Checklist

- [ ] Obtain/download jaundice detection dataset
- [ ] Implement data preprocessing pipeline (`src/data/`)
- [ ] Define model architecture (`src/training/`)
- [ ] Implement training script with PyTorch
- [ ] Train model and save checkpoints
- [ ] Implement inference utilities (`src/inference/`)
- [ ] Build Streamlit UI (`app/app.py`)
- [ ] Test end-to-end workflow
- [ ] Prepare demo presentation

## Running the Application

Once implemented, run the Streamlit app:
```bash
streamlit run app/app.py
```

## Development

- Configuration: Edit `src/config.py` for paths and constants
- Training: Use `scripts/train_local.sh` or run training scripts directly
- Testing: Add tests in `tests/` directory

## License

MIT License - see LICENSE file for details.

## Contributing

This is a hackathon project. Contributions and improvements welcome!
