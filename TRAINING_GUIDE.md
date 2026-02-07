# Training Guide for Jaundice Detection Model

## Overview
This guide will help you train a model to detect jaundice in images. The model classifies images into two categories: "jaundice" and "normal".

## Prerequisites

1. **Python 3.11+** installed
2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Structure

Your images should be organized as follows:
```
data/
  raw/
    jaundice/     # Images showing jaundice
    normal/       # Images without jaundice
```

## Training Steps

### Step 1: Split the Data
First, split your raw images into training, validation, and test sets:

```bash
python scripts/split_data.py
```

This will create:
```
data/
  processed/
    train/
      jaundice/
      normal/
    val/
      jaundice/
      normal/
    test/
      jaundice/
      normal/
```

### Step 2: Train the Model
Run the training script:

```bash
python train.py
```

Or alternatively:
```bash
python -m src.training.train
```

Or use the shell script:
```bash
bash scripts/train_local.sh
```

## Training Configuration

You can modify training settings in `src/config.py`:

- **BATCH_SIZE**: Number of images per batch (default: 32)
- **LEARNING_RATE**: Learning rate for optimizer (default: 0.001)
- **NUM_EPOCHS**: Maximum number of training epochs (default: 50)
- **MODEL_NAME**: Model architecture - "resnet18", "resnet34", or "resnet50" (default: "resnet18")
- **EARLY_STOPPING_PATIENCE**: Stop training if no improvement for N epochs (default: 10)

## Training Output

During training, you'll see:
- Progress bars for each epoch
- Training and validation loss/accuracy
- Best model saved to `models/best_model.pth`
- Latest checkpoint saved to `models/checkpoint_latest.pth`

## Model Files

After training, you'll find:
- `models/best_model.pth` - Best model based on validation accuracy
- `models/checkpoint_latest.pth` - Latest checkpoint

## Next Steps

Once training is complete, you can:
1. Use the trained model for inference on new images
2. Build a UI to upload and classify images (when ready)

## Troubleshooting

### "No images found" error
- Make sure images are in `data/raw/jaundice/` and `data/raw/normal/`
- Supported formats: .jpg, .jpeg, .png, .bmp

### CUDA/GPU issues
- The model will automatically use CPU if CUDA is not available
- To force CPU, set environment variable: `export CUDA_VISIBLE_DEVICES=""`

### Memory issues
- Reduce BATCH_SIZE in `src/config.py`
- Use a smaller model (resnet18 instead of resnet50)
