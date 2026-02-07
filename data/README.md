# Data Directory

This directory contains all datasets for the jaundice detection project.

## Structure

- **`raw/`**: Unmodified dataset downloads
  - Place your original dataset files here
  - Do not modify files in this directory
  - Expected format: Images organized by class (e.g., `jaundice/` and `normal/`)

- **`processed/`**: Cleaned and preprocessed data
  - After running preprocessing scripts, processed images will be stored here
  - Typically organized into `train/`, `val/`, and `test/` subdirectories
  - Each subdirectory should contain class folders

## Expected Classes

- `jaundice`: Images showing signs of jaundice
- `normal`: Images without jaundice signs

## Next Steps

1. Download or obtain a jaundice detection dataset
2. Place raw images in `raw/` directory
3. Run preprocessing scripts to generate processed data in `processed/`
