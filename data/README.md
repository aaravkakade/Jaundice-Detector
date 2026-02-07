# Data Directory

This directory contains all datasets for the jaundice detection project.

## Structure

- **`raw/`**: Manually placed images
  - **Manually place your images here** organized by class
  - Expected folder structure:
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
  - Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

- **`processed/`**: Automatically generated train/val/test splits
  - Created automatically when you run `scripts/split_data.py`
  - Organized into `train/`, `val/`, and `test/` subdirectories
  - Each subdirectory contains class folders (`jaundice/` and `normal/`)

## Expected Classes

- `jaundice`: Images showing signs of jaundice
- `normal`: Images without jaundice signs

## Setup Steps

1. **Manually place images** in `raw/` directory organized by class folders
2. Run `python scripts/split_data.py` to split data into train/val/test sets
3. The processed data will be automatically created in `processed/` directory
