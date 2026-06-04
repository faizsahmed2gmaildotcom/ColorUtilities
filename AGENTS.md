# ColorUtilities AI Agent Guide

## Project Overview
ColorUtilities is a Python-based system for analyzing fabric images to extract color, pattern, and weave information. It uses computer vision and machine learning to process images of textiles, identify dominant colors via k-means clustering, and classify patterns/weaves using PyTorch models.

## Architecture

### Core Components
- **Image Processing** (`pixelLib.py`): Handles pixel extraction, background removal, cropping, median filtering, and salient pixel spreading. Converts images to numpy arrays for analysis.
- **Color Detection** (`imageColorDetector.py`): Main pipeline that processes test images, applies k-means to find dominant colors, and maps them to predefined color names using CIE2000 color distance.
- **Pattern/Weave Classification** (`patternDetectorPyTorch.py`): Loads trained PyTorch models to predict fabric patterns and weaves. Supports ConvNeXt, ResNet, and RetinaNet architectures.
- **Data Management** (`spreadsheetLib.py`): Interfaces with Excel spreadsheets (template.xlsx → result.xlsx) for storing analysis results.
- **Configuration** (`config.py`, `config.toml`): Centralized config loading; defines image sizes, batch sizes, color palettes, and debug flags.
- **Training** (`MLMTrainerPyTorch*.py`): Scripts for training PyTorch models on fabric datasets, including data augmentation and preprocessing layers.

### Data Flow
1. Images from `test-images/` are processed by `imageColorDetector.py`
2. Pixels extracted and filtered via `pixelLib.py`
3. K-means clustering identifies dominant colors
4. Colors matched to names in `config.toml` ["fancy-colors"] or ["primary-colors"]
5. PyTorch models predict pattern/weave from `MLMs/{mode}/`
6. Results saved to `result.xlsx` and displayed via `gui.py`

### Key Directories
- `training-data/pattern/`: Pattern classification training images (subdirs per class)
- `training-data/weave/`: Weave classification training images (subdirs per class)
- `MLMs/{convnext,resnet,retinanet}/`: Trained model weights (.pt files)
- `logs/fit/`: Training logs with timestamps
- `processed-images/`: Intermediate processed images during analysis
- `test-images/`: Input images for color/pattern detection

## Critical Workflows

### Environment Setup
- Set CUDA paths: `export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH`
- Install dependencies: `pip install -r requirements.txt`
- Ensure PyTorch with CUDA support for GPU training/inference

### Training Models
- Run `python MLMTrainerPyTorchConvnext.py` (or ResNet/RetinaNet variants)
- Models saved as `pattern_best_model_{num_classes}.pt` and `weave_best_model_{num_classes}.pt` in `MLMs/{mode}/`
- Training uses data from `training-data/pattern/` and `training-data/weave/`
- Logs output to `logs/fit/YYYYMMDD-HHMMSS/`

### Running Inference
- Place images in `test-images/` (main images end with "_1f" suffix)
- Execute `python imageColorDetector.py` for batch processing
- Results: Console output + `result.xlsx` with columns: sku, Product_Name, color_filter_primary, color_filter_secondary, pattern, weave
- GUI: `python gui.py` displays results in a treeview

### Preparing Training Data
- `prepareTrainingData.py`: Processes Excel files or SQL data to organize images into class subdirs
- Creates `training-data/{pattern, weave}/` structure from spreadsheets

## Project Conventions

### Image Processing
- Images scaled by `img_scale_factor` (default 0.25) for performance
- Background removal assumes white edges; crops by `vertical_offset`/`horizontal_offset` (default 10)
- Median filter with `median_filter_size` (default 5) to reduce noise
- Salient pixels spread with `salient_pixel_bias` (default 10) to emphasize colors

### Color Matching
- Uses CIE2000 color distance for accuracy (via `colormath`)
- Primary colors from `config.toml` ["primary-colors"]; fancy names from ["fancy-colors"]
- K-means with `num_kmeans_centers` (default 1) for dominant color; secondary with 2 centers

### Model Architecture
- Preprocessing includes bilateral blur and FFT low-pass filters (via `kornia`)
- Models fine-tuned from torchvision pretrained weights
- Validation transforms match training: resize to `*_full_size`, random crop to `*_crop_size`
- Batch sizes: `pattern_batches`/`weave_batches` (default 8)

### File Naming
- Model files: `{pattern,weave}_best_model_{num_classes}.pt`
- Training images: Organized in subdirs by class name
- Processed images: Saved to `processed-images/` during debugging

### Debugging
- Set `debug = 1` in `config.toml` for verbose output and intermediate image saves
- Processed images saved as JPEGs in `processed-images/`

## Integration Points
- **External APIs**: `colorFinder.py` scrapes color names from colorhexa.com; `getPantoneColors.py` fetches Pantone colors
- **Libraries**: PyTorch/torchvision for ML; OpenCV for image ops; kornia for advanced filters; colormath for color science
- **Data Sources**: Training data from Excel/CSV; results to Excel
- **Hardware**: CUDA acceleration for PyTorch; CPU fallback available

## Common Patterns
- Config-driven: All parameters in `config.toml`; reload via `config.py`
- Modular processing: Chain functions in `pixelLib.py` for image prep
- Class-based models: `ConvnextModelClassifier`, etc., with custom heads
- Error handling: Try-except in image loading; skip invalid files
- Logging: Timestamps in `logs/`; print statements for progress
