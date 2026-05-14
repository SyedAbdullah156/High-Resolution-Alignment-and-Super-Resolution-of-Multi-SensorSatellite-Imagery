# High-Resolution Alignment and Super-Resolution of Multi-Sensor Satellite Imagery

## Project Overview

This project implements a comprehensive pipeline for enhancing satellite imagery through **high-resolution alignment** and **super-resolution** techniques. The primary goal is to generate high-quality, aligned satellite images from multi-sensor data sources, enabling better analysis for applications such as environmental monitoring, urban planning, and disaster response.

## Key Components

### 1. Data Processing (`high-resolution-satellite-images-1.ipynb` & `high-resolution-satellite-images-2.ipynb`)

These notebooks handle the preprocessing of raw satellite data:

- **Input Data**: WorldStrat dataset containing Sentinel-2 satellite imagery
- **Processing Steps**:
    - Extract RGB bands (Red: Band 4, Green: Band 3, Blue: Band 2) from multi-band TIFF files
    - Apply brightness gain correction (gain factor: 1.5) to align visual appearance
    - Convert scientific reflectance values (0-10000) to standard pixel values (0-255)
    - Generate paired low-resolution (LR) and high-resolution (HR) image datasets
    - Perform data validation and quality checks to ensure image integrity

### 2. Model Training

#### Primary Training Pipeline (`train.ipynb`)

This notebook implements the core super-resolution training using SwinIR transformer architecture:

- **Architecture**: SwinIR (Swin Transformer for Image Restoration)
- **Configuration**:
    - 4x upscaling factor
    - Input image size: 128x128 (LR), Output: 512x512 (HR)
    - Model parameters: depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6]
- **Training Details**:
    - Loss function: L1 Loss
    - Optimizer: Adam (learning rate: 2e-4)
    - Learning rate scheduler: StepLR (step_size=10, gamma=0.5)
    - Mixed precision training (AMP) for memory efficiency
    - Batch size: 2, Epochs: 50
    - Automatic model checkpointing (best and latest models)

#### Exntended Approaches (`high-resolution-satellite-images-2.ipynb`)

This notebook explores additional super-resolution techniques:

- **Architecture 1**: Histogram Matching (HM) for Image Color Correction
- **Architecture 2**: SR3 (Diffusion-based Generative Model for Super-Resolution)
- **Purpose**: Comparative evaluation of different super-resolution approaches
- **Features**: Includes inference testing and performance comparison between methods

### 3. Model Evaluation (`high-resolution-satellite-images-2.ipynb`)

- **Metrics**: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
- **Test Set**: Held-out images (last 5% of dataset) never seen during training
- **Visualization**: Side-by-side comparison of LR input, SR output, and HR ground truth

### 4. Interactive Visualizations

- **`vision_transf.html`**: Interactive Vision Transformer visualizer showing attention mechanisms and feature processing
- **`visual_guide.html`**: Comprehensive visual guide explaining the super-resolution process and results

### 5. Trained Models (`models/` directory)

- **`model_best.pth`**: Best performing SwinIR model (lowest validation loss)
- **`model_latest.pth`**: Most recent model checkpoint
- **`sr3_best.pth`**: SR3 (Score-based Generative Modeling) FINAL best model
- **`high_res_stn_model5.pth`**: High-resolution Spatial Transformer Network FINAL model

## Technical Details

### Dataset

- **Source**: WorldStrat dataset
- **Sensors**: Sentinel-2 multispectral satellite imagery
- **Bands Used**: RGB composite from Bands 2, 3, 4
- **Resolution**: Low-res: ~10m/pixel, High-res: ~2.5m/pixel (4x difference)

### Dependencies

- PyTorch (deep learning framework)
- Rasterio (satellite image processing)
- OpenCV (image manipulation)
- Timm (PyTorch Image Models)
- SwinIR (official repository cloned during setup)

### Hardware Requirements

- GPU recommended (CUDA-enabled)
- Minimum 8GB VRAM for training
- 16GB+ RAM for data processing

## Results

The trained models achieve:

- **PSNR**: 30-40+ dB (higher is better)
- **SSIM**: 0.8-0.95 (closer to 1.0 is better)
- 4x resolution enhancement while preserving spatial details and spectral characteristics

## Applications

- **Environmental Monitoring**: Enhanced vegetation analysis, deforestation tracking
- **Urban Planning**: Detailed city mapping, infrastructure assessment
- **Disaster Response**: Improved damage assessment from satellite imagery
- **Agriculture**: Precision farming with high-resolution crop monitoring

## Future Enhancements

- Multi-sensor fusion (combining data from different satellites)
- Temporal alignment for time-series analysis
- Real-time processing capabilities
- Integration with GIS systems for geospatial analysis

## Usage

1. Run data preprocessing notebooks to prepare the dataset
2. Execute training notebook to train the super-resolution model
3. Use trained models for inference on new satellite imagery
4. View interactive visualizations to understand model behavior

This project demonstrates the power of modern deep learning techniques in enhancing satellite imagery quality, enabling more accurate and detailed analysis of Earth's surface from space.
