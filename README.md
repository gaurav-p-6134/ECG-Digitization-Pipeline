# ECG Image Digitization Pipeline 🫀⚡

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Optimized-brightgreen)

## 📖 Overview
This project is a high-performance Computer Vision pipeline designed to digitize paper-based ECG records into clean 1D signals. In healthcare, millions of historical ECGs exist only on paper; digitizing them is crucial for building large-scale AI diagnostic models.

This repository contains an end-to-end inference pipeline that:
1.  **Segments** the ECG graph from scanned documents.
2.  **Rectifies** spatial distortions (rotation, skew, warping).
3.  **Extracts** the 1D voltage signal from the visual graph.

## 🚀 Key Technical Achievements
The core challenge was processing high-resolution images (~2200x1700) on limited hardware (T4 GPU, 16GB VRAM) under strict time constraints.

* **Memory Optimization:** Solved OOM (Out-of-Memory) errors by implementing **Dynamic Smart Batching**, processing smaller images in batches of 4 while automatically switching to serial processing for high-res inputs.
* **Speed Optimization:** Reduced inference time by **~40%** (from 9hrs down to ~5hrs) using a threaded Producer-Consumer pattern (`DataLoader` prefetching) and **FP16 Mixed Precision** inference.
* **Accuracy:** Achieved a competitive error score of **18.33** (MAE) by using Test-Time Augmentation (TTA) and robust signal post-processing.

## 🧠 Pipeline Architecture

### Stage 0: Segmentation (U-Net)
* **Input:** Raw Scanned Image
* **Model:** Lightweight U-Net
* **Task:** Identifies the region of interest (ROI) containing the lead signals, removing background noise and text.

### Stage 1: Rectification (Spatial Transformer)
* **Input:** Segmented Mask
* **Model:** Custom ResNet-based regressor
* **Task:** Predicts control points to un-warp the image, correcting for scanner skew or crumpled paper.
* **Optimization:** Uses "Pad-Inference-Crop" logic to handle variable aspect ratios without resizing artifacts.

### Stage 2: Signal Extraction (1D Regression)
* **Input:** Rectified Crop
* **Model:** ResNet34 Encoder + Custom Coordinate Decoder
* **Task:** Converts pixel intensity maps into 1D voltage arrays.
* **Optimization:** Ensemble of 5 TTA versions (Normal, Dark, Bright, Sharpened, CLAHE) to handle faint or degraded ink lines.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/ecg-digitization.git](https://github.com/YourUsername/ecg-digitization.git)
    cd ecg-digitization
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have PyTorch with CUDA support installed for GPU acceleration.)*

## ⚡ Usage

To run the full pipeline on a folder of test images:

```bash
# 1. Run Segmentation
python src/stage0.py

# 2. Run Rectification (Input: Stage 0 outputs)
python src/stage1.py

# 3. Run Signal Extraction (Input: Stage 1 outputs)
python src/stage2.py
