# Integrating Optical and InSAR Data for Machine Learning-Based Rock Glacier Activity Classification: Model Evaluation Across Mountain Regions

This repository contains the official PyTorch implementation and dataset specifications for the paper:
This project utilizes deep learning (**ResNet-101**) and machine learning (**Random Forest**) to automatically classify the activity status of rock glaciers (*Intact* vs. *Relict*) using high-resolution Planet satellite imagery.
## Key Features
* **Model Architecture:** ResNet-101 and Random Forest (Grid Search optimized).
* **Validation Strategy:** Stratified 5-Fold Cross-Validation to ensure robust performance evaluation.
* **Inference Logic:** Includes a **Majority Voting** aggregation mechanism to classify rock glacier polygons based on multiple sub-image crops.

## 📂 Repository Structure
* `main.py`: The main script for training the ResNet-101 model using 5-fold cross-validation.
* `model.py`: Definition of the ResNet-101 architecture (customized for binary classification).
* `inference.py`: Performs polygon-level classification using the Majority Voting strategy (corresponding to the paper's methodology).
* `predict.py`: A general-purpose script to predict the status of a single image or a directory of images.
* `rf_main.py`: The implementation for the Random Forest classifier including feature extraction (GLCM, Spectral) and Grid Search.
