# CSIRO Biomass Prediction using DINOv3

## Project Overview
This repository contains a deep learning solution developed for the CSIRO Biomass Prediction Kaggle Competition.
The objective is to predict pasture biomass components from ground-level grass images to support agricultural monitoring and grazing management.

The project focuses on modern computer vision techniques and demonstrates how large self-supervised vision transformers can be adapted efficiently for regression tasks using parameter-efficient fine-tuning.

Competition link: https://www.kaggle.com/competitions/csiro-biomass

---

## Motivation
Instead of training convolutional neural networks from scratch, this project explores the use of foundation vision models pre-trained with self-supervised learning.  
The main goals were:

- Efficient fine-tuning of large vision transformers
- Strong performance with limited GPU memory
- Clean and reproducible training pipeline
- Applicability to real-world remote sensing problems

This repository is presented as part of a professional machine learning portfolio.

---

## Dataset
The dataset consists of ground-level grass images paired with measured biomass values.
These are not satellite images, but close-range images of pasture vegetation.

Each image corresponds to one target sample.

### Target Variables
Initially, the task considered predicting five biomass-related values:

- Dry_Green_g  
- Dry_Dead_g  
- Dry_Clover_g  
- GDM_g  
- Dry_Total_g  

However, two of these targets are deterministic sums of others:
- Dry_Total_g = Dry_Green_g + Dry_Dead_g  
- GDM_g is derived from existing dry matter components  

To avoid redundant learning and improve training stability, the model is trained to predict only the three independent variables:

- Dry_Green_g  
- Dry_Dead_g  
- Dry_Clover_g  

The remaining targets are computed deterministically during post-processing.

## Methodology

### Model Architecture
- Backbone: DINOv3 Vision Transformer (ViT-Huge)
- Pre-trained weights loaded via timm
- Input image resolution: 512 x 512
- Regression head for multi-target prediction

### Fine-Tuning Strategy
- Parameter-efficient fine-tuning using LoRA
- LoRA applied to transformer layers
- Reduced number of trainable parameters
- Lower GPU memory usage and faster convergence

### Training Pipeline
- Framework: PyTorch
- Batch size: 1
- Gradient accumulation: 16 steps
- Effective batch size: 16
- Optimizer: AdamW
- Mixed precision training enabled

### Cross-Validation Strategy
The dataset was split using Stratified Group K-Fold cross-validation to ensure robust evaluation.
Because of computational constraints, only one fold was trained in this experiment.
The training setup supports full K-fold training and was designed with scalability in mind.

### Image Preprocessing
- Resizing images to 512 x 512
- Normalization
- Standard image augmentations using torchvision

### Inference
Inference uses test-time augmentation with four rotational transforms.
The final prediction is obtained by averaging model outputs across all augmentations.
---

## Results
This project was developed as part of the CSIRO Biomass Prediction Kaggle Competition and focuses on model design, training strategy, and efficient fine-tuning of large vision transformers.
---


