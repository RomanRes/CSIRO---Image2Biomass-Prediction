# CSIRO Biomass Prediction using DINOv3

## Project Overview
This repository contains a deep learning solution developed for the CSIRO Biomass Prediction Kaggle Competition.  
The objective is to predict pasture biomass components from satellite and aerial imagery to support agricultural monitoring and grazing management.

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
The competition dataset consists of satellite and aerial images paired with biomass measurements.

Each image corresponds to one target sample.

### Target Variables
The model predicts three biomass components:
- Dry_Green_g
- Dry_Dead_g
- Dry_Clover_g

---

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
- Stratified Group K-Fold cross-validation

### Image Preprocessing
- Resizing images to 512 x 512
- Normalization
- Standard image augmentations using torchvision

---

## Results
The project demonstrates that large self-supervised vision transformers can be successfully fine-tuned for biomass regression tasks using limited computational resources.

The training pipeline converges stably and scales well despite the large backbone size.

The main focus of this repository is on model design and training methodology rather than leaderboard optimization.

---

## Repository Structure
