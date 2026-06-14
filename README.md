# UWHVF Spatio-Temporal Transformer for Glaucoma Progression 👁️🧠

![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-Ready-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## 📖 Overview

This repository contains a fully reproducible, Kaggle-ready research pipeline for predicting glaucoma visual field (VF) progression. The core innovation is a highly compact, **hybrid dual-branch Spatio-Temporal Transformer (ST-Transformer)**.

Instead of relying solely on hand-crafted global indices, this architecture operates directly on raw 8x9 Humphrey Visual Field grids. It encodes TD and HVF sensitivity through distinct spatial branches, fuses them via gated cross-modal attention, and models longitudinal dynamics using an age-aware temporal Transformer. 

**Key Methodological Contributions:**
- Joint spatial-temporal representation learning on raw VF trajectories.
- Multi-task learning predicting both continuous future MD slope and fast-progressor binary status.
- Uncertainty-aware prediction utilizing evidential regression and MC-Dropout.
- Leakage-aware forward prediction with patient-level grouped splitting to prevent temporal crossover.

## 🧠 Architecture Highlights

| Component | Description | Tech Stack |
|-----------|-------------|------------|
| **Spatial Encoders** | Dual-branch CNN + Self-Attention processing raw 8x9 TD and HVF grids | PyTorch |
| **Cross-Modal Fusion** | Gated fusion mechanism to dynamically blend modality-specific topographic embeddings | PyTorch |
| **Temporal Encoder** | Age-aware temporal Transformer processing variable-length clinical histories | PyTorch |
| **Tabular Adapter** | Gated integration of derived tabular biomarkers alongside raw-grid spatial data | PyTorch |
| **Multi-Task Head** | Evidential regression for MD slope and classification for fast progression | PyTorch |

## 🛠️ Installation & Setup

**Prerequisites**:
- Python 3.12+
- CUDA-enabled GPU (Highly Recommended for `kaggle_fullspec` profile)

```bash
# Clone repository
git clone [https://github.com/yourusername/uwhvf-st-transformer.git](https://github.com/yourusername/uwhvf-st-transformer.git)
cd uwhvf-st-transformer

# Install strict dependencies
pip install torch>=2.2.0 numpy>=1.26.0 pandas>=2.2.0 scikit-learn>=1.4.0 scipy>=1.12.0
pip install matplotlib>=3.8.0 seaborn>=0.13.0 optuna>=3.6.0 shap>=0.45.0
pip install lightgbm>=4.3.0 xgboost>=2.0.0 catboost>=1.2.0 lifelines>=0.29.0
