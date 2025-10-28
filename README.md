# Multimodal Fusion for Deception Detection

> A modular multimodal pipeline for deception (lie) detection using **ATSFace** features. Visual, audio, and textual streams are modeled independently and then combined via a **softmax-gated fusion** mechanism for the final classification.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training Setup](#training-setup)
- [Per-Modality Models](#per-modality-models)
  - [Visual](#visual)
  - [Audio](#audio)
  - [Text](#text)
- [Multimodal Fusion](#multimodal-fusion)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Ethical Considerations](#ethical-considerations)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

---

## Overview
This project explores **multimodal integration** for lie detection with precomputed features from **ATSFace**:

- **Visual:** 128-D FaceNet embeddings per frame  
- **Text:** 128-D sentence embeddings (BERT), averaged per clip  
- **Audio:** 20-D MFCC features with 0.2s windows

Each modality is trained with its own lightweight Transformer/MLP head to produce a fixed-length representation. A **gated fusion** module learns modality weights and feeds a final binary classifier (deceptive vs. truthful).

---

## Results
Average over 5 folds:

| Modality | Accuracy | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|
| Visual | 0.46 | 0.48 | 0.60 | 0.52 |
| Audio  | 0.54 | 0.55 | 0.78 | 0.63 |
| **Fusion** | **0.69** | **0.74** | **0.76** | **0.72** |

Comparison with a published ATSFace baseline:

| Method | Accuracy | F1 |
|:--|--:|--:|
| Proposed (best fold) | 0.76 | 0.68 |
| Baseline (literature) | 0.79 | 0.79 |

Multimodal fusion markedly outperforms single-modality models.

---
