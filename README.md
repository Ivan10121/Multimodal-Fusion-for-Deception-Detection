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

## Repository Structure
```
.
├─ data/
│  ├─ visual/          # FaceNet .npy sequences: [T_v, 128]
│  ├─ audio/           # MFCC .npy sequences:   [T_a, 20]
│  ├─ text/            # Averaged BERT vectors: [128]
│  └─ splits/          # Fold indices / hold-out JSON files
├─ src/
│  ├─ models/
│  │  ├─ transformer_visual.py
│  │  ├─ transformer_audio.py
│  │  └─ fusion_gated.py
│  ├─ dataio/
│  │  ├─ datasets.py
│  │  └─ collate.py
│  ├─ train_visual.py
│  ├─ train_audio.py
│  ├─ extract_text_repr.py
│  ├─ train_fusion.py
│  ├─ eval_modality.py
│  └─ utils.py
├─ configs/
│  ├─ visual.yaml
│  ├─ audio.yaml
│  └─ fusion.yaml
├─ notebooks/
│  └─ exploration.ipynb
├─ requirements.txt
└─ README.md
```

---

## Requirements
- Python **3.10+**
- PyTorch **2.x** (CUDA optional)
- NumPy, SciPy, scikit-learn, tqdm, PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Installation
```bash
git clone https://github.com/Ivan10121/Multimodal-Fusion.git
cd Multimodal-Fusion
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset
**ATSFace** provides **precomputed features**, not raw videos.

- **Clips:** 309 (147 deceptive / 162 truthful)  
- **Avg duration:** ~23 s (≈10–50 s range)  
- **Recording:** iPhone 14 Pro, 1080p@30fps, Chinese prompts

Expected layout:
```
data/
  visual/clip123.npy    # [T_v, 128]
  audio/clip123.npy     # [T_a, 20]
  text/clip123.npy      # [128] (mean of sentence embeddings)
  splits/test_ids.json
  splits/folds.json
```
> Place `.npy` files under `data/` or update paths in `configs/*.yaml`.

---

## Training Setup
- **Hold-out test:** 15% (never used during training/fusion)
- **Cross-validation:** 5-fold stratified on the remaining 85%
- **Optimizer:** AdamW (+ annealing scheduler)
- **Early stopping:** validation accuracy
- **Per-fold features:** best model per fold extracts features for that fold’s validation set (no leakage).

---

## Per-Modality Models

### Visual
- **Input:** sequence of 128-D FaceNet frames  
- **Backbone:** Transformer encoder (1 layer, 4 heads, FF=256, dropout 0.5)  
- **Tokens:** learnable `[CLS]` + 128-D positional embeddings  
- **Head:** `128 → 64 → 2` (ReLU, dropout 0.5)  
Train:
```bash
python src/train_visual.py --config configs/visual.yaml
```

### Audio
- **Input:** sequence of 20-D MFCC frames (0.2 s)  
- **Backbone:** Transformer encoder (1 layer, 4 heads, FF=25, dropout 0.5)  
- **Tokens:** learnable `[CLS]` + 20-D positional embeddings  
- **Head:** `20 → 64 → 2` (ReLU, dropout 0.5)  
Train:
```bash
python src/train_audio.py --config configs/audio.yaml
```

### Text
- **Input:** sentence-level BERT embeddings (128-D)  
- **Clip representation:** mean pool across sentences → single 128-D vector  
Extract:
```bash
python src/extract_text_repr.py
```

---

## Multimodal Fusion
Each modality is projected to **128-D** via a small MLP:
- `Linear → LayerNorm → ReLU → Dropout`

The three 128-D vectors (`t`, `v`, `a`) are combined via **softmax-gated fusion**:
- Gating MLP: `concat(t, v, a) ∈ R^{384} → 128 → 3 → softmax` → weights `(w_t, w_v, w_a)`
- **Fused vector:** `w_t·t + w_v·v + w_a·a`
- **Final classifier:** `128 → 64 → 1` (BCEWithLogitsLoss)

Train fusion:
```bash
python src/train_fusion.py --config configs/fusion.yaml
```

---

## Evaluation
Compute metrics (accuracy, precision, recall, F1) and confusion matrices:
```bash
# Visual
python src/eval_modality.py --modality visual --checkpoint runs/visual/best.pt
# Audio
python src/eval_modality.py --modality audio  --checkpoint runs/audio/best.pt
# Fusion
python src/eval_modality.py --modality fusion --checkpoint runs/fusion/best.pt
```

---

## Reproducibility
- Fixed **15%** independent test split  
- **5-fold** stratified CV for model selection  
- **Early stopping** and consistent fold indices across all modalities and fusion

---

## Configuration
Example (`configs/fusion.yaml`):
```yaml
seed: 42
batch_size: 64
epochs: 100
optimizer:
  name: adamw
  lr: 1.5e-4
  weight_decay: 0.01
scheduler:
  name: cosine
  warmup_epochs: 5
fusion:
  proj_dim: 128
  dropout: 0.5
  gate_hidden: 128
  final_hidden: 64
paths:
  data_root: data/
  runs_root: runs/
```

---

## Roadmap
- Add attention visualizations and per-modality saliency
- Explore BERT fine-tuning and larger text encoders
- Temporal alignment and cross-modal attention
- Robustness checks (noise, missing modality)

---

## Ethical Considerations
Deception detection is **high-stakes**. Use models responsibly:
- Always include **human oversight** and due-process safeguards  
- Respect privacy, consent, and data governance rules  
- Avoid deployment for consequential decisions without rigorous validation

---

## Citation
```bibtex
@misc{Galvan2025MultimodalFusion,
  author       = {Iván Galván Gómez},
  title        = {Multimodal Fusion for Deception Detection},
  year         = {2025},
  howpublished = {https://github.com/Ivan10121/Multimodal-Fusion}
}
```

---

## Author
**Iván Galván Gómez**  
Universidad Panamericana – Aguascalientes, Mexico  
✉️ 0246325@up.edu.mx

---

## License
Released under the **MIT License**. See `LICENSE` for details.

