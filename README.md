# Multimodal Fusion for Deception Detection

> A modular multimodal pipeline for deception (lie) detection using **ATSFace** features. Visual, audio, and textual streams are modeled independently and then combined via a **softmax-gated fusion** mechanism for the final classification.



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

## Requirements
- Python **3.10+**
- PyTorch **2.x** (CUDA optional)
- NumPy, scikit-learn



---

## Dataset
**ATSFace** provides **precomputed features**, not raw videos.

- **Clips:** 309 (147 deceptive / 162 truthful)  
- **Avg duration:** ~23 s (≈10–50 s range)  
- **Recording:** iPhone 14 Pro, 1080p@30fps, Chinese prompts

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


### Audio
- **Input:** sequence of 20-D MFCC frames (0.2 s)  
- **Backbone:** Transformer encoder (1 layer, 4 heads, FF=25, dropout 0.5)  
- **Tokens:** learnable `[CLS]` + 20-D positional embeddings  
- **Head:** `20 → 64 → 2` (ReLU, dropout 0.5)  


### Text
- **Input:** sentence-level BERT embeddings (128-D)  
- **Clip representation:** mean pool across sentences → single 128-D vector  


---

## Multimodal Fusion
Each modality is projected to **128-D** via a small MLP:
- `Linear → LayerNorm → ReLU → Dropout`

The three 128-D vectors (`t`, `v`, `a`) are combined via **softmax-gated fusion**:
- Gating MLP: `concat(t, v, a) ∈ R^{384} → 128 → 3 → softmax` → weights `(w_t, w_v, w_a)`
- **Fused vector:** `w_t·t + w_v·v + w_a·a`
- **Final classifier:** `128 → 64 → 1` (BCEWithLogitsLoss)

---

## Author
**Iván Galván Gómez**  
Universidad Panamericana – Aguascalientes, Mexico  
✉️ 0246325@up.edu.mx


