# Food Safety Review Classifier (Chinese) — MacBERT + 5-Fold CV + OOF Threshold Tuning

This repository contains a GPU-friendly text classification pipeline that detects **food safety / hygiene risk complaints** in Chinese reviews.  
It is designed for **F1-based competitions**, using **5-fold Stratified Cross Validation**, **out-of-fold (OOF) probability aggregation**, and **threshold tuning** to maximize the positive-class F1 score.

## Key Features
- **Backbone:** `hfl/chinese-macbert-base` (base-size Chinese MacBERT)
- **5-fold Stratified CV** for robust validation
- **OOF threshold selection** to directly optimize **F1**
- **Class-weighted loss** to address class imbalance
- **Colab-friendly** settings (FP16, gradient accumulation, early stopping)
- **Diagnostics plots**:
  - F1 vs threshold
  - ROC (AUC)
  - Precision-Recall (AP)
  - Confusion matrix
  - Calibration curve
  - Training/eval loss and eval F1 curves
  - Per-fold F1@0.5

---


