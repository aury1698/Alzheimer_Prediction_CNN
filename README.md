# MRI Image Classification for Alzheimer’s Diagnosis: Development and Evaluation of a Custom CNN Model

This repository contains a PyTorch implementation and empirical evaluation of a custom Convolutional Neural Network (CNN) designed to classify brain MRI scans into four distinct stages of Alzheimer's Disease.

The study explores **8 distinct experimental configurations** to evaluate the trade-offs between subsampling methods, regularization techniques, and data augmentation in medical imaging.

---

## 📌 Key Highlights & Results

* **Top Performance (Ideal Conditions):** **98.28% Test Accuracy** achieved *without* Data Augmentation, using **Stride (2,2)** and **Dropout**.


* **Top Performance (Robustness Scenario):** **97.50% Test Accuracy** achieved *with* Data Augmentation, using **Stride (2,2)** and **No Dropout**.


* **Architectural Insight:** Strided convolutions consistently outperformed MaxPooling across all configurations, confirming their superior ability to preserve spatial context and fine-grained details in MRI scans.


* **Class Imbalance Strategy:** Handled directly via **Class-Weighted Cross-Entropy Loss** and `WeightedRandomSampler`, avoiding synthetic oversampling.



---

## 📊 Dataset & Class Imbalance

The experiments were conducted using the **Alzheimer MRI Preprocessed Dataset** (Kaggle), consisting of 6,400 MRI images resized to $128 \times 128$ pixels.

| Class | Sample Count | Description / Severity |
| --- | --- | --- |
| **Non Demented** | 3,200 | Healthy control group |
| **Very Mild Demented** | 2,240 | Early-stage impairment |
| **Mild Demented** | 896 | Noticeable cognitive decline  |
| **Moderate Demented** | 64 | Minority class (Severe impairment) |

**Data Split:** 80% Training ($5{,}120$ images), 10% Validation ($640$ images), 10% Testing ($640$ images).

---

## 🏗️ Model Architecture

The custom CNN architecture comprises **7 convolutional blocks**, designed with the following specifications:

1. **Block 1:** $32 \text{ filters } (3\times 3)$, Padding 1, Stride 1 $\rightarrow$ BatchNorm $\rightarrow$ ReLU.


2. **Block 2:** $128 \text{ filters } (3\times 3)$, Padding 1, Stride 1 $\rightarrow$ BatchNorm $\rightarrow$ ReLU.


3. **Block 3:** $128 \text{ filters } (3\times 3)$, Padding 1, Stride 1 (or Stride 2) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ *Optional 2x2 MaxPooling*.


4. **Block 4 ($1\times 1$ Bottleneck):** $64 \text{ filters } (1\times 1)$, Padding 0, Stride 1 $\rightarrow$ BatchNorm $\rightarrow$ ReLU (Channel dimensionality reduction).


5. **Blocks 5–7:** $256 \text{ filters } (3\times 3)$, Padding 1, Stride 1 (or Stride 2) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ *Optional 2x2 MaxPooling*.


6. **Dense Classifier:** Flatten $\rightarrow$ FC Layer ($128$ units) $\rightarrow$ *Optional Dropout ($0.5$)* $\rightarrow$ ReLU $\rightarrow$ FC Output ($4$ units) $\rightarrow$ LogSoftmax.



### ⚙️ Training Setup

* **Optimizer:** Adam ($\text{learning rate} = 10^{-4}$).


* **Loss Function:** Class-Weighted CrossEntropyLoss.


* **Scheduler:** `ReduceLROnPlateau` for dynamic learning rate adjustments.


* **Regularization & Early Stopping:** Early stopping with a patience of 5 epochs monitoring validation accuracy.


* **Reproducibility:** Fixed random seed set to `42`.



---

## 🔬 Experimental Results

All models were trained for a maximum of 20 epochs across 8 distinct setups.

### 1. Model Test Accuracy Summary

| Subsampling Method | Regularization | With Data Transformation | Without Data Transformation |
| --- | --- | --- | --- |
| **Stride (2,2)** | **No Dropout** | **97.50%**<br> | 98.13% |
| **Stride (2,2)** | **Dropout (0.5)** | 97.03%| **98.28%** ⭐ |
| **MaxPooling** | **No Dropout** | 96.09%| 96.25% |
| **MaxPooling** | **Dropout (0.5)** | 96.56%| 95.63% |

### 2. Key Findings & Discussion

* **Stride vs. MaxPooling:** Stride (2,2) consistently outperformed MaxPooling across all conditions. Strided convolutions preserve finer feature maps critical for detecting subtle anatomical anomalies in MRI scans (such as ventricular enlargement and hippocampal atrophy).


* **Data Augmentation Trade-offs:** Data transformations (rotations, cropping, Gaussian blur, brightness/contrast adjustments) introduce minor distortions that lower test accuracy under ideal conditions, but significantly boost model robustness for real-world noisy scans.


* **Dropout & BatchNorm Interaction:** Since Batch Normalization is applied in all convolutional blocks, the inclusion of Dropout in FC layers offered only marginal accuracy improvements without transformations, and slightly reduced accuracy when transformations were active.



---

## 📂 Repository Structure

```text
├── notebooks/
│   ├── alzheimer_with_data_transformation.ipynb     # Pipeline with Augmentation
│   └── alzheimer_without_data_transformation.ipynb  # Pipeline with Original Images
├── docs/
│   └── Alzheimer_CNN_Scientific_Report.pdf          # Complete IEEE-style Report
└──  README.md

```

---

## 🚀 Future Developments

* **Advanced Resampling:** Implementation of **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance without relying solely on class-weighted loss functions.


* **Cross-Dataset Validation:** Testing generalizability on external neuroimaging databases such as **ADNI** (Alzheimer's Disease Neuroimaging Initiative).


* **Domain-Specific Transfer Learning:** Fine-tuning pre-trained medical imaging weights rather than standard ImageNet backbones.
