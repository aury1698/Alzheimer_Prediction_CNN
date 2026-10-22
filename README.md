# MRI Image Classification for Alzheimer's Diagnosis: Custom CNN Model

A PyTorch-based deep learning framework featuring a custom Convolutional Neural Network (CNN) architecture designed for the multi-class classification of brain MRI scans into four distinct stages of Alzheimer's Disease (AD).

Developed as part of the **Deep Learning** course for the B.S. in Computer and Robotics Engineering at the **University of Perugia (UniPG)** – Department of Engineering.

---

## Project Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder and the leading cause of dementia globally. Early and accurate diagnosis via Magnetic Resonance Imaging (MRI) is essential for identifying key anatomical biomarkers such as hippocampal atrophy and ventricular enlargement.

This project introduces a **lightweight, custom 7-block CNN architecture** implemented in **PyTorch**. To evaluate architectural trade-offs, model robustness, and generalization, the study explores **8 distinct experimental configurations** comparing:
- **Subsampling Methods**: Strided Convolutions ($	ext{Stride}=2$) vs. **MaxPooling** ($2	imes 2$).
- **Regularization Techniques**: Dense Layer **Dropout** ($0.5$) vs. **No Dropout**.
- **Data Augmentation**: Training on **Transformed/Augmented Data** vs. **Original/Non-Transformed Data**.
- **Class Imbalance Mitigation**: **Class-Weighted Cross-Entropy Loss** to compensate for minority classes without artificial oversampling.

---

## Dataset & Class Imbalance

The experiments were conducted using the **Alzheimer MRI Preprocessed Dataset** (Kaggle), comprising **6,400 preprocessed brain MRI scans** ($128 	imes 128$ pixels).

### Class Distribution
| Class Label | Sample Count | Percentage | Clinical Stage |
| :--- | :---: | :---: | :--- |
| **Non Demented** | 3,200 | 50.0% | Healthy control group |
| **Very Mild Demented** | 2,240 | 35.0% | Early-stage cognitive impairment |
| **Mild Demented** | 896 | 14.0% | Noticeable cognitive decline |
| **Moderate Demented** | 64 | 1.0% | Severe dementia (Extreme minority class) |

### Dataset Partitioning & Preprocessing
- **Data Split**: **80% Training** ($5{,}120$ images), **10% Validation** ($640$ images), and **10% Testing** ($640$ images).
- **Normalization**: Pixel intensities scaled to $[0, 1]$.
- **Class Weighting**: To prevent model bias toward majority classes (`Non_Demented`), inverse class frequencies were computed and embedded directly into the loss function:
  $$	ext{Loss} = -\sum_{c=1}^{C} w_c \cdot y_c \log(\hat{y}_c)$$
- **Data Augmentation Pipeline**: Applied in `Alzheimer_Prediction_DataAug.ipynb` including Random Rotation (up to $30^\circ$), Random Resized Crop (scale $0.8$–$1.0$), Gaussian Blur (kernel $3	imes 3$), and Random Brightness/Contrast/Hue adjustments.

---

## Custom CNN Architecture

The model is engineered with **7 Convolutional Blocks** designed for feature extraction, including a $1	imes 1$ bottleneck block for channel dimensionality reduction and an optional FC Dropout layer.

```text
[Input 128x128x3]
       │
   [Conv Block 1] ──> 32 filters (3x3), Stride 1, Pad 1 ──> BatchNorm ──> ReLU
       │
   [Conv Block 2] ──> 128 filters (3x3), Stride 1, Pad 1 ──> BatchNorm ──> ReLU
       │
   [Conv Block 3] ──> 128 filters (3x3), Pad 1 (Stride 2 OR Stride 1 + 2x2 MaxPool) ──> BatchNorm ──> ReLU
       │
   [Conv Block 4] ──> 64 filters (1x1 Bottleneck), Stride 1, Pad 0 ──> BatchNorm ──> ReLU
       │
   [Conv Block 5] ──> 256 filters (3x3), Pad 1 (Stride 2 OR Stride 1 + 2x2 MaxPool) ──> BatchNorm ──> ReLU
       │
   [Conv Block 6] ──> 256 filters (3x3), Pad 1 (Stride 2 OR Stride 1 + 2x2 MaxPool) ──> BatchNorm ──> ReLU
       │
   [Conv Block 7] ──> 256 filters (3x3), Pad 1 (Stride 2 OR Stride 1 + 2x2 MaxPool) ──> BatchNorm ──> ReLU
       │
   [Flatten] ──> Linear (128 units) ──> [Optional Dropout (0.5)] ──> ReLU
       │
   [Output FC] ──> Linear (4 units) ──> LogSoftmax
```

### Key Architectural Highlights
1. **$1	imes 1$ Bottleneck Layer (Block 4)**: Reduces feature channel dimensions from 128 to 64 before expanding to 256 channels, significantly lowering total parameter count and computational cost.
2. **Batch Normalization**: Applied after every convolutional layer to stabilize learning and accelerate convergence.
3. **Flexible Subsampling**: Modular architecture allowing toggleable selection between Strided Convolutions and MaxPooling.

---

## Training Setup

- **Optimizer**: Adam ($	ext{learning rate} = 10^{-4}$).
- **Loss Function**: Class-Weighted `CrossEntropyLoss`.
- **Learning Rate Scheduler**: `ReduceLROnPlateau` (monitors validation loss, reduces lr on plateau).
- **Early Stopping**: Halts training if validation accuracy does not improve for **5 consecutive epochs** (patience = 5, max 20 epochs).
- **Reproducibility**: Global seed set to `42`.

---

## Empirical Results & Comparative Analysis

All 8 configurations were trained and systematically evaluated on the unseen $10\%$ test set ($640$ images).

### Test Classification Accuracy Summary

| Subsampling Method | Regularization | With Data Augmentation | Without Data Augmentation |
| :--- | :--- | :---: | :---: |
| **Stride (2,2)** | **No Dropout** | **97.50%** | 98.13% |
| **Stride (2,2)** | **Dropout (0.5)** | 97.03% | **98.28%** ⭐ *(Best Overall)* |
| **MaxPooling** | **No Dropout** | 96.09% | 96.25% |
| **MaxPooling** | **Dropout (0.5)** | 96.56% | 95.63% |

### Key Findings & Discussion
1. **Strided Convolutions vs. MaxPooling**:
   - Across all setups, **Stride (2,2)** consistently outperformed MaxPooling (achieving up to **+2.65% higher accuracy**).
   - *Medical Imaging Insight*: Strided convolutions preserve finer spatial context and spatial relationships, whereas MaxPooling discards subtle intensity gradients crucial for detecting early-stage neurodegeneration.
2. **Impact of Data Augmentation**:
   - The highest overall accuracy (**98.28%**) was obtained on non-transformed data under ideal scan conditions.
   - However, data augmentation achieved a high **97.50% accuracy**, demonstrating that synthetic perturbations (rotation, noise, blur) build a more resilient model suited for real-world clinical noise.
3. **Dropout vs. Batch Normalization**:
   - Because Batch Normalization acts as an inherent regularizer across all 7 conv blocks, adding Dropout to the dense layer provided only minor gains without transformations and slightly reduced performance when data augmentation was active.

---

## Repository Structure

```text
.
├── Alzheimer_Prediction_NODataAug.ipynb  # Pipeline training on original MRI images
├── Alzheimer_Prediction_DataAug.ipynb    # Pipeline training with data transformations
├── Deep_Learning_Report_Aurora_Traversini_361600.pdf # Scientific paper (IEEE format)
└── README.md                             # Project documentation
```

### File Descriptions
- `Alzheimer_Prediction_NODataAug.ipynb`: Complete PyTorch pipeline for non-augmented MRI dataset, including data loading, class-weighted loss definition, model variants (Stride vs MaxPool, Dropout vs No Dropout), training loops with early stopping, loss/accuracy curves, and confusion matrix evaluation.
- `Alzheimer_Prediction_DataAug.ipynb`: Complete PyTorch pipeline incorporating image transformations (rotation, crop, blur, color jitter) to test model generalization and noise robustness.
- `Deep_Learning_Report_Aurora_Traversini_361600.pdf`: Full scientific paper detailing the theoretical background, literature review, architectural schematics, confusion matrices, and empirical benchmarks.

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas numpy
```

### Execution
1. Download the **Alzheimer MRI Preprocessed Dataset** from Kaggle.
2. Place the dataset directory in the project root.
3. Launch Jupyter Notebook or Google Colab:
   - Run `Alzheimer_Prediction_NODataAug.ipynb` for baseline experiments on original scans.
   - Run `Alzheimer_Prediction_DataAug.ipynb` for augmented experiments.

---

## Future Work

- **Synthetic Resampling (SMOTE)**: Combining class-weighted losses with oversampling techniques for severe minority classes (`Moderate_Demented`).
- **External Dataset Validation**: Testing cross-dataset generalizability on the **ADNI** (Alzheimer's Disease Neuroimaging Initiative) database.
- **Domain Transfer Learning**: Fine-tuning pre-trained medical imaging backbones (e.g., Med3D or MONAI) versus standard ImageNet models.

---

## Author

* **Aurora Traversini** (Student ID: 361600)  
  *B.S. in Computer and Robotics Engineering*  
  **Department of Engineering – University of Perugia (UniPG)**  
  Email: `aurora.traversini@studenti.unipg.it`
