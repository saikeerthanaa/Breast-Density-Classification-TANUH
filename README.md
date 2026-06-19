# Mammography Breast Density Classification: Experimental Consolidation

## 1. Project Overview & Motive
This project addresses the challenge of **Breast Density Classification** (BI-RADS Categories A, B, C, D) and specifically investigates the **Domain Shift** encountered when models trained on US populations (**EMBED**) are deployed on Indian populations (**IBIA**).

### Motive
Breast density is a critical risk factor for breast cancer and can mask tumors on mammograms. While high-performance models exist, generalization across different ethnic and anatomical populations is often poor. The objectives include:
1. Establishment of a robust baseline using traditional machine learning (LightGBM) and deep ordinal regression.
2. Quantification of the domain shift between the EMBED (Source) and IBIA (Target) datasets.
3. Development and evaluation of mitigation strategies including Zero-shot inference, Head recalibration, and Fine-tuning.
4. Investigation of modern backbone architectures (ConvNeXt) and their cross-population transferability under ordinal regression frameworks.

---

## 2. Experimental Setup

| Component | Details |
| :--- | :--- |
| **GPU** | NVIDIA GB10 |
| **CUDA Version** | 13.0 (Driver: 580.95.05) |
| **CPU** | Cortex-X925 (10 cores) + Cortex-A725 (10 cores) |
| **RAM** | 119 GiB total, ~109 GiB available |
| **Python** | 3.12.3 |
| **PyTorch** | 2.12.0+cu130 |
| **Framework** | PyTorch + torchvision |
| **Mixed Precision** | torch.cuda.amp (GradScaler + autocast) |

---

## 3. Dataset Description

### Sample Mammograms

#### EMBED Dataset (Source — US Population)

| Fatty (A) | Scattered (B) | Heterogeneous (C) | Dense (D) |
| :---: | :---: | :---: | :---: |
| <img src="sample_images/embed_A.png" width="180"> | <img src="sample_images/embed_B.png" width="180"> | <img src="sample_images/embed_C.png" width="180"> | <img src="sample_images/embed_D.png" width="180"> |
| *BI-RADS A: Almost entirely fatty* | *BI-RADS B: Scattered fibroglandular* | *BI-RADS C: Heterogeneously dense* | *BI-RADS D: Extremely dense* |

#### IBIA Dataset (Target — Indian Population)

| Fatty (A) | Scattered (B) | Heterogeneous (C) | Dense (D) |
| :---: | :---: | :---: | :---: |
| <img src="sample_images/ibia_A.png" width="180"> | <img src="sample_images/ibia_B.png" width="180"> | <img src="sample_images/ibia_C.png" width="180"> | <img src="sample_images/ibia_D.png" width="180"> |
| *BI-RADS A: Almost entirely fatty* | *BI-RADS B: Scattered fibroglandular* | *BI-RADS C: Heterogeneously dense* | *BI-RADS D: Extremely dense* |

### Dataset Statistics

### EMBED (Source — US Population)
- **Scale:** 37,563 mammograms from 9,398 patients.
- **Stratification:** Patient-stratified split (80% Train, 10% Validation, 10% Test) to prevent data leakage.
- **Labels:** ACR BI-RADS 5th Edition breast density categories (A–D) [1].
- **Modality:** Full-Field Digital Mammography (FFDM).
- **Cohort:** Diverse US population.
- **Key Characteristics:**
    - ~49% of studies represent dense breasts (Categories C and D).
    - 99.8% of patients possess complete 4-view mammograms (L/R CC and MLO).

### IBIA (Target — Indian Population)
- **Scale:** 3,569 images from 583 patients.
- **Labels:** ACR BI-RADS 5th Edition (A–D) [1].
- **Modality:** Full-Field Digital Mammography (FFDM).
- **Cohort:** Indian clinical population.
- **Demographics:** Patient age range [23, 87], average age 50.0.
- **Key Characteristics:**
    - Significant anatomical distribution shift: ~82.1% are Non-Dense (A+B).
    - Average of ~6.1 images per patient.
    - Utilized for Zero-shot evaluation and small-split adaptation experiments.

---

## 4. Experimental Framework & Methodology

### Phase 1: Feature Extraction Baseline (LightGBM)
A baseline was established using fixed feature extractors (ResNet50 [2] and EfficientNet-B7 [3] pretrained on ImageNet [4]).
- **Methodology:** High-dimensional feature vectors (2048-d for ResNet, 2560-d for EffNet) were extracted from the global average pooling layer.
- **Classifier:** LightGBM gradient boosting machine, tuned for 4-class classification.
- **Limitation:** Treats density categories as independent classes, ignoring the ordinal relationship (A < B < C < D).

### Phase 2: Deep Ordinal Regression (End-to-End with ResNet50)
Three deep ordinal regression architectures were implemented using a ResNet50 backbone [2], partially unfrozen for domain-specific refinement.

1. **OR-NN (Ordinal Neural Network):** $K-1$ binary classifiers with independent weights predicting whether density exceeds each threshold [5].
2. **CORAL (Consistent Rank Logits):** Shared weights with unique, rank-consistent bias terms ensuring nested probability structure [6].
3. **CORN (Conditional Ordinal Regression):** Conditional probability framework where reaching rank $k$ is conditioned on passing all previous ranks [7].

### Phase 3: Advanced Backbone Ordinal Regression (ConvNeXt + CORAL)
The ResNet50 backbone was replaced with **ConvNeXt-Tiny** [8].
- **Architecture:** Incorporates inverted bottlenecks, larger kernel sizes ($7\times7$), and Layer Normalization — design principles inspired by Vision Transformers [9].
- **Objective:** Improve fine-grained parenchymal feature extraction for the CORAL ordinal head.

### Phase 4: Cross-Population Adaptation (ConvNeXt + CORAL on IBIA)
The EMBED-trained ConvNeXt + CORAL model was evaluated and adapted on IBIA using the same adaptation strategies as ResNet + CORAL, to assess whether architectural improvements translate to better cross-population transferability.

---

## 5. Key Results & Technical Inferences

### Inference 1: Systematic Over-prediction
Zero-shot evaluation on IBIA revealed systematic over-prediction — "A" frequently predicted as "B", "B" as "C" — attributed to the dense-skewed EMBED prior. This occurs in both backbones, confirming it is dataset-level rather than architecture-specific.

### Inference 2: Feature Space Domain Shift (t-SNE)
t-SNE projections of the ResNet50 feature space show EMBED and IBIA forming distinct, non-overlapping clusters.

<p align="center">
  <img src= "Ordinal_Regression/RESULTS/tsne_domain_shift.png"
   width="500">
  <br>
  <i>t-SNE visualization: Feature shift across geographic cohorts.</i>
</p>

**Technical Conclusion:** The domain shift is a **fundamental feature shift** — anatomical features are encoded differently for the two populations, not merely a label distribution mismatch.

### Inference 3: Failure of Optical Mitigation (Histogram Equalization)
Histogram equalization degraded performance (Kappa 0.48 → 0.33).

**Technical Conclusion:** The domain shift is **semantic/anatomical**, not optical. Contrast enhancement amplified the features that trigger high-density predictions in the EMBED-trained model.

### Inference 4: Efficacy of Head-Only Recalibration
Freezing the backbone and retraining only the CORAL head on 20% IBIA data yielded Kappa gains of **+0.09** (ResNet) and **+0.17** (ConvNeXt).

**Technical Conclusion:** A large fraction of domain shift is attributable to miscalibrated decision thresholds rather than inadequate features, especially for ConvNeXt whose features are inherently more expressive.

### Inference 5: Representation Capacity of ConvNeXt-Tiny
ConvNeXt-Tiny achieves the highest EMBED Kappa (0.8020) using only the simplest ordinal loss (CORAL), outperforming ResNet with more complex losses.

**Technical Conclusion:** ConvNeXt's transformer-inspired design excels at capturing fine-grained diffuse parenchymal patterns, providing more discriminative density boundaries [8].

### Inference 6: ConvNeXt Adaptation Superiority
Despite lower zero-shot Kappa (0.4514 vs 0.4844), ConvNeXt + CORAL outperforms ResNet + CORAL under both adaptation strategies.

**Technical Conclusion:** ConvNeXt encodes richer, more transferable representations that are initially over-biased toward EMBED priors but adapt rapidly with minimal target-domain data. **Backbone capacity is the dominant factor for adapted cross-population performance.**

---

## 6. Performance on EMBED (Training Domain)

| Method | Accuracy | Quadratic Kappa | MAE |
| :--- | :---: | :---: | :---: |
| **ResNet50 + LGBM (Baseline)** [2] | 0.6941 | ~0.72 | 0.35 |
| **ResNet50 + CORAL** [6] | 0.7335 | 0.7833 | 0.2700 |
| **ResNet50 + CORN** [7] | **0.7740** | 0.7884 | **0.2279** |
| **ResNet50 + OR-NN** [5] | 0.7508 | 0.8008 | 0.2508 |
| **ConvNeXt + CORAL (Proposed)** [6, 8] | 0.7663 | **0.8020** | 0.2361 |

*ConvNeXt + CORAL achieves the highest Quadratic Kappa (0.8020) using the simplest ordinal loss, showing that backbone capacity drives meaningful gains independent of loss complexity.*

---

## 7. Performance Matrix (Evaluation on IBIA)

### ResNet50 + CORAL (EMBED → IBIA)

| Strategy | Quadratic Kappa | Technical Result Summary |
| :--- | :---: | :--- |
| **Zero-shot** | 0.4844 | High bias due to learned EMBED priors. |
| **Histogram Equalization** | 0.3296 | Ineffective; amplified distribution shift. |
| **Head-Only Recalibration** | 0.5746 | Efficient adaptation via threshold shifting. |
| **Differential Fine-tuning** | 0.5971 | Optimal (Backbone 1e-5, Head 1e-3). |

### ConvNeXt + CORAL (EMBED → IBIA)

| Strategy | Quadratic Kappa | Accuracy | MAE | vs. ResNet |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-shot** | 0.4514 | 0.4217 | 0.6301 | -0.033 |
| **Head-Only Recalibration** | 0.6258 | 0.6579 | 0.3553 | **+0.051** |
| **Differential Fine-tuning** | **0.6535** | **0.6723** | **0.3417** | **+0.056** |

*ConvNeXt + CORAL differential fine-tuning achieves the best overall cross-population result, surpassing ResNet's best by +0.056 Kappa.*

---

## 8. Confusion Matrices
### ConvNeXt + CORAL — All Evaluation Conditions

<p align="center">
  <img src="RESULTS/confusion_matrices/convnext_coral_all_cms.png" width="850">
  <br>
  <i>Confusion matrices: ConvNeXt + CORAL across EMBED and three IBIA evaluation conditions.</i>
</p>

### Individual Matrices

<p align="center">
  <img src="RESULTS/confusion_matrices/embed_convnext_coral.png" width="450">
  <br>
  <i>ConvNeXt + CORAL on EMBED test set (Kappa: 0.8020).</i>
</p>

<p align="center">
  <img src="RESULTS/confusion_matrices/ibia_convnext_zeroshot.png" width="450">
  <br>
  <i>ConvNeXt + CORAL — IBIA Zero-Shot (Kappa: 0.4514).</i>
</p>

<p align="center">
  <img src="RESULTS/confusion_matrices/ibia_convnext_recalibration.png" width="450">
  <br>
  <i>ConvNeXt + CORAL — IBIA Head Recalibration (Kappa: 0.6258).</i>
</p>

<p align="center">
  <img src="RESULTS/confusion_matrices/ibia_convnext_differential_ft.png" width="450">
  <br>
  <i>ConvNeXt + CORAL — IBIA Differential Fine-Tuning (Kappa: 0.6535).</i>
</p>

### ResNet + CORAL — Best Adapted (IBIA Head Recalibration)

<p align="center">
  <img src="Ordinal_Regression/RESULTS/confusion_matrices/ibia_recal_coral.png" width="450">
  <br>
  <i>Recalibrated ResNet + CORAL on IBIA (Kappa: 0.5746).</i>
</p>

---

## 9. ConvNeXt + CORAL Per-Class Analysis (EMBED)

| BI-RADS Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Fatty (A)** | 0.59 | 0.75 | 0.66 | 388 |
| **Scattered (B)** | 0.77 | 0.75 | 0.76 | 1525 |
| **Heterogeneous (C)** | 0.85 | 0.79 | 0.82 | 1623 |
| **Dense (D)** | 0.62 | 0.77 | 0.69 | 216 |

Key observations:
- Only **9/3,752 cases (0.24%)** misclassified by more than one ordinal rank — confirming CORAL's structural constraint.
- Recall is balanced across all classes (0.75–0.79), showing no class-imbalance bias.
- Lower precision on extreme categories (Fatty: 0.59, Dense: 0.62) reflects inherent boundary ambiguity between adjacent BI-RADS categories [1].

---

## 10. Summary of Findings

| Finding | Result |
| :--- | :--- |
| Best in-distribution (EMBED) | ConvNeXt + CORAL, Kappa **0.8020** |
| Best zero-shot cross-population (IBIA) | ResNet + CORAL, Kappa **0.4844** |
| Best adapted cross-population (IBIA) | ConvNeXt + CORAL + Diff. FT, Kappa **0.6535** |
| Most efficient adaptation | ConvNeXt + CORAL head recalibration (+0.174 Kappa, <1% params updated) |

---

## 11. Folder Structure

- `baseline_LGBM/` — Baseline LightGBM experiments with fixed feature extraction.
- `ConvNeXt_CORAL/` — Training and evaluation scripts for ConvNeXt + CORAL.
  - `train_convnext.py` — Training script.
  - `test_convnext.py` — Evaluation script.
  - `results.txt` — Metrics and confusion matrix.
- `Ordinal_Regression/`
  - `BEST_MODELS/` — Serialized weights for CORAL, CORN, OR-NN, and ConvNeXt + CORAL.
  - `EXPERIMENTS/` — Training logs and hyperparameter details.
  - `RESULTS/` — Metrics, confusion matrices, and t-SNE visualizations.
- `sample_images/` — Representative PNG samples from EMBED and IBIA cohorts.
- `mammography-datasets/analysis/` — All training, evaluation, and adaptation scripts.

---


## 12. References

[1] Kumari and Singh (2024). *Deep Learning for Unsupervised Domain Adaptation in Medical Imaging: Recent Advancements and Future Perspectives.*

[2] Squires et al. (2024). *Model Uncertainty Estimates for Deep Learning Mammographic Density Prediction Using Ordinal and Classification Approaches.*

[3] Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks.*

[4] Sun and Saenko (2016). *Deep CORAL: Correlation Alignment for Deep Domain Adaptation.*

[5] Schmidt et al. (2024). *Fair Evaluation of Federated Learning Algorithms for Automated Breast Density Classification: The Results of the 2022 ACR-NCI-NVIDIA Federated Learning Challenge.*

[6] Niu et al. (2016). *Ordinal Regression with Multiple Output CNN for Age Estimation.*

[7] Yoon et al. (2023). *Domain Generalization for Medical Image Analysis: A Review.*

[8] Molina-Roman et al. (2025). *Comparison of ConvNeXt and Vision-Language Models for Breast Density Assessment in Screening Mammography.*

[9] Liu et al. (2022). *A ConvNet for the 2020s.*

[10] Lin et al. (2017). *Focal Loss for Dense Object Detection.*

[11] Zha et al. (2023). *Rank-N-Contrast: Learning Continuous Representations for Regression.*
