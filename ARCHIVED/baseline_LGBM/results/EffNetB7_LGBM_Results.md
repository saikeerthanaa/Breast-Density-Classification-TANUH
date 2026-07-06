# EfficientNet-B7 + LightGBM Classification Results

## Experiment Overview
- **Model:** LightGBM Classifier
- **Features:** 2560-dimensional EfficientNet-B7 features (ImageNet weights)
- **Target:** Breast Density Class (1-4)
- **Dataset Size:** 5,940 patients
- **Split:** 80% Train, 10% Val, 10% Test (Stratified)

## Performance Metrics
- **Final Test Accuracy:** 67.68%
- **Early Stopping Iteration:** 125

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Class 1 (Fatty) | 0.81 | 0.79 | 0.80 | 170 |
| Class 2 (Scattered) | 0.58 | 0.60 | 0.59 | 158 |
| Class 3 (Heterogeneous) | 0.59 | 0.58 | 0.59 | 151 |
| Class 4 (Dense) | 0.74 | 0.73 | 0.74 | 115 |
| **Accuracy** | | | **0.68** | **594** |
| **Macro Avg** | 0.68 | 0.68 | 0.68 | 594 |
| **Weighted Avg** | 0.68 | 0.68 | 0.68 | 594 |

### Confusion Matrix
| Actual \ Predicted | Class 1 | Class 2 | Class 3 | Class 4 |
| :--- | :---: | :---: | :---: | :---: |
| **Class 1** | 135 | 30 | 4 | 1 |
| **Class 2** | 29 | 95 | 30 | 4 |
| **Class 3** | 3 | 36 | 88 | 24 |
| **Class 4** | 0 | 4 | 27 | 84 |

## Comparison with ResNet50
- **ResNet50 Accuracy:** 69.41%
- **EfficientNet-B7 Accuracy:** 67.68%
- **Observation:** Interestingly, the simpler ResNet50 features outperformed the larger EfficientNet-B7 features by ~1.7%. This could suggest that for this specific resolution (1000x1000) and task, the ResNet50 feature space is slightly more discriminative or that the EfficientNet-B7 features require more specific hyperparameter tuning.
- **Similarities:** Both models struggle with the same Class 2 vs Class 3 boundary, and both excel at identifying Class 1 (Fatty).
