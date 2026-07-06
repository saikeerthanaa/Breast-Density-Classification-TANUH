# ResNet50 + LightGBM Classification Results

## Experiment Overview
- **Model:** LightGBM Classifier
- **Features:** 2048-dimensional ResNet50 features (ImageNet weights)
- **Target:** Breast Density Class (1-4)
- **Dataset Size:** 5,941 patients
- **Split:** 80% Train, 10% Val, 10% Test (Stratified)

## Performance Metrics
- **Final Test Accuracy:** 69.41%
- **Early Stopping Iteration:** 150

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Class 1 (Fatty) | 0.79 | 0.79 | 0.79 | 170 |
| Class 2 (Scattered) | 0.60 | 0.59 | 0.59 | 158 |
| Class 3 (Heterogeneous) | 0.63 | 0.66 | 0.64 | 151 |
| Class 4 (Dense) | 0.78 | 0.75 | 0.77 | 116 |
| **Accuracy** | | | **0.69** | **595** |
| **Macro Avg** | 0.70 | 0.70 | 0.70 | 595 |
| **Weighted Avg** | 0.70 | 0.69 | 0.69 | 595 |

### Confusion Matrix
| Actual \ Predicted | Class 1 | Class 2 | Class 3 | Class 4 |
| :--- | :---: | :---: | :---: | :---: |
| **Class 1** | 134 | 32 | 2 | 2 |
| **Class 2** | 33 | 93 | 30 | 2 |
| **Class 3** | 3 | 29 | 99 | 20 |
| **Class 4** | 0 | 2 | 27 | 87 |

## Observations
- The model performs best on the extreme classes (Fatty and Dense).
- There is significant overlap/confusion between Class 2 (Scattered) and Class 3 (Heterogeneously Dense), which is consistent with clinical variability.
- Almost all misclassifications are between adjacent classes, indicating that the model has learned the ordinal nature of breast density well.
