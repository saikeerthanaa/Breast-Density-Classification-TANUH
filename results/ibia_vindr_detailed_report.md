# Detailed Comparison Report: ConvNeXt + CE vs ConvNeXt + CORAL
Conducted on class-balanced versions of external datasets (IBIA & VinDr).

## 📊 Dataset: IBIA
----------------------------------------
### 📈 Per-Class Metrics Comparison
| Class | Metric | ConvNeXt + CE only | ConvNeXt + CORAL | Difference |
| :--- | :--- | :---: | :---: | :---: |
| Class A | Precision | 0.7500 | 0.6889 | -0.0611 |
| Class A | Recall | 0.3529 | 0.4559 | +0.1029 |
| Class A | F1-score | 0.4800 | 0.5487 | +0.0687 |
| Class B | Precision | 0.3030 | 0.3919 | +0.0889 |
| Class B | Recall | 0.2941 | 0.4265 | +0.1324 |
| Class B | F1-score | 0.2985 | 0.4085 | +0.1099 |
| Class C | Precision | 0.3925 | 0.4737 | +0.0812 |
| Class C | Recall | 0.6176 | 0.5294 | -0.0882 |
| Class C | F1-score | 0.4800 | 0.5000 | +0.0200 |
| Class D | Precision | 0.6866 | 0.7013 | +0.0147 |
| Class D | Recall | 0.6765 | 0.7941 | +0.1176 |
| Class D | F1-score | 0.6815 | 0.7448 | +0.0633 |


### 🧩 Confusion Matrices
#### ConvNeXt + CE only Confusion Matrix:
| True \ Pred | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 24 | 34 | 9 | 1 |
| **B** | 5 | 20 | 41 | 2 |
| **C** | 0 | 8 | 42 | 18 |
| **D** | 3 | 4 | 15 | 46 |

#### ConvNeXt + CORAL Confusion Matrix:
| True \ Pred | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 31 | 31 | 5 | 1 |
| **B** | 10 | 29 | 27 | 2 |
| **C** | 3 | 9 | 36 | 20 |
| **D** | 1 | 5 | 8 | 54 |


### 🔬 Statistical Significance Test (McNemar's Test)
- **Contingency Table (Correctness):**
  - Both Correct: 116
  - CE Correct, CORAL Incorrect: 16
  - CE Incorrect, CORAL Correct: 34
  - Both Incorrect: 106
- **Chi-Squared Statistic:** 5.7800
- **p-value:** 1.6210e-02
- **Conclusion:** **Statistically Significant (p < 0.05)**

================================================================================

## 📊 Dataset: VinDr
----------------------------------------
### 📈 Per-Class Metrics Comparison
| Class | Metric | ConvNeXt + CE only | ConvNeXt + CORAL | Difference |
| :--- | :--- | :---: | :---: | :---: |
| Class A | Precision | 1.0000 | 0.9643 | -0.0357 |
| Class A | Recall | 0.1800 | 0.2700 | +0.0900 |
| Class A | F1-score | 0.3051 | 0.4219 | +0.1168 |
| Class B | Precision | 0.4357 | 0.4573 | +0.0216 |
| Class B | Recall | 0.6100 | 0.7500 | +0.1400 |
| Class B | F1-score | 0.5083 | 0.5682 | +0.0598 |
| Class C | Precision | 0.4706 | 0.5698 | +0.0992 |
| Class C | Recall | 0.5600 | 0.4900 | -0.0700 |
| Class C | F1-score | 0.5114 | 0.5269 | +0.0155 |
| Class D | Precision | 0.7073 | 0.6967 | -0.0106 |
| Class D | Recall | 0.8700 | 0.8500 | -0.0200 |
| Class D | F1-score | 0.7803 | 0.7658 | -0.0145 |


### 🧩 Confusion Matrices
#### ConvNeXt + CE only Confusion Matrix:
| True \ Pred | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 18 | 69 | 13 | 0 |
| **B** | 0 | 61 | 37 | 2 |
| **C** | 0 | 10 | 56 | 34 |
| **D** | 0 | 0 | 13 | 87 |

#### ConvNeXt + CORAL Confusion Matrix:
| True \ Pred | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 27 | 70 | 2 | 1 |
| **B** | 0 | 75 | 21 | 4 |
| **C** | 1 | 18 | 49 | 32 |
| **D** | 0 | 1 | 14 | 85 |


### 🔬 Statistical Significance Test (McNemar's Test)
- **Contingency Table (Correctness):**
  - Both Correct: 201
  - CE Correct, CORAL Incorrect: 21
  - CE Incorrect, CORAL Correct: 35
  - Both Incorrect: 143
- **Chi-Squared Statistic:** 3.0179
- **p-value:** 8.2352e-02
- **Conclusion:** Not Statistically Significant (p >= 0.05)

================================================================================
