# Experiment Results Summary: Ordinal Regression

The experiments for Ordinal Regression (ORNN, CORAL, CORN) have completed. All three models show significant improvement over the initial ResNet50 baseline (~0.69 Accuracy).

## Performance Comparison

| Method | Accuracy | Quadratic Kappa | MAE |
| :--- | :--- | :--- | :--- |
| **ORNN** | 0.7508 | **0.8008** | 0.2508 |
| **CORN** | **0.7740** | 0.7884 | **0.2279** |
| **CORAL** | 0.7335 | 0.7833 | 0.2700 |

## Key Insights
*   **Best Ordinal Agreement:** **ORNN** achieved the highest Quadratic Kappa (0.8008). In mammography grading, where the ordinal nature (1-4 density) is critical, Kappa is often the most important metric as it penalizes larger mistakes more heavily.
*   **Best Absolute Performance:** **CORN** achieved the highest accuracy (0.7740) and lowest Mean Absolute Error (0.2279), making it the most precise in absolute terms.
*   **Baseline Comparison:** Both models significantly outperformed the previous ResNet50 baseline accuracy of ~0.69.

## Next Steps
*   **Recommendation:** Given the clinical importance of ordinal consistency in breast density classification, **ORNN** is likely the preferred candidate due to its superior Kappa score.
*   **Deployment:** The best models (`best_ornn_model.pt`, `best_corn_model.pt`, `best_coral_model.pt`) have been saved and are ready for inference.
