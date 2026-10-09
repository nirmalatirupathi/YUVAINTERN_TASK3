# Evaluation Plan

Metrics: accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix. Use stratified k-fold cross-validation on training data; keep the test set untouched until selection is complete. For churn, recall can be important because missing an at-risk customer may cost more than contacting a non-churner. All learned preprocessing and tuning must occur inside the training/CV process.
