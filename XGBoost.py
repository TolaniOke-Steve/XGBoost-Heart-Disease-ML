# Importing the libraries
import matplotlib
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Load the data
train_data = np.load('train_data.npz', allow_pickle=True)
val_data = np.load('val_data.npz', allow_pickle=True)
test_data = np.load('test_data.npz', allow_pickle=True)

X_train = train_data['X_train']
y_train = train_data['y_train']
x_val = val_data['X_val']
y_val = val_data['y_val']
X_test = test_data['X_test']
y_test = test_data['y_test']

# hyperparameter tuining and traning model
model = xgb.XGBClassifier(eval_metric="mlogloss", max_depth=5, learning_rate=0.069, n_estimators=201, gamma=0.98, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train)

# Predicting probabilities
pred_probs = model.predict_proba(X_test)

# Collapse classes
y_test_binary = (y_test > 0).astype(int)  # Convert Class 1,2,3 → 1 (Disease) | Class 0 → 0 (No Disease)
prob_disease = pred_probs[:, 1:].sum(axis=1)  # Sum probabilities of class 1, 2, 3

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test_binary, prob_disease)

# Youden's Index to get the best threshold
youden_index = recall + precision - 1
best_threshold_index = np.argmax(youden_index)
best_threshold = thresholds[best_threshold_index]


# Precision and Recall vs Threshold
matplotlib.use('TkAgg') 

prob_disease_test = pred_probs[:, 1:].sum(axis=1)

# ROC curve and AUC for the test set
fpr_test, tpr_test, _ = roc_curve(y_test_binary, prob_disease_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Probabilities for the train set
train_pred_probs = model.predict_proba(X_train)
prob_disease_train = train_pred_probs[:, 1:].sum(axis=1)

# Convert y_train to binary
y_train_binary = (y_train > 0).astype(int)

# ROC curve and AUC for the train set
fpr_train, tpr_train, _ = roc_curve(y_train_binary, prob_disease_train)
roc_auc_train = auc(fpr_train, tpr_train)

# ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', label=f'Test AUC = {roc_auc_test:.2f}')
plt.plot(fpr_train, tpr_train, color='green', linestyle='--', label=f'Train AUC = {roc_auc_train:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Reference diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve: Train vs Test')
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Score (Precision / Recall)')
plt.title('Precision and Recall vs Threshold for Detecting Heart Disease')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# optimal threshold prediction
pred_optimal = (prob_disease >= best_threshold).astype(int)

print("Optimal Decision Threshold:", best_threshold)
print("\nAccuracy:", accuracy_score(y_test_binary, pred_optimal))
print("\nClassification Report:\n", classification_report(y_test_binary, pred_optimal))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_binary, pred_optimal))



"""matplotlib.use('TkAgg') 

plt.figure(figsize=(10,6))
xgb.plot_importance(model, height=0.5, grid=False, show_values=True)
plt.title('XGBoost Feature Importance')
plt.show()
"""
