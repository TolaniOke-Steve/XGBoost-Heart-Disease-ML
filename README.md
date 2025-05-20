# XGBoost-Heart-Disease-ML
## Overview
This repository is for my ENSE 411 project which contains a jupyter source file and a python script  that builds, test and trains a predicitve model using the XGBoost classifier. This model is desigend to use a binary classifier to make predictions.
## Prerequisites
To run this script, you will need to have at least Python 3.11 installed. Adittoally, you will need to have the following libraries installed:
- pandas
- numpy
- XBGBoost
- seaborn
- matplotlib
- sickit learn
- imbalanced learn

## Installation
Install the above libraries using pip:
```
!pip install pandas numpy matplotlib seaborn
```
```
!pip install --upgrade scikit-learn
!pip install --upgrade imbalanced-learn
!pip install imbalanced-learn
!pip install xgboost
````

## Model Evaluation
#### 1. Precision-Recall Plot
The Precision recall plot provides insight into the optimal treshold for this model. With an optimal treshold of 0.39, the model effectively identifies a large portion of heart disease without overly increasing the amount of false positives.
![](https://github.com/TolaniOke-Steve/XGBoost-Heart-Disease-ML/blob/main/uci/PR%20plot.png)

#### 2. AUC-ROC Curve
The AUC-ROC curve demonstrates the performance of this model in distinguishing between classes. With a score of 0.97 on the training set and 0.91 on the test set, this indicates the this model has a strong preference in differentiating cases with and cases without heart disease.
![](https://github.com/TolaniOke-Steve/XGBoost-Heart-Disease-ML/blob/main/uci/AUC-ROC.png)

#### 3. Confusion Matrix for Accuracy Metrics
The confusion matrix provides insights into the performance of the model by comparing true labels with predicted labels.

- True Positive Rate (No heart disease): 143, indicating that the model correctly predicted "heart disease" for 143 individuals.
- True Negative Rate (Heart disease): 86, indicating that the model correctly predicted "no heart disease" for 86 individuals.

![](https://github.com/TolaniOke-Steve/XGBoost-Heart-Disease-ML/blob/main/uci/Heatmap.png)

### Evaluation Summary
|Metrics|No Disease| Disease|
|----------|----------|----------|
|Accuracy| 0.84| 0.84|
|Precision|0.94| 0.79|
|Recall| 0.68| 0.97|
|F1-Score| 0.79| 0.87|
