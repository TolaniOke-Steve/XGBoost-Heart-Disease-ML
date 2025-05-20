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
The Precision recall plot provides insight into the optimal treshold for this model. With an optimal treshoold of 0.39,the model effectively identifies a large portion of heart disease without overly increasing the amount of false positives.
![](https://github.com/TolaniOke-Steve/XGBoost-Heart-Disease-ML/blob/main/uci/PR%20plot.png)
