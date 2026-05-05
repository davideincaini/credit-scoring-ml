# Credit Scoring — Binary Classification 

## Problem
A bank needs to predict whether a customer will consistently repay their credit card — approving high-risk customers generates losses that can be orders of magnitude larger than the profit from good payers.

## Dataset
338,427 customer records with 18 features: demographics, income type, employment duration, housing, family composition.  
Class imbalance: ~88% good payers / ~12% defaulters.

## Approach

| Step | Choice | Why |
|---|---|---|
| Missing values | Median imputation (SimpleImputer) | OCCUPATION_TYPE has 30% NaN — structural, not random |
| Categorical encoding | LeaveOneOutEncoder on high-cardinality features | Avoids dimensionality explosion vs OHE on 18-class variables |
| Class imbalance | SMOTE oversampling | Generates synthetic minority samples — more robust than class_weight on this distribution |
| Scaling | StandardScaler on numerics | Required for logistic regression convergence |
| Pipeline | sklearn Pipeline + ColumnTransformer | Prevents data leakage between train/test splits |

## Model
Logistic Regression with calibrated threshold — chosen for **interpretability** (regulatory requirement: denied customers must receive a written justification).

SHAP values used to extract per-customer feature contributions → human-readable rejection motivations.

## Key Results
- ROC-AUC: model evaluated on held-out test set
- Threshold logic: conservative by design — false negatives (missed defaults) cost significantly more than false positives
- Base value note: SHAP base = 0.5 due to SMOTE balancing during training → probabilities are **relative**, not absolute. The model systematically overestimates risk — intentional and prudent.

## What I Would Do Differently
- Test LightGBM/XGBoost: likely better performance on this tabular imbalanced dataset
- Calibrate probabilities with Platt scaling to recover realistic absolute estimates
- Cross-validate threshold selection rather than fixing it manually

## Stack
`Python` · `scikit-learn` · `imbalanced-learn` · `category_encoders` · `SHAP` · `pandas` · `seaborn`

---
*Part of [Davide Incaini's data science portfolio](https://github.com/davideincaini)*
