# Project: Predict house prices given specified home features using linear regression

## Overview

- This project contains Allison Dzubak's models for predicting house prices using linear regression. The dataset came from Kaggle.

## Data
- data_description.txt is the data description from Kaggle
- data_orig.csv contains the original house price dataset from Kaggle
- data_clean.csv contains the preprocessed data from notebook 0-Data-Preprocessing.ipynb

## Notebooks

### myplotprefs.py
- Plotting preferences used by the jupyter .ipynb notebooks in this directory

### 0-Data-Preprocessing.ipynb 
This notebook contains exploratory data analysis for the Kaggle house prices dataset

- Explore dataset sizes, structures, features, null/duplicate values, summary statistics
- Perform univariate and bivariate analysis / visualization
- Address null values
- Perform feature engineering, address collinear features
- Save cleaned data as data_clean.csv

### 1-Linear-Regression.ipynb
This notebook contains model testing for predicting house prices using linear regression. 

- Data preparation: 
  - Perform one-hot encoding where a 'standard reference category' feature value is dropped for interpretability
  - Scale features with MinMaxScaler or StandardScaler, for comparison
  - Check the variance inflation factor

- Description of models: 
  - Model 1: OLS 
  - Model 2: OLS with non-linearity in features accounted for 
  - Model 3: Linear model with Lasso regularization

- Model testing: 
  - Drop feature with maximum p-value until all p-values are <= 0.05
  - Compare performance of minmax versus standard scaling using LOOCV

- Model analysis and comparison: 
  - Compare LOOCV residuals for all models and error metrics
  - Perform outlier analysis
  - Perform feature importance analysis 




