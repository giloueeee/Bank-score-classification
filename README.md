# Bank Score Classification

A machine learning project for credit score classification using XGBoost.

## Overview

This project implements a credit score classification model that predicts customer credit scores based on various financial and personal features. The model uses extensive data cleaning and preprocessing techniques to handle missing values, outliers, and categorical variables.

<p align="center">
<img width="600px" src="https://github.com/user-attachments/assets/86cc219c-882f-4932-867a-e1d6bb1dcadc"/>
</p>
<p align="center">
Accuracy scores
</p>
## Features

- Comprehensive data cleaning and preprocessing
- Outlier detection and removal
- Feature engineering (one-hot encoding, ordinal encoding)
- Class balancing for imbalanced datasets
- XGBoost classifier with hyperparameter tuning
- Model evaluation with accuracy, precision, recall, and F1-score metrics

## Requirements

Install the required packages using:


pip install -r requirements.txt


## Data

Place your data files in the `data/` directory:
- `data/train.csv` - Training dataset with features and target variable (Credit_Score)
- `data/test.csv` - Test dataset for predictions

## Usage

Run the training script:

python train.py


The script will:
1. Load and clean the training data
2. Perform feature engineering and preprocessing
3. Balance the dataset
4. Train multiple XGBoost models with different hyperparameters
5. Display the best model performance metrics
6. Generate heatmaps showing model performance across different hyperparameter combinations

## Output

- `data/cleaned_data.csv` - Cleaned and preprocessed dataset (generated automatically)

## Model Performance

The script performs a grid search over:
- `n_estimators`: [110, 125, 150, 170, 200, 300, 400, 500, 750, 1000]
- `max_depth`: [13, 14, 15, 20, 25]
- `learning_rate`: [0.05, 0.1, 0.15]

The best model configuration and metrics are printed to the console, and heatmaps are displayed for visualization.

## Data Cleaning Steps

The preprocessing pipeline includes:
- Age normalization and outlier removal (18-60 years)
- Occupation cleaning
- Annual income outlier removal (< 2,000,000)
- Bank accounts, credit cards, and loan count validation
- Interest rate outlier removal (< 35)
- Payment behavior and credit history age processing
- Missing value imputation
- Categorical variable encoding



