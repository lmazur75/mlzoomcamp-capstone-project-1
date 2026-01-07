# Machine Failure Prediction -- Capstone Project

## Project Overview

This project aims to build a machine learning model capable of
predicting machine failures based on operational and process-related
data. The goal is to anticipate failures before they occur, supporting
preventive maintenance and reducing downtime.

The project was developed as a capstone following best practices in data
preparation, exploratory data analysis, model selection, and
hyperparameter tuning.

## Dataset

The dataset used comes from Kaggle:

**Machine Failure Predictions**\
Source: Kaggle (Dinesh Manikanta)

The dataset contains numerical sensor readings and categorical
attributes related to machine operation, along with a binary target
variable indicating machine failure.

## Project Structure

    .
    ├── notebook.ipynb        # Main analysis notebook
    ├── README.md             # Project documentation
    └── data/                 # Dataset files (not included in repo)

## Methodology

### 1. Data Preparation and Cleaning

-   Verified data types and handled categorical features
-   Used `ColumnTransformer` with:
    -   Numerical features passed through unchanged
    -   Categorical features encoded using `OneHotEncoder`
-   Ensured no data leakage by fitting preprocessing only on training
    data
-   Integrated preprocessing and modeling using `Pipeline`

### 2. Exploratory Data Analysis (EDA)

-   Analyzed class distribution of machine failures
-   Examined feature distributions and correlations
-   Identified potentially important sensor and process variables

### 3. Feature Importance Analysis

Feature importance was analyzed using tree-based models: - **Random
Forest** for exploratory importance analysis - **XGBoost** for final
model interpretation

Feature importances were extracted correctly from pipeline models,
ensuring alignment with encoded feature names.

### 4. Model Selection and Hyperparameter Tuning

The following models were evaluated: - Logistic Regression - Decision
Tree - Random Forest - XGBoost

Hyperparameter tuning was performed using `GridSearchCV` with
cross-validation. Model selection was based on validation performance
and robustness.

## Final Model

-   **XGBoost Classifier**
-   Integrated into a full preprocessing + modeling pipeline
-   Provided the best balance between performance and interpretability

## Key Results

-   Tree-based models outperformed linear models
-   Sensor and process-related features were strong predictors of
    failures
-   Feature importance analysis provided actionable insights into
    failure drivers

## Tools and Libraries

-   Python
-   pandas, numpy
-   scikit-learn
-   xgboost
-   matplotlib

## How to Run

1.  Clone the repository

2.  Create a virtual environment

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Open and run `notebook.ipynb`

## Notes

-   XGBoost requires the `xgboost` package to be installed separately
-   Feature importance extraction from pipelines requires accessing the
    classifier via `named_steps`

