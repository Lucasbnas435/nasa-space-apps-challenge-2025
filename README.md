# A World Away: Hunting for Exoplanets with AI - NASA Space Apps Challenge 2025

This repository contains a machine learning project developed for the **NASA Space Apps Challenge 2025**. The goal is to build a classification model that can accurately predict whether an object of interest observed by the Kepler Space Telescope is a potential exoplanet candidate.

Using the cumulative Kepler Object of Interest (KOI) dataset, this project involves data exploration, preprocessing, feature selection, and the training and evaluation of multiple classification models to identify the most effective one for this task.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Project Pipeline](#project-pipeline)
    1.  [Data Loading and EDA](#1-data-loading-and-exploratory-data-analysis-eda)
    2.  [Data Preprocessing](#2-data-preprocessing)
    3.  [Correlation Analysis and Feature Selection](#3-correlation-analysis-and-feature-selection)
    4.  [Model Training and Evaluation](#4-model-training-and-evaluation)
    5.  [Pipeline Exporting](#5-pipeline-exporting)
4.  [How to Use](#how-to-use)
    1.  [Prerequisites](#prerequisites)
    2.  [Installation](#installation)
    3.  [Running the Notebook](#running-the-notebook)
5.  [Model Evaluation Summary](#model-evaluation-summary)
6.  [Conclusion](#conclusion)
7.  [Future Improvements](#future-improvements)

## Project Overview

The search for exoplanets is one of the most exciting frontiers in modern astronomy. The Kepler mission has provided a vast catalog of potential candidates, but distinguishing genuine exoplanets from astrophysical false positives is a significant challenge. This project leverages machine learning to automate and improve the accuracy of this classification process.

The core objective is to build a binary classification model that takes in stellar and transit-related features from the Kepler dataset and predicts whether a Kepler Object of Interest (KOI) is a `CANDIDATE` or a `FALSE POSITIVE`.

## Dataset

The data used for this project is the cumulative Kepler Object of Interest (KOI) dataset provided by NASA. The notebook loads the data from the file `cumulative_2025.10.04_06.31.08.csv`.

  * **Rows:** 9,564
  * **Columns:** 50
  * **Target Variable:** `koi_pdisposition` - This column indicates the disposition of the KOI, which can be either `CANDIDATE` or `FALSE POSITIVE`. The dataset is well-balanced, with approximately 51% `FALSE POSITIVE` and 49% `CANDIDATE`.

The features include a variety of measurements related to the host star (e.g., effective temperature `koi_steff`, surface gravity `koi_slogg`) and the potential planet's transit (e.g., orbital period `koi_period`, transit duration `koi_duration`, transit depth `koi_depth`).

## Project Pipeline

The notebook follows a standard machine learning workflow, which is broken down into the following key stages:

### 1\. Data Loading and Exploratory Data Analysis (EDA)

The dataset is loaded into a pandas DataFrame. The initial EDA involves:

  * **Structure Analysis:** Examining the shape, data types, and first few rows of the data using `.shape`, `.info()`, and `.head()`.
  * **Missing Values:** Identifying columns with missing data. It was noted that `koi_teq_err1` and `koi_teq_err2` were completely empty.
  * **Duplicates:** Checking for and confirming the absence of duplicate rows.
  * **Descriptive Statistics:** Generating summary statistics for numerical columns with `.describe()` to understand their distribution.

### 2\. Data Preprocessing

This stage prepares the data for model training:

  * **Initial Cleaning:** The two entirely empty columns (`koi_teq_err1` and `koi_teq_err2`) were dropped immediately.
  * **Data Leakage and Irrelevant Feature Removal:** Several columns were removed to prevent data leakage and to discard irrelevant identifiers.
      * **Identifiers:** `loc_rowid`, `kepid`, `kepoi_name`, `kepler_name`.
      * **Leakage-Prone Features:** `koi_disposition` (the final adjudicated label), `koi_score` (a post-processed confidence score), and all `koi_fpflag_*` columns (flags indicating the reason for a false positive). These features contain information that would not be available for a new, unclassified object.
  * **Train/Test Split:** The data was split into training (70%) and testing (30%) sets. Stratification was used on the target variable `y` to ensure that the class distribution was maintained in both splits.
  * **Target Encoding:** The categorical target variable (`koi_pdisposition`) was mapped to a numerical format (`FALSE POSITIVE`: 0, `CANDIDATE`: 1).
  * **Feature Transformation:** A `ColumnTransformer` pipeline was built to handle categorical and numerical features separately:
      * **Categorical Features:** The `koi_tce_delivname` column was imputed using the most frequent value and then one-hot encoded.
      * **Numerical Features:** Missing values in numerical columns were handled implicitly by the tree-based models used later.

### 3\. Correlation Analysis and Feature Selection

To further refine the feature set and reduce model complexity, a correlation analysis was performed:

  * A heatmap was generated to visualize the correlation between features.
  * The correlation of each feature with the target variable was calculated.
  * Features with low correlation to the target or high multicollinearity were dropped. The following 20 columns were removed in this step:
    ```
    "koi_impact", "koi_duration", "koi_srad_err1", "koi_time0bk_err1", "koi_tce_plnt_num", "koi_time0bk", "koi_depth_err1", "koi_slogg_err2", "koi_impact_err1", "ra", "dec", "koi_depth_err2", "koi_kepmag", "koi_srad_err2", "koi_srad", "koi_slogg", "koi_impact_err2", "koi_steff", "koi_slogg_err1", "koi_tce_delivname_q1_q17_dr25_tce"
    ```

### 4\. Model Training and Evaluation

Four different tree-based classification models were trained and evaluated:

  * XGBoost (`XGBClassifier`)
  * Random Forest (`RandomForestClassifier`)
  * LightGBM (`LGBMClassifier`)
  * Decision Tree (`DecisionTreeClassifier`)

A custom evaluation function was created to report **accuracy**, a full **classification report** (precision, recall, f1-score), and a **confusion matrix** for each model. Based on the evaluation metrics, **LightGBM** was identified as the best-performing model.

### 5\. Pipeline Exporting

For deployment and reusability, a complete, end-to-end pipeline was created for each trained model. This pipeline encapsulates all the necessary steps:

1.  Initial preprocessing (imputation and one-hot encoding).
2.  Conversion of the processed array back to a DataFrame.
3.  Secondary feature selection (dropping low-correlation columns).
4.  The trained classifier model.

These pipelines were then serialized and saved as `.pkl` files using `cloudpickle`, which is robust for handling custom classes and complex objects.

## How to Use

### Prerequisites

This project uses Python and several common data science libraries. You can install all necessary packages using the provided `requirements.txt` file.

```
pandas
matplotlib
seaborn
scikit-learn
numpy
xgboost
lightgbm
cloudpickle
```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Lucasbnas435/nasa-space-apps-challenge-2025.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd machine_learning
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook

1.  Place the dataset file `cumulative_2025.10.04_06.31.08.csv` in the `../data/` directory relative to the notebook's location.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
3.  Open and run the `exoplanet_classification.ipynb` notebook. The notebook will train the models, display the evaluation results, and save the final pipelines to the `../models/` directory.

## Model Evaluation Summary

All models were evaluated on the test set. The LightGBM model demonstrated the best overall performance, particularly in accuracy and F1-score.

| Model | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.8540** | 0.85 | 0.86 | **0.86** | **0.86** | 0.84 | **0.85** |
| XGBoost | 0.8481 | 0.84 | 0.86 | 0.85 | 0.85 | 0.84 | 0.84 |
| Random Forest | 0.8443 | 0.83 | **0.87** | 0.85 | 0.86 | 0.82 | 0.84 |
| Decision Tree | 0.8014 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |

## Conclusion

This project successfully demonstrates the use of machine learning to classify Kepler Objects of Interest. After a thorough process of data cleaning, feature selection, and comparative model evaluation, the **LightGBM classifier** was selected as the final model due to its superior performance in accuracy and F1-score. The final, deployable pipeline, which includes all preprocessing steps, was exported for potential use in a production environment or further analysis.

## Future Improvements

  * **Hyperparameter Tuning:** Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the LightGBM model, which could further boost its performance.
  * **Advanced Feature Engineering:** Create new features from existing ones (e.g., ratios of stellar and transit properties) that might capture more complex relationships in the data.
  * **Handling Missing Data:** Experiment with more sophisticated imputation techniques, such as K-Nearest Neighbors (KNN) imputer, instead of relying on the default model handling.
  * **Explore Other Models:** Test other powerful classification algorithms like Support Vector Machines (SVM) or neural networks.