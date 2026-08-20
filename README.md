# Comment Category Prediction

A multiclass Natural Language Processing (NLP) project for classifying user comments into four categories using TF-IDF representations, text feature engineering, and LightGBM.

## Overview

The objective of this project is to classify user-generated comments into one of four target categories using both textual and structured features.

The approach combines word-level and character-level TF-IDF representations with engineered text features and a LightGBM multiclass classifier.

The final model achieved a **0.8091 Macro F1 Score** on a stratified validation set.

## Dataset

The dataset consists of:

- `train.csv` — Training data containing the target variable
- `test.csv` — Test data without the target variable
- `sample_submission.csv` — Sample submission format

Important features include:

- `comment` — Raw text content
- `created_date` — Comment timestamp
- `post_id` — Discussion thread identifier
- `emoticon_1`, `emoticon_2`, `emoticon_3` — Emoticon indicators
- `upvote`, `downvote` — Reaction counts
- `if_1`, `if_2` — Internal platform features
- `race`, `religion`, `gender`, `disability` — Platform-provided indicators
- `label` — Target class

The target variable contains **four distinct classes**.

## Model Comparison

Multiple classification models were evaluated on the same validation set:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM

The models were compared using **Accuracy and Macro F1 Score**.

The experiments showed that different models performed best depending on the evaluation metric. **XGBoost achieved the strongest validation accuracy, while LightGBM achieved the strongest Macro F1 performance.**

Since the competition involves a multiclass classification problem where performance across individual classes is important, Macro F1 was used as the primary metric for subsequent model selection and hyperparameter tuning.

The top-performing models were then shortlisted for hyperparameter tuning, taking into account the computational and time constraints of the Kaggle competition.


## Approach

The project follows an end-to-end machine learning pipeline:

```text
Data Cleaning
      ↓
Text Feature Engineering
      ↓
Word-level TF-IDF
      ↓
Character-level TF-IDF
      ↓
Feature Combination
      ↓
Stratified Train/Validation Split
      ↓
LightGBM Classification
      ↓
Macro F1 Evaluation
      ↓
Final Model Training
      ↓
Test Prediction

