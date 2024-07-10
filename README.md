
# Term Project: Credit Card Transaction Fraud Detection Using Machine Learning

## Overview

This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset consists of credit card transactions, including both legitimate and fraudulent activities. The primary goal is to compare the performance of different models and identify the best-performing model for fraud detection.

## Dataset Description

The dataset used in this project is the Credit Card Transactions Fraud Detection Dataset from Kaggle. It covers transactions from January 1, 2019, to December 31, 2020, involving 1000 customers and 800 merchants. The dataset contains 23 columns and 287,015 observations.

## Data Preprocessing

- Renaming columns for clarity:
  - `trans_date_trans_time` → `transaction_time`
  - `cc_num` → `credit_card_number`
  - `amt` → `amount(usd)`
  - `trans_num` → `transaction_id`
- Feature selection:
  - `transaction_id`
  - `hour_of_day`
  - `category`
  - `amount(usd)`
  - `merchant`
  - `job`

## Model Training and Evaluation

The following machine learning models were trained and evaluated:

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Gradient Boosting
- XGBoost
- Artificial Neural Network (ANN)

### Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

## Installation and Usage

### Prerequisites

- Python 3.7 or later
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `keras`, `tensorflow`, `matplotlib`, `seaborn`

### Usage

1. Open Jupyter Notebook
2. Download Dataset: [Kaggle Fraud Detection Data](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
3. Install required packages
4. Run the code


