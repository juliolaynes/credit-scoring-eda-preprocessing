# Credit Scoring: End-to-End Machine Learning Pipeline

This project builds a robust, end-to-end Machine Learning pipeline to analyze, preprocess, and classify customer credit risk. The core objective is to evaluate financial risk, helping to determine which profiles are eligible for higher credit limits and which present a higher risk of default.

## 💼 Business Problem & Objectives
In the financial sector, accurately assessing credit risk is crucial to balance profitability and default rates. This repository contains a complete data science solution to:
* **Understand the socio-economic and behavioral factors** that influence credit scores.
* **Perform univariate and bivariate analyses** to uncover hidden patterns in data distributions.
* **Clean, transform, and balance the dataset** addressing real-world issues like missing values and highly imbalanced classes.
* **Train and evaluate a Predictive Model** capable of accurately classifying new applicants into Low, Average, or High credit scores.

## ⚙️ Data Pipeline & Methodology
The project is structured into three main phases using Python:

### 1. Exploratory Data Analysis (EDA)
* **Univariate Analysis:** Inspecting individual variables (income, age, education) to understand their distribution and detect outliers using Pandas and Seaborn.
* **Bivariate Analysis:** Crossing applicant features with their credit classification to identify strong correlations and risk indicators.
* **Interactive Data Visualization:** Utilizing Plotly to build dynamic charts, allowing for a deeper look into specific data points and distributions.

### 2. Data Preprocessing & Feature Engineering
* **Data Cleaning:** Handling missing records and formatting inconsistent structural data (such as cleaning currency strings into numerical floats).
* **Feature Encoding:** Transforming categorical features into numerical binaries via One-Hot Encoding (`pd.get_dummies`), avoiding data leakage by splitting the data first.
* **Class Imbalance Handling:** Using **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training set to balance the target classes (High, Average, Low) and prevent the model from biasedly favoring the majority class.

### 3. Model Training & Evaluation
* Trained a **Random Forest Classifier** on the balanced training data and evaluated its performance on an untouched test set (25% of the data).

## 📊 Model Performance & Business Results

The predictive pipeline utilized a **Random Forest Classifier** and achieved an outstanding overall **Accuracy of 94%** on the test dataset. Instead of just chasing high numbers, the model was evaluated based on metrics that directly impact financial risk:

* **Perfect Risk Detection (Low Class - 100% Recall & Precision):** In credit scoring, the most expensive mistake for a financial institution is granting credit to a high-risk applicant (a False Negative). The model achieved a perfect score (**1.00**) in both identifying every single high-risk profile and ensuring zero false alarms for this class.
* **Excellent Identification of High-Eligibility Profiles (High Class - 100% Precision):** The model achieved maximum precision when flagging clients with excellent credit quality. This ensures that credit limit expansions or premium product offers are strictly targeted at the right audience, safeguarding the bank's margin.
* **Balanced Performance via SMOTE:** Despite the original dataset containing less than 9% of high-risk ('Low') profiles, the synthetic over-sampling strategy successfully trained the algorithm to recognize risk patterns without bias toward the majority class.

### 🧠 Key Insights from the Results:
* **Perfect Risk Detection (Low):** The model achieved a **100% Recall and Precision for the 'Low' credit score class**. In credit scoring, failing to detect a high-risk profile (false negative) is the most expensive mistake for a bank. The pipeline completely neutralized this risk.
* **Robust Balanced Learning:** Despite the original dataset having only ~8% of 'Low' score instances, the implementation of SMOTE successfully enabled the algorithm to master the characteristics of high-risk clients.

## 🛠️ Technologies Used
* **Python** (Core workflow)
* **Pandas & NumPy** (Data manipulation, profiling, and cleaning)
* **Scikit-Learn** (Train/test split, Random Forest Classifier, and evaluation metrics)
* **Imbalanced-Learn** (SMOTE for over-sampling)
* **Plotly** (Interactive data visualization)
* **Matplotlib & Seaborn** (Static data visualization and correlation heatmaps)
