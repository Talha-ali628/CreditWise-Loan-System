# CreditWise Loan System

A machine learning project that analyzes historical loan application data and predicts whether a loan application is likely to be **Approved** or **Rejected**.

## 📌 Project Overview

CreditWise is designed around a loan approval problem described for **SecureTrust Bank**, where manual evaluation of income, employment details, credit history, and other applicant information can be time-consuming and inconsistent.

The objective of this project is to build a machine learning based system that can learn patterns from historical loan applications and predict the loan approval outcome before final human verification.

> **Note:** This repository contains a machine learning prototype developed for educational/project purposes. It should not be treated as a production lending decision system.

## 🎯 Problem Statement

The bank receives loan applications from customers across urban and rural regions of India. The existing manual verification process creates two major challenges:

- Good customers may be rejected, resulting in lost business.
- High-risk customers may be approved, potentially resulting in financial losses.

The proposed system uses machine learning to analyze applicant information and predict whether the loan should be **Approved (1)** or **Rejected (0)**.

## 📊 Dataset

The dataset contains **1,000 loan application records** and initially includes **20 columns**.

Each row represents one loan applicant.

### Features

| Feature | Description |
|---|---|
| Applicant_ID | Unique applicant identifier |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Employment type/status |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of existing loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Requested loan amount |
| Loan_Term | Loan duration in months |
| Loan_Purpose | Purpose of the loan |
| Property_Area | Urban / Semiurban / Rural |
| Education_Level | Education level |
| Gender | Applicant gender |
| Employer_Category | Employer category |
| Loan_Approved | Target: 1 = Approved, 0 = Rejected |

The dataset contains missing values. The notebook handles these before continuing with analysis and modeling.

## 🔎 Exploratory Data Analysis

The project performs EDA to understand the dataset and the relationship between applicant characteristics and loan approval.

The analysis includes:

- Loan approval class distribution
- Education-level distribution
- Applicant income distribution
- Coapplicant income distribution
- Outlier analysis using box plots
- Credit Score vs. loan approval analysis
- Applicant Income vs. loan approval analysis
- Correlation heatmap
- Correlation of individual features with the target variable

The observed target distribution contains **722 rejected applications and 278 approved applications** before modeling.

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Missing Value Treatment

- Numerical columns are imputed using the **mean**.
- Categorical columns are imputed using the **most frequent value**.

### 2. Remove Identifier

`Applicant_ID` is removed because it is an identifier rather than a predictive feature.

### 3. Encoding

- `Education_Level` and the target variable are encoded using `LabelEncoder`.
- Other categorical variables are converted using `OneHotEncoder`.
- `drop="first"` is used to avoid redundant dummy variables.
- `handle_unknown="ignore"` is used for categorical encoding.

### 4. Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

with `random_state=42`.

### 5. Feature Scaling

`StandardScaler` is applied to scale the model input features.

## 🤖 Machine Learning Models

The project compares three classification algorithms:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Gaussian Naive Bayes**

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 📈 Initial Model Results

Before feature engineering, the models produced the following test-set results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.865 | 0.783 | 0.770 | 0.777 |
| KNN | 0.760 | 0.627 | 0.525 | 0.571 |
| Gaussian Naive Bayes | 0.865 | 0.804 | 0.738 | 0.769 |

## 🛠️ Feature Engineering

Additional features are created to capture possible non-linear relationships:

- `DTI_Ratio_sq = DTI_Ratio²`
- `Credit_Score_sq = Credit_Score²`

The original `Credit_Score` and `DTI_Ratio` features are then removed from the model input after their squared versions are created.

The models are retrained and evaluated on the engineered feature set.

## 📊 Final Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.880 | 0.785 | 0.836 | 0.810 |
| KNN | 0.785 | 0.673 | 0.574 | 0.619 |
| Gaussian Naive Bayes | 0.860 | 0.811 | 0.705 | 0.754 |

In the final comparison:

- **Logistic Regression** achieved an accuracy of **88.0%** and an F1 score of approximately **0.810**.
- **Gaussian Naive Bayes** achieved the highest precision among the final models at approximately **0.811**.
- **KNN** achieved an accuracy of **78.5%**.

These metrics are based on the notebook's 20% test split.

## 🧠 Project Workflow

```text
Loan Application Dataset
          ↓
   Data Inspection
          ↓
   Missing Value Handling
          ↓
   Exploratory Data Analysis
          ↓
   Remove Applicant_ID
          ↓
      Encoding
          ↓
   Train-Test Split
          ↓
     Feature Scaling
          ↓
  Train Classification Models
          ↓
      Model Evaluation
          ↓
    Feature Engineering
          ↓
 Retrain & Compare Models
```

## 🛠️ Technologies Used

- **Python**
- **Pandas** – data manipulation and analysis
- **NumPy** – numerical operations
- **Matplotlib** – data visualization
- **Seaborn** – statistical visualization
- **Scikit-learn** – preprocessing, model training and evaluation
- **Jupyter Notebook** – development environment

## 📁 Project Structure

```text
CreditWise-Loan-System/
│
├── credit_wise.ipynb
├── loan_approval_data.csv
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CreditWise-Loan-System.git
```

### 2. Open the project

```bash
cd CreditWise-Loan-System
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
credit_wise.ipynb
```

Make sure `loan_approval_data.csv` is in the same project directory before running the notebook.

## 📚 Key Learning Outcomes

Through this project, I worked with:

- Data inspection and cleaning
- Missing value imputation
- Exploratory Data Analysis
- Categorical variable encoding
- One-Hot Encoding
- Train-test splitting
- Feature scaling
- Classification algorithms
- Confusion matrices
- Accuracy, Precision, Recall and F1 Score
- Feature engineering
- Model comparison

## 🚀 Future Improvements

Possible improvements for a future version include:

- Hyperparameter tuning and cross-validation
- Building a reusable prediction pipeline
- Saving the trained model and preprocessing objects
- Creating an interactive Streamlit prediction interface
- Adding model explainability
- Deploying the application

## 👤 Author

**Ab Talha**

This project was developed as part of my machine learning and data science portfolio.
