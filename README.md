# 📦 E-Commerce Product Delivery Prediction

## 📌 Project Overview

This project focuses on predicting whether an e-commerce product will reach the customer on time.

The project uses machine learning classification algorithms to analyze customer, product, and shipment-related information and predict the delivery outcome.

The objective is to support better delivery planning, improve customer satisfaction, and provide useful operational insights.

---

## 🎯 Problem Statement

The goal of this project is to predict whether an e-commerce order will reach the customer on time based on factors such as:

- Warehouse block
- Mode of shipment
- Product importance
- Customer care calls
- Customer rating
- Cost of the product
- Prior purchases
- Discount offered
- Weight of the product
- Gender

The target variable is:

`Reached.on.Time_Y.N`

- `1` = Reached On Time
- `0` = Not Reached On Time

---

## 📊 Dataset

The dataset contains **10,999 records and 12 columns**.

### Dataset Features

| Feature | Description |
|---|---|
| ID | Unique order/customer identifier |
| Warehouse_block | Warehouse block handling the order |
| Mode_of_Shipment | Shipment method |
| Customer_care_calls | Number of customer care calls |
| Customer_rating | Customer rating |
| Cost_of_the_Product | Product cost |
| Prior_purchases | Number of previous purchases |
| Product_importance | Importance level of the product |
| Gender | Customer gender |
| Discount_offered | Discount offered on the product |
| Weight_in_gms | Product weight in grams |
| Reached.on.Time_Y.N | Delivery outcome |

---

## 🔍 Exploratory Data Analysis

The project includes exploratory data analysis to understand the dataset and identify important patterns.

The following analyses were performed:

- Dataset structure and data types
- Missing value analysis
- Duplicate value analysis
- Descriptive statistics
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Categorical feature analysis
- Numerical feature analysis
- Correlation analysis
- Target variable distribution

### Key EDA Observations

- The dataset contains no missing values.
- No duplicate rows were found.
- Approximately **59.7%** of the orders reached on time.
- Approximately **40.3%** of the orders did not reach on time.
- `Discount_offered` showed a noticeable relationship with delivery status.
- `Weight_in_gms` also showed a meaningful relationship with delivery outcome.
- Product cost and customer-related features showed comparatively smaller relationships with the target variable.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Duplicate value checking
2. Missing value checking
3. Outlier analysis using the IQR method
4. Feature and target separation
5. Removal of the `ID` feature from model training
6. Train-test split
7. One-hot encoding of categorical variables
8. Feature scaling using StandardScaler

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

Stratified splitting was used to maintain the target class distribution.

After preprocessing, the dataset contained **19 model features**.

---

## 🛠️ Feature Engineering

Categorical variables were converted into numerical representations using **OneHotEncoder**.

Categorical features:

- Warehouse_block
- Mode_of_Shipment
- Product_importance
- Gender

Numerical features:

- Customer_care_calls
- Customer_rating
- Cost_of_the_Product
- Prior_purchases
- Discount_offered
- Weight_in_gms

Feature importance was also analyzed using a Random Forest model.

The most important features in the preliminary feature-importance analysis were:

1. Weight in grams
2. Discount offered
3. Cost of the product
4. Prior purchases
5. Customer rating
6. Customer care calls

---

## 🤖 Machine Learning Models

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

Both baseline models and hyperparameter-tuned models were evaluated.

---

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**.

Two tuning approaches were evaluated:

### Accuracy-Based Tuning

The models were tuned using accuracy as the scoring metric.

### F1-Score-Based Tuning

The models were also tuned using F1-score as the scoring metric.

The comparison demonstrated that optimizing one metric does not necessarily improve all other evaluation metrics.

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

### Baseline Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 63.77% | 70.67% | 67.17% | 68.88% |
| Decision Tree | 65.09% | 70.08% | 72.43% | **71.24%** |
| Random Forest | 65.36% | 75.44% | 62.22% | 68.20% |
| KNN | 62.41% | 69.76% | 65.35% | 67.48% |
| SVM | 64.68% | 77.35% | 57.73% | 66.11% |

---

## 🏆 Final Model

The **Baseline Decision Tree Classifier** was selected as the final model based on its F1-score among the evaluated models.

### Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 65.09% |
| Precision | 70.08% |
| Recall | 72.43% |
| F1-Score | 71.24% |

The final Decision Tree model is used in the Streamlit application for delivery prediction.

---

## 🌐 Streamlit Web Application

A Streamlit application was developed to allow users to enter product, customer, and shipment information and receive a predicted delivery outcome.

### Application Features

- Warehouse selection
- Shipment mode selection
- Product importance selection
- Product cost input
- Discount input
- Gender selection
- Customer care calls
- Customer rating
- Prior purchases
- Product weight
- Delivery prediction
- Estimated prediction probabilities
- Entered product details
- Model performance information

The application uses the same preprocessing workflow used during model development.

---

## 📁 Project Structure

```text
E-Commerce-Product-Delivery-Prediction/
│
├── E_Commerce.csv
├── E-Commerce_Product_Delivery_Prediction.ipynb
├── app.py
├── decision_tree_model.pkl
├── preprocessor.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
