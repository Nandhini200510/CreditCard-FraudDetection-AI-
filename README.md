 # Credit-Card-Fraud-Detection
A real-time AI system to detect and prevent fraudulent credit card transactions using advanced machine learning techniques like LightGBM, SMOTE, and SHAP. This project addresses the critical need for robust, interpretable, and highly accurate fraud detection in the digital financial ecosystem.

# 📌 Guarding Transactions with AI-Powered Credit Card Fraud Detection and Prevention

## 🚀 Overview
With the rise of online payments and digital banking, credit card fraud has become a significant threat. This project presents a machine learning pipeline that detects fraudulent credit card transactions using **LightGBM**, manages class imbalance using **SMOTE**, and provides interpretability through **SHAP** values. The goal is to develop a transparent and reliable fraud detection model that can be integrated into real-time systems.

## ✨ Features
- ⚡ **High-Performance Model:** Uses **LightGBM**, a fast, efficient gradient boosting framework.
- ⚖ **SMOTE (Oversampling):** Balances the highly imbalanced dataset for improved fraud recall.
- 📉 **Explainable AI:** SHAP values provide feature-level interpretability.
- 🧪 **Visual Analysis:** EDA with matplotlib/seaborn and SHAP for insight into model decisions.
- 🛡 **Fraud Prevention:** Designed to reduce financial loss by proactively detecting anomalies.

## 💻 Hardware and Software Requirements
### 🔹 Hardware
- Processor: Intel i5/i7 or equivalent AMD
- RAM: Minimum 8GB (16GB recommended)
- Storage: 10GB+ free space

### 🔹 Software
- Python 3.8+
- Jupyter Notebook
- Key Libraries:  
  `lightgbm`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `shap`, `imblearn`

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Fraudulent:** 492 (0.17% of the data)
- **Features:** Time, V1-V28 (PCA-transformed), Amount, Class (target)

## 🔧 Methodology
### 1️⃣ Data Preprocessing
- Feature scaling (for Amount)
- Null value check and cleanup
- Train-test split
- **SMOTE** applied on training set to address class imbalance

### 2️⃣ Model Building - LightGBM
- Trained using optimized hyperparameters
- Model Evaluation:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - ROC-AUC

### 3️⃣ Interpretability - SHAP
- SHAP summary plot and bar chart
- Identifies top predictors of fraud
- Enhances trust in model outputs

## 📈 Results
| Metric         | Value     |
|----------------|-----------|
| **Accuracy**   | 99.85%    |
| **Precision**  | 93.12%    |
| **Recall**     | 94.45%    |
| **F1-Score**   | 93.78%    |
| **ROC-AUC**    | 0.99      |

## 📊 Visualizations
- Class distribution before and after SMOTE
- SHAP value plots for global and local feature importance
- Confusion matrix and ROC curve

## 🧠 Model Insights (via SHAP)
- **V14, V17, V10** are top influential features
- SHAP values highlight how individual features drive a prediction toward fraud or legitimate

## 📎 Tools & Technologies Used
- Python (Jupyter Notebook)
- LightGBM
- scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP
- matplotlib & seaborn

## 🎯 Conclusion
This project demonstrates a scalable, interpretable, and effective credit card fraud detection solution using machine learning. With high accuracy and model transparency, it has real-world potential to assist financial institutions in preventing fraudulent transactions in real time.

---

