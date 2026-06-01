# Customer Churn Prediction

## 📌 Project Overview

This project predicts whether a customer is likely to churn (leave a service) based on customer demographics, account information, and service usage patterns. The goal is to help businesses identify at-risk customers and improve customer retention strategies.

---

## 🎯 Problem Statement

Customer churn is a major challenge for subscription-based businesses. By leveraging Machine Learning, this project aims to classify customers into:

- Churn = Yes
- Churn = No

---

## 📊 Dataset Information

The dataset contains customer information such as:

- Gender
- Senior Citizen Status
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

**Target Variable:** `Churn`

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

## 🔄 Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Train-Test Split
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Deployment using Streamlit

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate model performance:

### Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Precision

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

$$
Recall = \frac{TP}{TP + FN}
$$

### F1 Score

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### ROC-AUC Score

$$
AUC = \int_{0}^{1} TPR(FPR)\,d(FPR)
$$

---

## 📊 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 73.95 %  |
| F1 Score | 61.49 %  |
| ROC-AUC | 84.11 %   |

---

## 🚀 Project Structure

```text
Customer-Churn-Prediction/
│
├── artifacts/
├── notebook/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│
├── app.py
├── requirements.txt
├── README.md
└── setup.py
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/SurajSaithwar18/Customer-Churn-using-Machine-Learning
```

Move to project directory:

```bash
cd customer-churn-prediction
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit App:

```bash
streamlit run app.py
```

---

## 📸 Application Preview

Add screenshots of your Streamlit application here.

---

## 🔮 Future Improvements

- Deep Learning Models
- Explainable AI (SHAP)
- Real-time Predictions
- Cloud Deployment
- Automated Retraining Pipeline

---

## 👨‍💻 Author

Suraj Saithwar

Computer Science Engineering Student

Machine Learning & Data Science Enthusiast
