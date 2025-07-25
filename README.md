# 🛡️ Credit Card Fraud Detection API

This is a Flask-based REST API that detects fraudulent credit card transactions using a trained **Random Forest classifier**.\
The model is trained on the popular [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), where the goal is to identify transactions likely to be fraudulent.

---

## 🚀 Features

- Predicts if a transaction is fraudulent (`1`) or valid (`0`)
- Returns both class prediction and probability scores
- Accepts JSON input with 30 transaction features (`Time`, `V1–V28`, `Amount`)
- Ready for deployment and frontend integration
- Easily extendable for batch predictions or CSV file uploads

---

## 📁 Project Structure

```
CREDITCARDFRAUDDETECTION/
├── dataset/
│   └── creditcard.csv (not included due to size)
├── EDA/
│   ├── amount_distribution.png
│   ├── class_amount_distribution.png
│   ├── class_distribution.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── correlation_matrix.png
│   ├── features_of_interest_by_class.png
│   ├── features_of_interest_distribution.png
│   └── time_amount_relationship.png
├── model/
│   └── rf_fraud_model.pkl
├── templates/
│   └── index.html
├── app.py               # Flask API
├── main.py              # Model training + EDA
├── test.py              # Script to test API
├── requirements.txt     
└── README.md
```

---

## 🛆 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/Decadent-tech/creditcardfrauddetection.git
cd creditcardfrauddetection
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

4. **Run the API**

```bash
python app.py
```

---

## 🔗 API Endpoint

- **URL:** `http://127.0.0.1:5000`
- **Route:** `POST /predict`

### 📝 Request JSON Format

```json
{
  "features": [10000.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.046, 0.24, 0.09, 0.36,
               0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.020, 0.022,
               0.40, 0.25, -0.01, 0.27, -0.11, 0.06, 0.12, -0.18, 0.013, 0.04, 149.62]
}
```

### ✅ Sample Response

```json
{
  "prediction": 0,
  "probability_valid": 0.9873,
  "probability_fraud": 0.0127
}
```

---

## 📊 Model Details

- **Algorithm:** RandomForestClassifier
- **Accuracy:** \~99.9%
- **Data Handling:** Balanced using undersampling or SMOTE
- **Trained on:** Kaggle [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  - 284,807 transactions
  - 492 frauds
  - Features anonymized (`V1–V28`)

---

## 👩‍💼 Author

**Debosmita Chatterjee**\
*Data Science Enthusiast | Building ML Apps*

---

## 📜 License

This project is licensed under the MIT License.

