# 🧾 Vendor Invoice Processing System

An AI-driven vendor invoice analytics system that uses Machine Learning to predict freight costs and identify invoices that require manual approval.

The project combines **SQL-based data extraction, data preprocessing, feature engineering, Machine Learning, model evaluation, model inference, and a Streamlit web application** into an end-to-end invoice analytics workflow.

---

## 🚀 Project Overview

Vendor invoices contain important information such as invoice amount, quantity, freight cost, purchase order details, and receiving delays.

This project uses historical vendor invoice and purchase data to build two Machine Learning solutions:

### 1. 💰 Freight Cost Prediction

Predicts the expected freight cost of an invoice based on its dollar value.

### 2. 🚨 Invoice Flagging

Classifies invoices into:

- `0` → Normal invoice
- `1` → Invoice requiring manual approval

The system is designed to help finance and procurement teams identify invoices that may require additional review.

---

## 🎯 Objectives

- Predict freight costs using Machine Learning
- Identify potentially abnormal invoices
- Reduce manual invoice review
- Support vendor and procurement analytics
- Build reusable ML training and inference pipelines
- Provide an interactive interface using Streamlit

---

## 🛠️ Technologies Used

### Programming
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Linear Regression
- Decision Tree
- Random Forest
- GridSearchCV
- StandardScaler

### Database
- SQLite
- SQL

### Model Persistence
- Joblib

### Application
- Streamlit

### Development
- VS Code
- Jupyter Notebook
- Git & GitHub

---

# 📊 Project Modules

## 💰 1. Freight Cost Prediction

The freight prediction module predicts the expected freight cost using invoice dollar value.

### Feature Used

```text
Dollars
```

### Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Evaluation Metrics

- MAE
- RMSE
- R² Score

The trained freight prediction model is saved using Joblib and reused during inference.

---

## 🚨 2. Invoice Flagging

The invoice classification module predicts whether an invoice should be flagged for manual approval.

### Features

```text
invoice_quantity
invoice_dollars
Freight
days_po_to_invoice
total_item_quantity
total_item_dollars
avg_receiving_delay
```

### Classification Model

A Random Forest Classifier is used for the invoice flagging pipeline.

### Hyperparameter Optimization

GridSearchCV is used to search across different Random Forest configurations.

The grid includes parameters such as:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `criterion`

The model is evaluated using F1 score during Grid Search.

---

# 🗄️ Database

The project uses an SQLite database containing vendor and purchasing information.

### Main Tables

```text
purchases
purchase_prices
vendor_invoice
begin_inventory
end_inventory
```

The data is combined using SQL queries and feature engineering before being passed to the Machine Learning pipelines.

---

# 🔄 ML Pipeline

The overall workflow is:

```text
                SQLite Database
                       │
                       ▼
               SQL Data Extraction
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
                 Train / Test Split
                       │
                       ▼
                    Scaling
                       │
                       ▼
              Machine Learning Model
                       │
                       ▼
                 Model Evaluation
                       │
                       ▼
               Save Trained Model
                       │
                       ▼
                   Inference
                       │
                       ▼
                Streamlit Application
```

---

# 🧠 Invoice Flagging Logic

The invoice flag target was created using business rules based on invoice and receiving information.

An invoice is flagged when:

```python
if abs(invoice_dollars - total_item_dollars) > 5:
    return 1

if avg_receiving_delay > 10:
    return 1

return 0
```

Therefore:

```text
1 → Invoice requires attention/manual approval
0 → Normal invoice
```

> The `1` class represents an invoice requiring review based on the project's defined rules. It is not automatically classified as fraud.

---

# 📈 Model Results

## Freight Prediction

The regression models were evaluated using MAE, RMSE, and R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ~24 | ~125 | ~97% |
| Decision Tree | ~33 | ~150 | ~96% |
| Random Forest | ~27 | ~132 | ~97% |

These results were obtained during the project's model evaluation and are approximate because the exact values can vary depending on the current training run.

---

## Invoice Classification

The invoice classification pipeline evaluates the Random Forest model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The model achieved approximately **96%+ accuracy** during evaluation.

---

# 🌐 Streamlit Application

The project includes an interactive Streamlit application.

The application provides two modules:

### Freight Cost Prediction

Users enter:

```text
Invoice Dollars
```

The system returns:

```text
Estimated Freight Cost
```

### Invoice Manual Approval Prediction

Users enter:

```text
Invoice Quantity
Invoice Dollars
Freight Cost
Days from PO to Invoice
Total Item Quantity
Total Item Dollars
Average Receiving Delay
```

The system returns either:

```text
🚨 Invoice requires MANUAL APPROVAL
```

or:

```text
✅ Invoice is SAFE for Auto-Approval
```

---

# 📁 Project Structure

```text
Vendor-Invoice-Processing-System/
│
├── data/
│   └── inventory.db
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── evaluate_classification_model.py
│   └── train.py
│
├── inferencing/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── models/
│   ├── predict_freight_model.pkl
│   └── random_forest_model.pkl
│
├── notebook/
│   ├── invoiceflagging.ipynb
│   └── PredictingFreightCost.ipynb
│
├── tests/
│   └── preprocessingtest.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> The database file is excluded from Git through `.gitignore` because of its large size.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/PrathikshaRS/Vendor-Invoice-Processing-System.git
```

## 2. Navigate to the Project

```bash
cd Vendor-Invoice-Processing-System
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Virtual Environment

### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

---

# 🔬 Running the ML Pipelines

## Freight Prediction

```bash
python freight_cost_prediction/train.py
```

## Invoice Classification

```bash
python invoice_flagging/train.py
```

---

# 🔮 Running Inference

## Freight Prediction

```bash
python inferencing/predict_freight.py
```

## Invoice Flag Prediction

```bash
python inferencing/predict_invoice_flag.py
```

---

# 📓 Notebooks

The project contains notebooks used for experimentation, analysis, and model development.

### `PredictingFreightCost.ipynb`

Contains the analysis and development of the freight cost prediction model.

### `invoiceflagging.ipynb`

Contains the analysis, preprocessing, classification modeling, and evaluation for invoice flagging.

---

# 🧪 Testing

The project includes preprocessing tests under:

```text
tests/preprocessingtest.py
```

Tests can be extended to validate:

- Data loading
- Feature creation
- Label generation
- Preprocessing
- Model input structure

---

# 📌 Key Concepts Demonstrated

This project demonstrates practical implementation of:

- SQL data extraction
- SQLite database handling
- Data preprocessing
- Feature engineering
- Exploratory Data Analysis
- Correlation analysis
- Train-test splitting
- Feature scaling
- Regression
- Classification
- Random Forest
- Hyperparameter tuning
- GridSearchCV
- Model evaluation
- Confusion matrix
- MAE
- RMSE
- R²
- Precision
- Recall
- F1 Score
- Model persistence using Joblib
- ML inference
- Streamlit application development
- Git and GitHub

---

# 🔮 Future Improvements

- Add invoice history and search functionality
- Add database integration directly into the Streamlit interface
- Add prediction confidence/probability
- Add visual analytics dashboards
- Add vendor-level risk analytics
- Improve model monitoring
- Add automated invoice data extraction using OCR
- Deploy the Streamlit application to a cloud platform
- Improve inference preprocessing consistency with the training pipeline
- Add automated tests for the complete ML pipeline

---

# 👩‍💻 Author

**Prathiksha R S**

GitHub: https://github.com/PrathikshaRS

---

## ⭐ Project Summary

**Vendor Invoice Processing System** is an end-to-end Machine Learning project that combines **SQL, Python, Machine Learning, model deployment, and Streamlit** to support vendor invoice analysis and financial operations.
