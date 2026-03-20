# Streamlit Supervised Learning Demo

A simple Streamlit app to load a CSV dataset and train a supervised learning model (classification or regression) using scikit-learn.

## ✅ What it does
- Loads a CSV (from a path or via upload)
- Lets you choose a target column and feature columns
- Auto-detects whether the problem is classification or regression
- Trains a Random Forest model and shows evaluation metrics

## 🚀 Getting started
1. Create a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run one of the Streamlit apps:

```powershell
streamlit run app.py
```

or (Logistic Regression focused):

```powershell
streamlit run app_logistic.py
```

4. In either app UI:
- If you are running on Kaggle, the default path should work (`/kaggle/input/amazon-products-dataset/Amazon-Products.csv`).
- Otherwise, toggle **Upload CSV instead** and upload the `Amazon-Products.csv` file.

## 🧠 Notes
- For small datasets, the default Random Forest settings work well.
- If the dataset has many distinct string values, consider selecting fewer feature columns or filtering in advance.
