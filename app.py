import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Machine Learning app using Credit Card Applications dataset")

# --------------------------------------------------
# Load data safely
# --------------------------------------------------
@st.cache_data
def load_data():
    file_path = "Credit_Card_Applications.csv"

    if not os.path.exists(file_path):
        st.error("❌ Dataset file not found. Make sure Credit_Card_Applications.csv is in the GitHub repo.")
        st.stop()

    return pd.read_csv(file_path)

df = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "EDA", "Model Training", "Prediction"]
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if menu == "Overview":
    st.header("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    Predict fraudulent / risky credit card applications using Machine Learning.

    **Target Column:**  
    `Class` (1 = Fraud / Risky, 0 = Safe)

    **Algorithm Used:**  
    - Random Forest  
    - SMOTE for class imbalance
    """)

    col1, col2 = st.columns(2)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Fraud Cases", int(df["Class"].sum()))

    st.dataframe(df.head())

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Class Distribution")
    st.bar_chart(df["Class"].value_counts())

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

# --------------------------------------------------
# Model Training
# --------------------------------------------------
elif menu == "Model Training":
    st.header("🤖 Model Training")

    X = df.drop(columns=["Class", "CustomerID"])
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success("✅ Model trained and saved successfully")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
elif menu == "Prediction":
    st.header("🔍 Credit Card Application Prediction")

    if not os.path.exists("model.pkl"):
        st.warning("⚠️ Train the model first from 'Model Training' tab.")
        st.stop()

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    input_data = {}

    cols = df.drop(columns=["Class", "CustomerID"]).columns

    col1, col2, col3 = st.columns(3)
    for i, col in enumerate(cols):
        if i % 3 == 0:
            input_data[col] = col1.number_input(col, float(df[col].min()))
        elif i % 3 == 1:
            input_data[col] = col2.number_input(col, float(df[col].min()))
        else:
            input_data[col] = col3.number_input(col, float(df[col].min()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        if pred == 1:
            st.error("🚨 High Risk / Fraudulent Application")
        else:
            st.success("✅ Safe Application")
