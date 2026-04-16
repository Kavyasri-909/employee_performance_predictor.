import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

st.title("👨‍💼 Employee Performance Predictor Dashboard")

# -------------------------------
# Generate Data
# -------------------------------
@st.cache_data
def generate_data(n=500):
    np.random.seed(42)

    df = pd.DataFrame({
        "age": np.random.randint(22, 60, n),
        "experience": np.random.randint(1, 20, n),
        "salary": np.random.randint(20000, 150000, n),
        "training_hours": np.random.randint(0, 100, n),
        "projects": np.random.randint(1, 10, n),
        "attendance": np.random.uniform(0.5, 1.0, n)
    })

    def label(row):
        score = (row['experience'] * 0.3 +
                 row['training_hours'] * 0.2 +
                 row['projects'] * 0.3 +
                 row['attendance'] * 10)

        if score > 20:
            return "High"
        elif score > 12:
            return "Medium"
        else:
            return "Low"

    df["performance"] = df.apply(label, axis=1)
    return df

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("performance", axis=1)
    y = df["performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Load Data & Train
df = generate_data()
model, X_test, y_test = train_model(df)

# -------------------------------
# Layout (Two Columns)
# -------------------------------
col1, col2 = st.columns([1, 2])

# -------------------------------
# LEFT SIDE (INPUTS)
# -------------------------------
with col1:
    st.subheader("🧾 Employee Input")

    age = st.slider("Age", 22, 60, 30)
    experience = st.slider("Experience", 1, 20, 5)
    salary = st.slider("Salary", 20000, 150000, 50000)
    training_hours = st.slider("Training Hours", 0, 100, 20)
    projects = st.slider("Projects", 1, 10, 3)
    attendance = st.slider("Attendance", 0.5, 1.0, 0.8)

    input_data = pd.DataFrame([{
        "age": age,
        "experience": experience,
        "salary": salary,
        "training_hours": training_hours,
        "projects": projects,
        "attendance": attendance
    }])

    predict_btn = st.button("🔮 Predict")

# -------------------------------
# RIGHT SIDE (OUTPUTS)
# -------------------------------
with col2:

    # Prediction
    st.subheader("🎯 Prediction")

    if predict_btn:
        prediction = model.predict(input_data)[0]

        if prediction == "High":
            st.success(f"Performance: {prediction} 🚀")
        elif prediction == "Medium":
            st.warning(f"Performance: {prediction}")
        else:
            st.error(f"Performance: {prediction}")

    # Model Performance Table
    st.subheader("📈 Model Performance")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.dataframe(pd.DataFrame(report).transpose(), height=250)

    # Smaller Graph
    st.subheader("📊 Distribution")

    fig, ax = plt.subplots(figsize=(4, 3))  # 👈 reduced size

    df["performance"].value_counts().plot(kind="bar", ax=ax)

    ax.set_title("Performance Distribution", fontsize=10)
    ax.set_xlabel("Performance", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    st.pyplot(fig, use_container_width=False)

# -------------------------------
# Dataset Preview (Collapsed)
# -------------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(df.head())