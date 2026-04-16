import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 🔹 Generate Synthetic Dataset
# -------------------------------
@st.cache_data
def generate_data(n=500):
    np.random.seed(42)

    data = pd.DataFrame({
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

    data['performance'] = data.apply(label, axis=1)
    return data


# -------------------------------
# 🔹 Train Model
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


# -------------------------------
# 🔹 UI
# -------------------------------
st.set_page_config(page_title="Employee Performance Predictor", layout="centered")

st.title("👨‍💼 Employee Performance Predictor")
st.write("Predict employee performance using Machine Learning")

# Generate data
df = generate_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Train model
model, X_test, y_test = train_model(df)

# -------------------------------
# 🔹 User Input
# -------------------------------
st.subheader("🧾 Enter Employee Details")

age = st.slider("Age", 22, 60, 30)
experience = st.slider("Experience (years)", 1, 20, 5)
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

# -------------------------------
# 🔹 Prediction
# -------------------------------
if st.button("Predict Performance"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Performance: {prediction}")

# -------------------------------
# 🔹 Model Performance
# -------------------------------
st.subheader("📈 Model Performance")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.write(pd.DataFrame(report).transpose())

# -------------------------------
# 🔹 Visualization
# -------------------------------
st.subheader("📊 Performance Distribution")

fig, ax = plt.subplots()
sns.countplot(x="performance", data=df, ax=ax)
st.pyplot(fig)