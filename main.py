# main.py

import sys
import os

# 🔥 Fix: Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Now import modules (WITHOUT src.)
from data_generator import generate_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_data

import pandas as pd

# Create required folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("🚀 Starting Employee Performance Predictor...")

# Step 1: Generate Data
df = generate_data()
df.to_csv("data/employee_data.csv", index=False)
print("✅ Dataset generated and saved!")

# Step 2: Preprocess
X, y = preprocess_data(df)
print("✅ Data preprocessing completed!")

# Step 3: Train Model
model, X_test, y_test = train_model(X, y)
print("✅ Model training completed!")

# Step 4: Evaluate Model
evaluate_model(model, X_test, y_test)
print("✅ Model evaluation completed!")

# Step 5: Visualization
visualize_data()
print("✅ Visualization saved!")

print("🎉 Project executed successfully!")