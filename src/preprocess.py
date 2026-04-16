import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.drop_duplicates()
    df = df.dropna()

    X = df.drop("performance", axis=1)
    y = df["performance"]

    return X, y