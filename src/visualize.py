import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data():
    df = pd.read_csv("data/employee_data.csv")

    sns.countplot(x="performance", data=df)
    plt.title("Performance Distribution")

    # Save image
    plt.savefig("outputs/performance_distribution.png")
    plt.close()