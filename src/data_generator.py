import pandas as pd
import numpy as np

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


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/employee_data.csv", index=False)
    print("Dataset created!")