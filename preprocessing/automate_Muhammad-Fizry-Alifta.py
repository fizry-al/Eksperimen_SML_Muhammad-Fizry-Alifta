import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data():
    input_path = "../loan_approval_dataset_raw.csv"
    output_path = "loan_approval_preprocessed.csv"

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    le = LabelEncoder()
    for col in ['education', 'self_employed', 'loan_status']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    df.to_csv(output_path, index=False)
    print("Dataset preprocessing berhasil dibuat.")

if __name__ == "__main__":
    preprocess_data()

