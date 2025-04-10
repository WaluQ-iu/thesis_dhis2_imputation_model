import pandas as pd
from scipy.stats import skew, kurtosis, mode
import numpy as np

def describe_dataset(file_path):
    # Load data
    df = pd.read_csv(file_path)

    print("\n Basic Dataset Info")
    print(df.info())

    print("\n Summary Statistics (Numerical Columns)")
    print(df.describe().T)

    print("\n Skewness and Kurtosis")
    num_df = df.select_dtypes(include='number')
    skew_kurt = pd.DataFrame({
        'skewness': num_df.apply(skew, nan_policy='omit'),
        'kurtosis': num_df.apply(kurtosis, nan_policy='omit')
    })
    print(skew_kurt)

    print("\n Missing Values (Count and Percentage)")
    missing = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percent': df.isnull().mean() * 100
    })
    print(missing[missing['missing_count'] > 0])

    print("\n Mode of Each Column")
    modes = df.mode(numeric_only=False).iloc[0]
    print(modes)

    print("\n Data Types and Unique Values")
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'n_unique': df.nunique()
    })
    print(summary)

    print("\n Top 5 Most Frequent Values (Categorical Columns)")
    cat_df = df.select_dtypes(include='object')
    for col in cat_df.columns:
        print(f"\n {col}")
        print(df[col].value_counts().head())

if __name__ == "__main__":
    file_path = "data/thesis_dataset_26022025_final.csv"
    describe_dataset(file_path)
