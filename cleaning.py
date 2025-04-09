import pandas as pd

# Load the dataset
df = pd.read_csv('data/thesis_dataset_26022025_final.csv')

print("\n--- First 5 Rows ---")
print(df.head())

# Show column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Show basic info about the dataset
print("\n--- Dataset Info ---")
print(df.info())

# Check the percentage of missing values per column
print("\n--- Missing Data (%) by Column ---")
missing_percent = df.isnull().mean() * 100
print(missing_percent.sort_values(ascending=False))
