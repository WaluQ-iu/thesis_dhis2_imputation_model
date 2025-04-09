import pandas as pd
import numpy as np
import time
from sklearn.experimental import enable_iterative_imputer  # Enables IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Start timing the process
start_time = time.time()

# Load the masked dataset with simulated missing values
df_masked = pd.read_csv('data/masked_dataset.csv')

# Load the original ground truth dataset for evaluation
df_truth = pd.read_csv('data/ground_truth_dataset.csv')

# Define the numeric columns selected for imputation
columns_to_impute = [
    'Is this girl eligible for enrollment',
    'Enrol in DREAMS',
    'AGYW Age',
    'consent_obtained',
    'AGYW Agrees to Participate?'
]

# Display the number of missing values before imputation
print("Missing values BEFORE imputation:")
print(df_masked[columns_to_impute].isnull().sum())

# Subset the data to include only the selected columns
data_to_impute = df_masked[columns_to_impute]

# Configure the IterativeImputer using RandomForestRegressor
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    max_iter=10,
    random_state=42
)

# Perform the imputation
imputed_values = imputer.fit_transform(data_to_impute)

# Insert the imputed values back into the original masked dataset
df_imputed = df_masked.copy()
df_imputed[columns_to_impute] = imputed_values

# Display the number of missing values after imputation
print("\n Missing values AFTER imputation:")
print(df_imputed[columns_to_impute].isnull().sum())

# Save the fully imputed dataset
df_imputed.to_csv('data/imputed_dataset.csv', index=False)

# Save only the imputed columns for evaluation
df_imputed[columns_to_impute].to_csv('data/imputed_only.csv', index=False)
df_truth[columns_to_impute].to_csv('data/truth_only.csv', index=False)

# Report how long the imputation process took
end_time = time.time()
print(f"\n Imputation completed in {end_time - start_time:.2f} seconds.")
print("Files saved:")
print("   • data/imputed_dataset.csv")
print("   • data/imputed_only.csv")
print("   • data/truth_only.csv")
