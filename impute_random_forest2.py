import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

#Load the masked dataset with simulated missingness
df_masked = pd.read_csv('data/masked_dataset.csv')

#Load the original ground truth dataset for evaluation
df_truth = pd.read_csv('data/ground_truth_dataset.csv')

#Define specific numeric columns to impute
columns_to_impute = [
    'Is this girl eligible for enrollment',
    'Enrol in DREAMS',
    'AGYW Age',
    'consent_obtained',
    'AGYW Agrees to Participate?'
]

#Subset the data for imputation
data_to_impute = df_masked[columns_to_impute]

# Initialize IterativeImputer using RandomForestRegressor
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    max_iter=10,
    random_state=42
)

#Perform the imputation
imputed_values = imputer.fit_transform(data_to_impute)

#Replace the imputed columns in the original dataframe
df_imputed = df_masked.copy()
df_imputed[columns_to_impute] = imputed_values

#Save full imputed dataset
df_imputed.to_csv('data/imputed_dataset.csv', index=False)

#Save only imputed columns and corresponding ground truth for evaluation
df_imputed[columns_to_impute].to_csv('data/imputed_only.csv', index=False)
df_truth[columns_to_impute].to_csv('data/truth_only.csv', index=False)

print("Complete Succesfully")
print("Saved files:")
print(" - data/imputed_dataset.csv")
print(" - data/imputed_only.csv")
print(" - data/truth_only.csv")
