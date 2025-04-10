import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Columns to impute and evaluate
columns = [
    'Is this girl eligible for enrollment',
    'Enrol in DREAMS',
    'AGYW Age',
    'consent_obtained',
    'AGYW Agrees to Participate?'
]

# Step 1: Load original masked and ground truth datasets
masked = pd.read_csv("data/masked_dataset.csv")[columns]
truth = pd.read_csv("data/truth_only.csv")[columns]

# Step 2: Create cleaned version of truth data with no missing values
truth_clean = truth.dropna(subset=columns)
truth_clean.to_csv("data/truth_eval_clean.csv", index=False)

results = []

# Step 3: Mean Imputation
mean_imputer = SimpleImputer(strategy='mean')
mean_imputed = mean_imputer.fit_transform(masked)
mean_df = pd.DataFrame(mean_imputed, columns=columns)
mean_df.to_csv("data/imputed_mean.csv", index=False)

for col in columns:
    mae = mean_absolute_error(truth_clean[col], mean_df.loc[truth_clean.index, col])
    rmse = mean_squared_error(truth_clean[col], mean_df.loc[truth_clean.index, col]) ** 0.5
    results.append({
        'Method': 'Mean',
        'Variable': col,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    })

# Step 4: MICE Imputation using Bayesian Ridge
mice_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
mice_imputed = mice_imputer.fit_transform(masked)
mice_df = pd.DataFrame(mice_imputed, columns=columns)
mice_df.to_csv("data/imputed_mice.csv", index=False)

for col in columns:
    mae = mean_absolute_error(truth_clean[col], mice_df.loc[truth_clean.index, col])
    rmse = mean_squared_error(truth_clean[col], mean_df.loc[truth_clean.index, col]) ** 0.5
    results.append({
        'Method': 'MICE',
        'Variable': col,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    })

# Step 5: Evaluate Random Forest output if it exists
rf_path = "data/imputed_dataset.csv"
if os.path.exists(rf_path):
    rf_df = pd.read_csv(rf_path)[columns]
    for col in columns:
        mae = mean_absolute_error(truth_clean[col], rf_df.loc[truth_clean.index, col])
        rmse = mean_squared_error(truth_clean[col], mean_df.loc[truth_clean.index, col]) ** 0.5
        results.append({
            'Method': 'Random Forest',
            'Variable': col,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        })

# Step 6: Save and display results
eval_df = pd.DataFrame(results)
eval_df.to_csv("outputs/evaluation_all_methods.csv", index=False)

print("Evaluation complete. Results saved to: outputs/evaluation_all_methods.csv")
print(eval_df)
