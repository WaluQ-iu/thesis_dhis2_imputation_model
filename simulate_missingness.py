import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv('data/cleaned_dataset.csv')

# Define the proportion of known (complete) data to artificially mask
mask_fraction = 0.10  # 10%

# Select columns for simulation (excluding identifiers)
columns_to_mask = [
    'Is this girl eligible for enrollment',
    'Enrol in DREAMS',
    'AGYW Date of Birth',
    'AGYW Age',
    'AGYW Sex',
    'consent_obtained',
    'AGYW Agrees to Participate?',
    'AGYW District/Town of Birth'
]

# Store a copy of the original values for validation
df_truth = df.copy()

# Create mask on non-null values only
mask = df[columns_to_mask].notnull()
random_mask = mask & (np.random.rand(*mask.shape) < mask_fraction)

# Apply masking (simulate missing values)
df_masked = df.copy()
df_masked[columns_to_mask] = df_masked[columns_to_mask].mask(random_mask)

# Save masked dataset and the ground truth
df_masked.to_csv('data/masked_dataset.csv', index=False)
df_truth.to_csv('data/ground_truth_dataset.csv', index=False)

print(f"Simulated missingness on {mask_fraction*100}% of selected columns.")
print("Masked dataset saved as 'data/masked_dataset.csv'")
print("Ground truth saved as 'data/ground_truth_dataset.csv'")

