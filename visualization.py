import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set style and font sizes
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# Load datasets
masked_df = pd.read_csv("data/masked_dataset.csv")
imputed_df = pd.read_csv("data/imputed_dataset.csv")
truth_df = pd.read_csv("data/ground_truth_dataset.csv")

columns_to_evaluate = [
    'Is this girl eligible for enrollment',
    'Enrol in DREAMS',
    'AGYW Age',
    'consent_obtained',
    'AGYW Agrees to Participate?'
]

# Output directory
os.makedirs("outputs/figures", exist_ok=True)

# Calculate error metrics
mae_vals, rmse_vals = [], []
for col in columns_to_evaluate:
    y_true = truth_df[col]
    y_pred = imputed_df[col]
    mask = y_true.notnull()
    mae_vals.append(mean_absolute_error(y_true[mask], y_pred[mask]))
    rmse_vals.append(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))

error_df = pd.DataFrame({
    'Column': columns_to_evaluate,
    'MAE': mae_vals,
    'RMSE': rmse_vals
})

# --- MAE Plot ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=error_df, x='Column', y='MAE', palette="Blues_d")
plt.title("Mean Absolute Error by Column", fontsize=14)
plt.ylabel("MAE", fontsize=12)
plt.xlabel("Column", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", 
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("outputs/figures/mae_barplot.png", dpi=300)
plt.clf()

# --- RMSE Plot ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=error_df, x='Column', y='RMSE', palette="Greens_d")
plt.title("Root Mean Squared Error by Column", fontsize=14)
plt.ylabel("RMSE", fontsize=12)
plt.xlabel("Column", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", 
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("outputs/figures/rmse_barplot.png", dpi=300)
plt.clf()

# --- Missing Value Count Before Imputation ---
missing_counts_before = masked_df[columns_to_evaluate].isnull().sum()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=missing_counts_before.index, y=missing_counts_before.values, palette="Reds")
plt.title("Missing Values Before Imputation", fontsize=14)
plt.ylabel("Missing Value Count", fontsize=12)
plt.xlabel("Column", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)

for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2, p.get_height() + 1000),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("outputs/figures/missingness_before.png", dpi=300)
plt.clf()

# --- Missing Value Count After Imputation ---
missing_counts_after = imputed_df[columns_to_evaluate].isnull().sum()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=missing_counts_after.index, y=missing_counts_after.values, palette="pastel")
plt.title("Missing Values After Imputation", fontsize=14)
plt.ylabel("Missing Value Count", fontsize=12)
plt.xlabel("Column", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)

for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2, p.get_height() + 100),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("outputs/figures/missingness_after.png", dpi=300)
plt.clf()

print("Updated visualizations saved to outputs/figures/")
