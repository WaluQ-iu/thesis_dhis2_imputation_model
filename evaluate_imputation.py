import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# === Load the imputed and ground truth data ===
imputed_df = pd.read_csv('data/imputed_only.csv')
truth_df = pd.read_csv('data/truth_only.csv')

# === Initialize results dictionary ===
results = {}

# === Evaluate column-by-column ===
for column in truth_df.columns:
    y_true = truth_df[column]
    y_pred = imputed_df[column]

    # Drop NaNs from ground truth
    mask = y_true.notnull()
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

    results[column] = {
        'MAE': mae,
        'RMSE': rmse,
    }

# === Display the results ===
print("📊 Imputation Evaluation Metrics:")
print("---------------------------------")
for col, metrics in results.items():
    print(f"{col}")
    print(f"  ✅ MAE : {metrics['MAE']:.4f}")
    print(f"  ✅ RMSE: {metrics['RMSE']:.4f}")
    print("---------------------------------")

# === Optional: Save results to CSV ===
results_df = pd.DataFrame(results).T
results_df.to_csv('outputs/imputation_evaluation_metrics.csv')

print("📁 Saved evaluation metrics to 'outputs/imputation_evaluation_metrics.csv'")
