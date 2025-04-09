import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Loading dataset
file_path = "thesis_dataset_26022025_final.csv"

# Read the data
df = pd.read_csv(file_path)

# Show basic info
print("Dataset Shape:", df.shape)
print("\nMissing Value Summary:")
print(df.isnull().sum())

# Visualize missing data matrix
msno.matrix(df)
plt.title("Missing Data Matrix")
plt.tight_layout()
plt.show()
