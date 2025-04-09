import pandas as pd

# Load dataset
df = pd.read_csv('data/thesis_dataset_26022025_final.csv')

# Drop column with >80% missing data
df.drop(columns=['Screened By-New'], inplace=True)

# Convert date columns
date_cols = [
    'AGYW Date of Birth', 'Date of First Contact', 'enrollmentdate',
    'incidentdate', 'executiondate', 'duedate', 'completeddate'
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce', dayfirst=True)

# Convert selected columns to category
cat_cols = ['AGYW Age Group', 'AGYW Age group', 'AGYW Sex', 'consent_obtained']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Save cleaned dataset
df.to_csv('data/cleaned_dataset.csv', index=False)


print("\n Cleaning completed. Cleaned file saved as 'data/cleaned_dataset.csv'")
print("Remaining Columns:", df.columns.tolist())
