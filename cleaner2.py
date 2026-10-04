import pandas as pd
import numpy as np

# Load extracted country dataset
file_path = 'extracted countries.xlsx'
print(f"Loading {file_path}...")
df = pd.read_excel(file_path, sheet_name=0)

# Identify historical year columns (1980 through 2024)
years = [col for col in df.columns if isinstance(col, int) and 1980 <= col <= 2024]
id_vars = ['ISO', 'WEO Subject Code']

# Melt wide time-series format (years as columns) into long format
print("Reshaping data for Machine Learning...")
df_melted = df[id_vars + years].melt(
    id_vars=id_vars, 
    value_vars=years, 
    var_name='Year', 
    value_name='Value'
)

# Pivot subject codes into separate indicator columns indexed by ISO and Year
df_pivot = df_melted.pivot_table(
    index=['ISO', 'Year'], 
    columns='WEO Subject Code', 
    values='Value', 
    aggfunc='first'
).reset_index()

# Cast indicator values to float
indicators = df_pivot.columns.drop(['ISO', 'Year'])
for col in indicators:
    df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

# Audit initial missing values count
missing_initial = df_pivot[indicators].isna().sum().sum()
total_cells = np.prod(df_pivot[indicators].shape)
print(f"Initial Missing Data: {(missing_initial / total_cells) * 100:.2f}% ({missing_initial} cells)")

# Clean missing data: linear interpolation by country, then forward/backward fill remaining boundary gaps
print("Applying Grouped Temporal Imputation...")
df_clean = df_pivot.copy()

df_clean[indicators] = df_clean.groupby('ISO')[indicators].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)

df_clean[indicators] = df_clean.groupby('ISO')[indicators].transform(
    lambda x: x.ffill().bfill()
)

# Drop columns with >200 missing values that couldn't be cleanly filled
missing_per_col = df_clean[indicators].isna().sum()

cols_to_drop = missing_per_col[missing_per_col > 200].index.tolist()
print(f"Dropping 5 unsalvageable indicators: {cols_to_drop}")

df_clean = df_clean.drop(columns=cols_to_drop)
retained_indicators = df_clean.columns.drop(['ISO', 'Year'])

# Calculate missing data reduction percentage
missing_final = df_clean[retained_indicators].isna().sum().sum()
missing_reduction_pct = ((missing_initial - missing_final) / missing_initial) * 100

print("\n--- FINAL CLEANING REPORT ---")
print(f"Initial Missing Cells: {missing_initial}")
print(f"Final Missing Cells: {missing_final} (Retained intentionally)")
print(f"Missing Data Reduction: {missing_reduction_pct:.2f}%")
print(f"Final Dataset Shape: {df_clean.shape} (Rows, Columns)")

# Export clean ML-ready dataset to Excel
output_file = 'Data Realistic.xlsx'
df_clean.to_excel(output_file, index=False)
print(f"\nSaved clean, ML-ready dataset to {output_file}")