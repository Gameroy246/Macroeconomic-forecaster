import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read macroeconomic dataset
file_path = 'Data Realistic.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Isolate numeric feature columns, removing 'Year'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Year' in numeric_cols:
    numeric_cols.remove('Year')

# Mean impute any residual NaNs
X = df[numeric_cols].copy().fillna(df[numeric_cols].mean())

# Scale features before PCA
X_scaled = StandardScaler().fit_transform(X)

# Fit unconstrained PCA to evaluate cumulative variance spectrum
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Determine component count needed for 85% variance threshold
target_variance = 0.85
n_components_optimal = np.argmax(cumulative_variance >= target_variance) + 1

# Reduce feature dimensions down to target variance threshold
pca = PCA(n_components=target_variance)
X_reduced = pca.fit_transform(X_scaled)

# Reconstruct DataFrame with metadata identifiers (ISO, Year)
reduced_columns = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
reduced_df = pd.DataFrame(X_reduced, columns=reduced_columns)
reduced_df['ISO'] = df['ISO'].values
reduced_df['Year'] = df['Year'].values

# Export reduced dataset to Excel
output_file = 'Dimensionality_Reduction_Output.xlsx'
reduced_df.to_excel(output_file, index=False)

print(f"Original Feature Width: {X.shape[1]}")
print(f"Reduced Component Width: {n_components_optimal}")
print(f"Cumulative Variance Explained: {cumulative_variance[n_components_optimal-1] * 100:.2f}%")
print(f"New dataset successfully exported to '{output_file}'")