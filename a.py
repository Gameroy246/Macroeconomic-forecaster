import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read cleaned macroeconomic dataset
file_path = 'Data Realistic.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Pull numeric indicators, excluding the Year column
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Year' in numeric_cols:
    numeric_cols.remove('Year')

# Fill any residual missing values with column means
X = df[numeric_cols].fillna(df[numeric_cols].mean())

# Scale features so variables with larger magnitudes don't skew PCA loadings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA targeting 85% explained cumulative variance
pca = PCA(n_components=0.85)
pca.fit(X_scaled)

# Loadings dataframe (Principal Components x Original Indicators)
loadings = pd.DataFrame(
    pca.components_,
    columns=numeric_cols,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

# Print top 3 driving features per Principal Component for interpretation
for pc in loadings.index:
    top_features = loadings.loc[pc].abs().sort_values(ascending=False).head(3)
    print(f"{pc} Top Driving Parameters: {list(top_features.index)}")