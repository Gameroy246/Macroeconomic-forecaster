import pandas as pd
import numpy as np

# Load clean macroeconomic dataset
file_path = 'Data Realistic.xlsx'
df_clean = pd.read_excel(file_path, sheet_name=0)

key_vars = ['NGDP_RPCH', 'PCPIPCH', 'GGXWDG_NGDP', 'BCA_NGDPD', 'GGXCNL_NGDP']

# 1. Five-point summary metrics (min, Q1, median, Q3, max)
summary_table = pd.DataFrame({
    'Count': df_clean[key_vars].count(),
    'Min (0%)': df_clean[key_vars].min(),
    'Q1 (25%)': df_clean[key_vars].quantile(0.25),
    'Median (50%)': df_clean[key_vars].quantile(0.50),
    'Q3 (75%)': df_clean[key_vars].quantile(0.75),
    'Max (100%)': df_clean[key_vars].max()
}).round(3)

# 2. Univariate dispersion analysis (mean, std dev, variance, skewness, kurtosis)
univariate_analysis = pd.DataFrame({
    'Mean': df_clean[key_vars].mean(),
    'Std Dev': df_clean[key_vars].std(),
    'Variance': df_clean[key_vars].var(),
    'Skewness': df_clean[key_vars].skew(),
    'Kurtosis': df_clean[key_vars].kurtosis()
}).round(3)

# 3. Pairwise correlation matrix
corr_matrix = df_clean[key_vars].corr().round(3)

# 4. Outlier & anomaly detection using dynamic 1.5 * IQR rule on GDP growth
target = 'NGDP_RPCH'
series = df_clean[target].dropna()
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

# Filter anomalous observations and group by year to spot crisis shocks
anomalies = df_clean[(df_clean[target] < lower_threshold) | (df_clean[target] > upper_threshold)]
macro_shocks = anomalies.groupby('Year').size().reset_index(name='Anomaly_Count')
macro_shocks = macro_shocks.sort_values(by='Anomaly_Count', ascending=False)

# Export all summary sheets to Excel workbook
output_file = 'Complete Analysis.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_table.to_excel(writer, sheet_name='1_Five_Point_Summary')
    univariate_analysis.to_excel(writer, sheet_name='2_Univariate_Analysis')
    corr_matrix.to_excel(writer, sheet_name='3_Multivariate_Correlation')
    macro_shocks.to_excel(writer, sheet_name='4_Detected_Anomalies', index=False)

print(f"Complete Analysis successfully generated and exported to {output_file}")