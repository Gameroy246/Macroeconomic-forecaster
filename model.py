import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

print("Loading data and training models...")

# Read dataset and clean missing values
df = pd.read_excel('Data Realistic.xlsx')

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

features = ['GGXCNL_NGDP', 'GGXWDG_NGDP', 'PCPIPCH', 'BCA_NGDPD', 'NGDP_D']
target = 'NGDP_RPCH'

X = df[features]
y = df[target]

# Scale features for L2 Ridge regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit forecasting algorithms
ridge = Ridge(alpha=100.0).fit(X_scaled, y)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42).fit(X, y)

# Save model artifacts to disk
joblib.dump(ridge, 'ridge_model.pkl')
joblib.dump(gbr, 'gbr_model.pkl')
joblib.dump(scaler, 'scaler_artifact.pkl')
print("Model artifacts successfully saved to disk.\n")

# Filter for Australia (ISO: AUS) with fallback check
if 'AUS' in df['ISO'].values:
    country_df = df[df['ISO'] == 'AUS'].sort_values('Year')
elif 'Australia' in df['ISO'].values:
    country_df = df[df['ISO'] == 'Australia'].sort_values('Year')
else:
    country_iso = df['ISO'].unique()[0]
    print(f"Warning: 'AUS' not found. Defaulting to {country_iso}.")
    country_df = df[df['ISO'] == country_iso].sort_values('Year')

last_known_year = int(country_df['Year'].max())
target_forecast_year = 2030

# Extrapolate future predictor features using 5-year rolling trend drift with 30% annual dampening
def project_future_features(country_data, feat_cols, start_yr, end_yr):
    recent_data = country_data.sort_values('Year').tail(5)
    trend_drift = (recent_data[feat_cols].iloc[-1] - recent_data[feat_cols].iloc[0]) / 4.0
    
    future_rows = []
    current_vals = recent_data[feat_cols].iloc[-1].copy()
    
    for yr in range(start_yr + 1, end_yr + 1):
        trend_drift *= 0.70 # Dampen trend by 30% per year to prevent exponential explosion
        current_vals += trend_drift
        
        row = {'Year': yr, 'ISO': country_data['ISO'].iloc[0]}
        row.update(current_vals.to_dict())
        future_rows.append(row)
        
    return pd.DataFrame(future_rows)

future_features_df = project_future_features(country_df, features, last_known_year, target_forecast_year)

# Predict future GDP growth using both trained models
X_future = future_features_df[features]
X_future_scaled = scaler.transform(X_future)

future_features_df['Ridge_Predicted_GDP (%)'] = ridge.predict(X_future_scaled)
future_features_df['GBR_Predicted_GDP (%)'] = gbr.predict(X_future)

# Format table for output display
output_df = future_features_df[['Year', 'Ridge_Predicted_GDP (%)', 'GBR_Predicted_GDP (%)']].copy()
output_df['Ridge_Predicted_GDP (%)'] = output_df['Ridge_Predicted_GDP (%)'].apply(lambda x: f"{x:+.3f}%")
output_df['GBR_Predicted_GDP (%)'] = output_df['GBR_Predicted_GDP (%)'].apply(lambda x: f"{x:+.3f}%")

print("=========================================================")
print(f"  FUTURE REAL GDP GROWTH FORECAST: AUSTRALIA (AUS) ")
print(f"  Projection Horizon: {last_known_year + 1} to {target_forecast_year} ")
print("=========================================================")
print(output_df.to_string(index=False))
print("=========================================================")