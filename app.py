import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Page setup and custom styling for KPI summary cards
st.set_page_config(page_title="Future Macroeconomic Forecaster", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .kpi-box { background-color: #f8f9fa; border-left: 5px solid #1f4e78; padding: 15px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .kpi-title { font-size: 13px; color: #7f8c8d; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;}
    .kpi-value { font-size: 26px; color: #2c3e50; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Cache data loading so Streamlit doesn't reload the Excel file on every click
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('Data Realistic.xlsx')
    except Exception:
        st.error("Error: Could not find 'Data Realistic.xlsx'.")
        st.stop()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    return df

# Train forecasting algorithms using data up to the chosen cutoff year
@st.cache_resource
def train_forecasting_models(df, cutoff_year):
    features = ['GGXCNL_NGDP', 'GGXWDG_NGDP', 'PCPIPCH', 'BCA_NGDPD', 'NGDP_D']
    train_df = df[df['Year'] <= cutoff_year]
    X_train, y_train = train_df[features], train_df['NGDP_RPCH']
    
    # Scale features for L2 Ridge regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    ridge_model = Ridge(alpha=100.0).fit(X_train_scaled, y_train)
    gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42).fit(X_train, y_train)
    
    return ridge_model, gbr_model, scaler, features

# Project future predictor features using 5-year rolling trend drift with 30% annual dampening
def project_future_features(country_df, features, last_actual_year, forecast_horizon_year):
    if forecast_horizon_year <= last_actual_year: return pd.DataFrame()
    recent_data = country_df.sort_values('Year').tail(5)
    trend_drift = (recent_data[features].iloc[-1] - recent_data[features].iloc[0]) / 4.0
    
    future_rows = []
    current_vals = recent_data[features].iloc[-1].copy()
    for yr in range(last_actual_year + 1, forecast_horizon_year + 1):
        trend_drift *= 0.70 # Dampen trend by 30% per year to prevent unrealistic exponential jumps
        current_vals += trend_drift
        row = {'Year': yr, 'ISO': country_df['ISO'].iloc[0]}
        row.update(current_vals.to_dict())
        future_rows.append(row)
    return pd.DataFrame(future_rows)

df_clean = load_data()
max_year_in_data = int(df_clean['Year'].max())
min_year_in_data = int(df_clean['Year'].min())

# Sidebar controls
st.sidebar.title("Forecast Controls")
selected_country = st.sidebar.selectbox("Select a Country:", sorted(df_clean['ISO'].unique()))
model_choice = st.sidebar.radio(" Predictive Algorithm:", ("Gradient Boosting (Non-Linear)", "Ridge Regression (Linear)"))

cutoff_year = st.sidebar.slider("Model Training Cutoff Year:", min_value=min_year_in_data + 10, max_value=max_year_in_data - 1, value=max_year_in_data - 3)
forecast_horizon = st.sidebar.slider("Future Forecast Horizon:", min_value=max_year_in_data, max_value=max_year_in_data + 15, value=max_year_in_data + 5)

# Run model predictions
ridge_model, gbr_model, scaler, feature_cols = train_forecasting_models(df_clean, cutoff_year)
country_df = df_clean[df_clean['ISO'] == selected_country].copy().sort_values('Year')
future_features_df = project_future_features(country_df, feature_cols, max_year_in_data, forecast_horizon)

full_country_df = pd.concat([country_df, future_features_df], ignore_index=True) if not future_features_df.empty else country_df.copy()
X_full, X_full_scaled = full_country_df[feature_cols], scaler.transform(full_country_df[feature_cols])

if model_choice == "Gradient Boosting (Non-Linear)":
    full_country_df['ML_Forecast'] = gbr_model.predict(X_full)
else:
    full_country_df['ML_Forecast'] = ridge_model.predict(X_full_scaled)

# Title & KPI summary metrics
st.title("Global Macroeconomic GDP Forecaster")
st.caption(f"Analyzing {selected_country} | Algorithm: {model_choice} | Training Cutoff: {cutoff_year}")

col1, col2, col3, col4 = st.columns(4)
latest_actual = country_df[country_df['Year'] <= max_year_in_data]['NGDP_RPCH'].iloc[-1]
horizon_pred = full_country_df[full_country_df['Year'] == forecast_horizon]['ML_Forecast'].values[0] if forecast_horizon in full_country_df['Year'].values else 0

with col1:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Latest Actual GDP Growth</div><div class='kpi-value'>{latest_actual:.2f}%</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Forecasted GDP ({forecast_horizon})</div><div class='kpi-value'>{horizon_pred:.2f}%</div></div>", unsafe_allow_html=True)
with col3:
    debt_val = country_df['GGXWDG_NGDP'].iloc[-1]
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Latest Gross Debt (% GDP)</div><div class='kpi-value'>{debt_val:.1f}%</div></div>", unsafe_allow_html=True)
with col4:
    inflation_val = country_df['PCPIPCH'].iloc[-1]
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Latest Inflation Rate</div><div class='kpi-value'>{inflation_val:.2f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Main Plotly visualization comparing historical actuals vs model forecast
fig = go.Figure()
hist_df = full_country_df[full_country_df['Year'] <= max_year_in_data]
future_df = full_country_df[full_country_df['Year'] > max_year_in_data]

fig.add_trace(go.Scatter(x=hist_df['Year'], y=hist_df['NGDP_RPCH'], mode='lines+markers', name='Actual Real GDP Growth', line=dict(color='#1f4e78', width=3)))
fig.add_trace(go.Scatter(x=hist_df['Year'], y=hist_df['ML_Forecast'], mode='lines', name='Model In-Sample / Backtest', line=dict(color='#e74c3c', width=2, dash='dash')))

if not future_df.empty:
    bridge_year = [hist_df['Year'].iloc[-1]] + list(future_df['Year'])
    bridge_val = [hist_df['ML_Forecast'].iloc[-1]] + list(future_df['ML_Forecast'])
    fig.add_trace(go.Scatter(x=bridge_year, y=bridge_val, mode='lines+markers', name='Out-of-Sample Future Forecast', line=dict(color='#27ae60', width=3)))

fig.add_vline(x=cutoff_year, line_dash="dash", line_color="orange", annotation_text="Training Cutoff", annotation_position="top left")
fig.add_vline(x=max_year_in_data, line_dash="solid", line_color="gray", annotation_text="Data Boundary", annotation_position="top right")

fig.update_layout(title=f"Macroeconomic GDP Growth Trajectory & Forecast ({selected_country})", xaxis_title="Year", yaxis_title="Real GDP Growth Rate (%)", hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Interactive data table expander
with st.expander("View Underlying Forecast Dataset"):
    num_fmt_cols = ['NGDP_RPCH', 'ML_Forecast'] + feature_cols
    st.dataframe(full_country_df[['Year', 'ISO', 'NGDP_RPCH', 'ML_Forecast'] + feature_cols].style.format("{:.3f}", subset=num_fmt_cols, na_rep="-"))