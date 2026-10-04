import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Load dataset and clean missing values
df = pd.read_excel('Data Realistic.xlsx')

# Mean impute missing numeric values to preserve sample size
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Define target (Real GDP Growth Rate %) and features
# Features: Deficit (% GDP), Debt (% GDP), Inflation (%), Current Account (% GDP), GDP Deflator
target = 'NGDP_RPCH'
features = ['GGXCNL_NGDP', 'GGXWDG_NGDP', 'PCPIPCH', 'BCA_NGDPD', 'NGDP_D']

X = df[features]
y = df[target]

# Scale features for L2 Ridge regression (penalty depends on feature variance scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up 5-fold cross validation with fixed seed for reproducible splits
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = {
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'R2': 'r2'
}

# Model 1: L2 Regularized Ridge Regression
ridge = Ridge(alpha=100.0)
cv_results_ridge = cross_validate(ridge, X_scaled, y, cv=cv_strategy, scoring=scoring_metrics)

# Model 2: Gradient Boosting Regressor (shallow max_depth=4 to prevent overfitting)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
cv_results_gbr = cross_validate(gbr, X, y, cv=cv_strategy, scoring=scoring_metrics)

# Gather fold-by-fold validation metrics
folds_data = []
for i in range(5):
    folds_data.append({
        'Fold Number': f'Fold {i+1}',
        'Ridge RMSE': np.sqrt(-cv_results_ridge['test_MSE'][i]),
        'Ridge MAE': -cv_results_ridge['test_MAE'][i],
        'Ridge R²': cv_results_ridge['test_R2'][i],
        'GBR RMSE': np.sqrt(-cv_results_gbr['test_MSE'][i]),
        'GBR MAE': -cv_results_gbr['test_MAE'][i],
        'GBR R²': cv_results_gbr['test_R2'][i]
    })
df_folds = pd.DataFrame(folds_data)

# Aggregate mean and std across 5 folds
summary_data = [
    {
        'Algorithm': 'Ridge Regression (L2 Linear)',
        'CV Mean RMSE': np.mean([np.sqrt(-x) for x in cv_results_ridge['test_MSE']]),
        'CV Std RMSE (Variance)': np.std([np.sqrt(-x) for x in cv_results_ridge['test_MSE']]),
        'CV Mean MAE': np.mean(-cv_results_ridge['test_MAE']),
        'CV Mean R²': np.mean(cv_results_ridge['test_R2'])
    },
    {
        'Algorithm': 'Gradient Boosting Regressor',
        'CV Mean RMSE': np.mean([np.sqrt(-x) for x in cv_results_gbr['test_MSE']]),
        'CV Std RMSE (Variance)': np.std([np.sqrt(-x) for x in cv_results_gbr['test_MSE']]),
        'CV Mean MAE': np.mean(-cv_results_gbr['test_MAE']),
        'CV Mean R²': np.mean(cv_results_gbr['test_R2'])
    }
]
df_summary = pd.DataFrame(summary_data)

# Document key analytical limitations for decision makers
limitations_data = [
    {
        'Limitation Category': 'Limitation 1: Exogenous Shock Vulnerability (Unobserved Variables)',
        'Detailed Description': 'Both models rely strictly on endogenous fiscal and monetary parameters (Debt, Deficit, Inflation). They inherently fail to capture unobserved exogenous shocks (e.g., geopolitical conflicts, global pandemics like COVID-19, natural disasters) which structurally break statistical economic trends. Without external dummy variables, the CV residual errors widen severely during these black-swan macroeconomic events.'
    },
    {
        'Limitation Category': 'Limitation 2: Time-Series Autocorrelation Ignorance (Leakage Risk)',
        'Detailed Description': 'The current 5-Fold Cross-Validation randomly shuffles the data (Standard KFold). Macroeconomic data inherently violates the i.i.d. assumption because GDP growth is strongly autocorrelated across time. Randomly splitting country-years leaks future economic trajectory information into the training folds, making the Cross-Validation RMSE overly optimistic. A rigorous TimeSeriesSplit (rolling origin forecasting) would yield more realistic out-of-sample error estimates.'
    }
]
df_limitations = pd.DataFrame(limitations_data)

# Export results to Excel
output_excel = 'Model_Evaluation_and_Validation.xlsx'

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='5-Fold CV Summary', index=False)
    df_folds.to_excel(writer, sheet_name='Fold-by-Fold Metrics', index=False)
    df_limitations.to_excel(writer, sheet_name='Model Limitations', index=False)

# Excel styling formatting
wb = openpyxl.load_workbook(output_excel)
header_fill = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
header_font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
border = Border(left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'), 
                top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9'))

for ws in wb.worksheets:
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = border
            if isinstance(cell.value, float):
                cell.number_format = '0.0000'
            if ws.title == 'Model Limitations':
                cell.alignment = Alignment(wrap_text=True, vertical='top')
                
    for col in ws.columns:
        max_length = max(len(str(cell.value or '')) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_length + 2, 85)

wb.save(output_excel)
print(f"Data Analyst Pipeline Executed. Output saved to: {output_excel}")