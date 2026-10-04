import pandas as pd

# List of target country ISO codes to filter from the main IMF WEO dataset
iso_list = [
    'CAN', 'NOR', 'FIN', 'GBR', 'FRA', 'ESP', 'JPN', 'ISL', 'BEL', 'DNK', 
    'NLD', 'SWE', 'NZL', 'ITA', 'CHE', 'PRT', 'GRC', 'AUT', 'AUS', 'DEU', 
    'PRY', 'JOR', 'TWN', 'IRL', 'COL', 'KOR', 'HKG', 'CYP', 'CHL', 'MEX', 
    'ETH', 'MAR', 'SYC', 'LUX', 'GNQ', 'USA', 'SWZ', 'CRI', 'BOL', 'BRB', 
    'LSO', 'PHL', 'PAN', 'MYS', 'DZA', 'KEN'
]

# Load raw dataset
file_path = 'uncleaned.xlsx'
print(f"Loading data from {file_path}...")
df = pd.read_excel(file_path, sheet_name=0)

# Filter dataset for target ISO codes
print("Filtering data...")
filtered_df = df[df['ISO'].isin(iso_list)]

# Save extracted subset to Excel
output_filename = 'extracted countries.xlsx'
filtered_df.to_excel(output_filename, index=False)

# Log extraction summary statistics
print(f"Extraction complete! Filtered data saved to {output_filename}")
print(f"Original row count: {len(df)}")
print(f"Filtered row count: {len(filtered_df)}")
print(f"Unique countries extracted: {filtered_df['ISO'].nunique()}")