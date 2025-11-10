import pandas as pd
from synthetic_distribution_imputer import SyntheticDistributionImputer

# Load your 1 lakh row dataset
df = pd.read_csv('raw_data/chromatography_final_merged_data.csv')

# Path is set according to where the project is running in the file system: where you're running this script!

# Initialize the synthetic imputer
imputer = SyntheticDistributionImputer(
    method='kde',           # Use KDE for synthetic values
    random_state=42,        # For reproducibility
    bandwidth='scott',      # Automatic bandwidth selection
    bounds_buffer=0.05      # Allow 5% extrapolation
)

# Impute all columns OR specific columns
df_imputed = imputer.fit_transform(df, columns=['peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 'peak_width_10'])
# OR: df_imputed = imputer.fit_transform(df, columns=['col1', 'col2'])

print(f"Columns in original data: {df.columns}")
print(f"Columns in imputed data: {df_imputed.columns}")

# Get detailed statistics
report = imputer.get_imputation_report()
print(report)

# Verify distribution preservation
validation = imputer.statistical_validation(df, df_imputed)
print(validation)


#Exporting imputed data to a new csv file
df_imputed.to_csv('synthetic_data/synthetic_chromatography_final_merged_data.csv', index=False)

# Visualize to confirm
#imputer.visualize_distribution(df, df_imputed, columns=['retention_time', 'peak_width_5'])