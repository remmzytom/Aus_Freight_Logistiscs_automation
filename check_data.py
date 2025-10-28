import pandas as pd

# Check 2024-2025 data
print("=== CHECKING 2024-2025 DATA ===")
df_2024_2025 = pd.read_csv('data/exports_2024_2025.csv')
print(f"Records: {len(df_2024_2025):,}")
print(f"Years: {sorted(df_2024_2025['year'].unique())}")
print(f"Date range: {df_2024_2025['month'].min()} to {df_2024_2025['month'].max()}")

print("\n=== CHECKING CLEANED DATA ===")
df_cleaned = pd.read_csv('data/exports_cleaned.csv')
print(f"Records: {len(df_cleaned):,}")
print(f"Years: {sorted(df_cleaned['year'].unique())}")
print(f"Date range: {df_cleaned['month'].min()} to {df_cleaned['month'].max()}")

print("\n=== SAMPLE DATA ===")
print("2024-2025 sample:")
print(df_2024_2025[['month', 'year', 'country_of_destination', 'value_fob_aud']].head())
print("\nCleaned sample:")
print(df_cleaned[['month', 'year', 'country_of_destination', 'value_fob_aud']].head())

