import pandas as pd
import zipfile
import requests
import os

def check_available_years():
    """Check what years are actually available in the ABS dataset"""
    print("=== CHECKING AVAILABLE YEARS IN ABS DATASET ===")
    
    # Download the dataset
    url = "https://aueprod01ckanstg.blob.core.windows.net/public-catalogue/public/46d46443-58ba-42b4-bbeb-45c021b8257a/exports.csv.zip"
    
    print("Downloading dataset...")
    response = requests.get(url, timeout=60, verify=False)
    response.raise_for_status()
    
    # Save ZIP file
    with open("temp_exports.zip", "wb") as f:
        f.write(response.content)
    
    print("Extracting and checking years...")
    
    with zipfile.ZipFile("temp_exports.zip", 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        csv_file = csv_files[0]
        
        # Read first 100k rows to check years
        with zip_ref.open(csv_file) as f:
            sample_df = pd.read_csv(f, nrows=100000)
            
            # Extract years
            sample_df['year'] = sample_df['month'].astype(str).str.extract(r'(\d{4})')
            
            print(f"Sample size: {len(sample_df):,} records")
            print(f"Available years: {sorted(sample_df['year'].unique())}")
            print(f"Year distribution:")
            year_counts = sample_df['year'].value_counts().sort_index()
            for year, count in year_counts.items():
                print(f"  {year}: {count:,} records")
            
            print(f"\nSample months:")
            month_counts = sample_df['month'].value_counts().head(10)
            for month, count in month_counts.items():
                print(f"  {month}: {count:,} records")
    
    # Clean up
    os.remove("temp_exports.zip")
    print("\nDone!")

if __name__ == "__main__":
    check_available_years()

