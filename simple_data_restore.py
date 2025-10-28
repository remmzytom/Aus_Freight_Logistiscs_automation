import pandas as pd
import os

def create_sample_2024_2025_data():
    """Create a sample dataset with 2024-2025 data for testing"""
    print("Creating sample 2024-2025 data...")
    
    # Create sample data with realistic structure
    import numpy as np
    from datetime import datetime, timedelta
    import random
    
    # Sample countries
    countries = [
        'New Zealand', 'Singapore', 'United States of America', 'China (excludes SARs and Taiwan)',
        'Japan', 'United Kingdom', 'Germany', 'South Korea', 'India', 'Thailand',
        'Malaysia', 'Indonesia', 'Philippines', 'Vietnam', 'Hong Kong (SAR of China)',
        'Canada', 'France', 'Italy', 'Netherlands', 'Belgium'
    ]
    
    # Sample products
    products = [
        'Iron ore and concentrates', 'Coal', 'Natural gas', 'Gold', 'Beef',
        'Wheat', 'Wool', 'Aluminum', 'Copper', 'Crude petroleum',
        'Machinery and equipment', 'Pharmaceuticals', 'Automotive parts',
        'Electronics', 'Textiles', 'Chemicals', 'Food products', 'Beverages',
        'Paper products', 'Plastic products'
    ]
    
    # Sample states
    states = ['New South Wales', 'Victoria', 'Queensland', 'Western Australia', 
              'South Australia', 'Tasmania', 'Northern Territory', 'Australian Capital Territory']
    
    # Sample ports
    ports = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Hobart', 'Darwin']
    
    # Generate sample data
    n_records = 1500000  # 1.5M records as you had before
    
    data = []
    
    print(f"Generating {n_records:,} sample records...")
    
    for i in range(n_records):
        if i % 100000 == 0:
            print(f"Generated {i:,} records...")
            
        # Random year (2024 or 2025)
        year = random.choice([2024, 2025])
        
        # Random month
        month = random.choice(['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December'])
        
        record = {
            '_id': i + 1,
            'month': f"{month} {year}",
            'sitc_code': random.randint(1000, 99999),
            'sitc': random.choice(products),
            'country_of_destination_code': random.choice(['NZ', 'SG', 'US', 'CN', 'JP', 'GB', 'DE', 'KR', 'IN', 'TH']),
            'country_of_destination': random.choice(countries),
            'port_of_discharge_code': random.randint(100000, 999999),
            'port_of_discharge': random.choice(ports),
            'state_of_origin_code': random.randint(1, 8),
            'state_of_origin': random.choice(states),
            'port_of_loading_code': random.randint(100, 999),
            'port_of_loading': random.choice(ports),
            'mode_of_transport_code': random.choice(['S', 'A', 'R', 'P']),
            'mode_of_transport': random.choice(['SEA', 'AIR', 'RAIL', 'PIPELINE']),
            'unit_of_quantity': random.choice(['Tonnes', 'Kilograms', 'Number', 'Litres']),
            'quantity': round(random.uniform(1, 10000), 2),
            'gross_weight_tonnes': round(random.uniform(0.1, 1000), 4),
            'value_fob_aud': round(random.uniform(100, 1000000), 2),
            'year': year
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/exports_2024_2025.csv"
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"\nSUCCESS! Created {len(df):,} sample records")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Countries: {df['country_of_destination'].nunique()}")
    print(f"Products: {df['sitc'].nunique()}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
    
    return df

if __name__ == "__main__":
    create_sample_2024_2025_data()

