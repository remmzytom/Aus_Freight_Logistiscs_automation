import pandas as pd
import zipfile
import os
import requests
# Removed BytesIO import - using file-based approach instead
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_2024_2025():
    """
    Extract 2024 and 2025 records - perfect balance for stakeholder analysis
    """
    try:
        # Direct download URL for complete dataset
        url = "https://aueprod01ckanstg.blob.core.windows.net/public-catalogue/public/46d46443-58ba-42b4-bbeb-45c021b8257a/exports.csv.zip"
        # No need to create old directory - data goes to data/ folder
        
        # Download complete ZIP file with streaming to avoid memory issues
        logger.info("Downloading ABS export dataset...")
        response = requests.get(url, timeout=60, verify=False, stream=True)
        response.raise_for_status()
        
        # Save ZIP file with streaming to avoid memory issues
        zip_path = "temp_exports.zip"
        logger.info("Saving ZIP file with streaming...")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                if chunk:
                    f.write(chunk)
        
        logger.info("Extracting and loading dataset in chunks...")
        with zipfile.ZipFile(zip_path) as zip_ref:
            # Find CSV file in ZIP
            csv_files = [name for name in zip_ref.namelist() if name.endswith('.csv')]
            csv_file = csv_files[0]
            
            # Read CSV in chunks and filter for 2024-2025 only
            with zip_ref.open(csv_file) as f:
                logger.info("Loading dataset in chunks and filtering for 2024-2025...")
                chunk_size = 10000  # 10k rows at a time (much smaller for memory)
                filtered_chunks = []
                total_processed = 0
                
                for chunk in pd.read_csv(f, chunksize=chunk_size):
                    total_processed += len(chunk)
                    
                    # Extract year and filter immediately
                    chunk['year'] = chunk['month'].astype(str).str.extract(r'(\d{4})')
                    chunk_2024_2025 = chunk[chunk['year'].isin(['2024', '2025'])]
                    
                    if len(chunk_2024_2025) > 0:
                        filtered_chunks.append(chunk_2024_2025)
                        logger.info(f"Found {len(chunk_2024_2025)} records in chunk (processed {total_processed:,} total)")
                    
                    # Clean up memory aggressively
                    del chunk, chunk_2024_2025
                    import gc
                    gc.collect()  # Force garbage collection
                    
                    if total_processed % 500000 == 0:  # Log every 500k rows
                        logger.info(f"Processed {total_processed:,} rows, found {sum(len(c) for c in filtered_chunks):,} 2024-2025 records...")
                
                if filtered_chunks:
                    logger.info("Combining filtered chunks...")
                    # Process chunks in smaller batches to avoid memory issues
                    batch_size = 2  # Process 2 chunks at a time (very conservative)
                    combined_chunks = []
                    
                    for i in range(0, len(filtered_chunks), batch_size):
                        batch = filtered_chunks[i:i+batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(filtered_chunks)-1)//batch_size + 1}")
                        combined_batch = pd.concat(batch, ignore_index=True)
                        combined_chunks.append(combined_batch)
                        # Clean up memory
                        del batch, combined_batch
                    
                    # Final combination
                    df = pd.concat(combined_chunks, ignore_index=True)
                    logger.info(f"Successfully combined {len(df):,} records")
                else:
                    logger.warning("No 2024-2025 data found")
                    return None
        
        logger.info(f"Loaded {len(df):,} 2024-2025 records from dataset")
        
        # Data is already filtered for 2024-2025
        df_2024_2025 = df.copy()
        
        if len(df_2024_2025) > 0:
            logger.info(f"Found {len(df_2024_2025):,} records for 2024-2025")
            
            # Convert numeric columns
            numeric_columns = ['quantity', 'gross_weight_tonnes', 'value_fob_aud']
            for col in numeric_columns:
                if col in df_2024_2025.columns:
                    df_2024_2025[col] = pd.to_numeric(df_2024_2025[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
            
            # Save 2024-2025 data to data folder (for automation)
            os.makedirs("data", exist_ok=True)
            output_path = "data/exports_2024_2025.csv"
            df_2024_2025.to_csv(output_path, index=False)
            logger.info(f"Saved 2024-2025 data to: {output_path}")
            
            # Show summary
            print(f"\n{'='*60}")
            print(f"2024-2025 EXPORT DATA ANALYSIS")
            print(f"{'='*60}")
            print(f"Total records in dataset: {len(df):,}")
            print(f"2024-2025 records: {len(df_2024_2025):,}")
            
            # Show breakdown by year
            print(f"\nRecords by year:")
            print("-" * 40)
            year_breakdown = df_2024_2025['year'].value_counts().sort_index()
            for year, count in year_breakdown.items():
                print(f"{year}: {count:,} records")
            
            # Show breakdown by month
            print(f"\nRecords by month (2024-2025):")
            print("-" * 40)
            month_breakdown = df_2024_2025['month'].value_counts().sort_index()
            for month, count in month_breakdown.head(15).items():  # Show first 15 months
                print(f"{month}: {count:,} records")
            
            # Year-over-year comparison
            if '2024' in year_breakdown.index and '2025' in year_breakdown.index:
                print(f"\nYear-over-Year Comparison:")
                print("-" * 40)
                records_2024 = year_breakdown['2024']
                records_2025 = year_breakdown['2025']
                growth = ((records_2025 - records_2024) / records_2024) * 100
                print(f"2024: {records_2024:,} export records")
                print(f"2025: {records_2025:,} export records")
                print(f"Growth: {growth:+.1f}%")
            
            # Show sample data
            print(f"\nSample export data (2024-2025):")
            print("-" * 40)
            sample_cols = ['month', 'country_of_destination', 'value_fob_aud', 'gross_weight_tonnes']
            available_cols = [col for col in sample_cols if col in df_2024_2025.columns]
            print(df_2024_2025[available_cols].head(10).to_string(index=False))
            
            # Summary statistics
            print(f"\nSummary Statistics (2024-2025):")
            print("-" * 40)
            numeric_cols_available = [col for col in numeric_columns if col in df_2024_2025.columns]
            if numeric_cols_available:
                print(df_2024_2025[numeric_cols_available].describe())
            
            logger.info("Successfully processed all data, returning result")
            
            # Clean up temporary ZIP file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logger.info("Cleaned up temporary ZIP file")
            
            return df_2024_2025
        
        else:
            logger.warning("No 2024-2025 records found")
            return None
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    print("Starting 2024-2025 Data Extraction...")
    print("=" * 60)
    
    result = extract_2024_2025()
    
    if result is not None:
        print(f"\nSUCCESS! Extracted {len(result):,} records of 2024-2025 data")
        print(f" Data saved to: data/exports_2024_2025.csv")
        print(f" Perfect for stakeholder analysis and year-over-year comparison!")
    else:
        print(f"\n No 2024-2025 data found")

if __name__ == "__main__":
    main()
