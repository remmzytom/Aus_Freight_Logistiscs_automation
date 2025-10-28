# ABS International Merchandise Trade Data Extractor

This project queries the Australian Bureau of Statistics (ABS) International Merchandise Trade – Exports dataset via the official Data API, specifically filtering for 2025 records.

## Features

- **API Integration**: Uses the official CKAN Data API from `catalogue.data.infrastructure.gov.au`
- **JSON-based Filtering**: Uses simple JSON filters instead of SQL queries - no SQL knowledge required!
- **Real-time Data**: Queries live data directly from the API without downloading files
- **Data Processing**: Converts specified columns (`quantity`, `gross_weight_tonnes`, `value_fob_aud`) to numeric format
- **Output**: Saves cleaned data as `exports_2025.csv`
- **Comprehensive Logging**: Provides detailed logging and summary statistics

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the data extraction script:
```bash
python abs_extractor.py
```

## Output

The script will:
1. Connect to the ABS Data API
2. Query the dataset schema to understand available fields
3. Use JSON-based filtering: `{"month": "2025"}` to get 2025 records
4. Process the API response and convert to pandas DataFrame
5. Convert specified columns to numeric format
6. Save the cleaned data to `exports_2025.csv`
7. Display the number of rows and first 5 records
8. Show summary statistics for numeric columns

## Files Generated

- `exports_2025.csv` - Processed 2025 data from API (saved in `Aus_fright_logistic` folder)
- Log output with processing details

## API Endpoints Used

- **Data Search**: `https://catalogue.data.infrastructure.gov.au/api/3/action/datastore_search`
- **Resource ID**: `46d46443-58ba-42b4-bbeb-45c021b8257a`
- **Filter Method**: JSON-based filtering (no SQL required!)

## Data Source

Dataset: ABS International Merchandise Trade – Exports
API Base URL: https://catalogue.data.infrastructure.gov.au/api/3/action/
Resource: https://catalogue.data.infrastructure.gov.au/dataset/abs-data-for-international-merchandise-trade/resource/46d46443-58ba-42b4-bbeb-45c021b8257a
