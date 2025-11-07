# COMPREHENSIVE STREAMLIT DASHBOARD - REUSING ALL YOUR NOTEBOOK CODE
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import gc

# Disable Pillow decompression bomb check
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Set up plotting style (from your notebook)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Page configuration
st.set_page_config(
    page_title="Australian Freight Export Analysis Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Professional Website Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0 3rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card h3 {
        color: #2c3e50;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 3rem 0 2rem 0;
        padding: 1rem 0;
        border-bottom: 4px solid #667eea;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Subsection Headers */
    .subsection-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Info Boxes */
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #2196f3;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 1px solid #4caf50;
        border-radius: 10px;
    }
    
    /* Error Messages */
    .stError {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 1px solid #f44336;
        border-radius: 10px;
    }
    
    /* Loading Spinner */
    .stSpinner {
        color: #667eea;
    }
    
    /* Data Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .hero-title {
            font-size: 2rem;
        }
        .metric-card {
            padding: 1.5rem;
        }
        .section-header {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Beautiful Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Australian Freight Export Analysis</h1>
    <p class="hero-subtitle">Comprehensive Business Intelligence Dashboard for 2024-2025 Export Data</p>
    <p style="font-size: 1rem; opacity: 0.8;">Real-time insights • Strategic Analysis • Performance Metrics</p>
</div>
""", unsafe_allow_html=True)

# Load data function with accurate KPIs and fast dashboard
def ensure_data_file() -> str:
    """Ensure cleaned data file exists and is fresh (auto-refreshes if >30 days old).
    - First checks if cleaned file from notebook exists (preferred)
    - If missing: generate immediately
    - If older than 30 days: regenerate from ABS website
    Returns the relative path to the cleaned CSV.
    """
    import os
    import time
    os.makedirs('data', exist_ok=True)

    cleaned_path = 'data/exports_cleaned.csv'
    
    # Check if cleaned file exists and is fresh (from notebook or previous generation)
    if os.path.exists(cleaned_path):
        # Check file age
        file_age_days = (time.time() - os.path.getmtime(cleaned_path)) / (60 * 60 * 24)
        
        # If file is less than 30 days old, check if it has required columns
        if file_age_days <= 30:
            try:
                import pandas as pd
                sample_df = pd.read_csv(cleaned_path, nrows=1)
                # Check if it has all required columns
                required_cols = ['month_number', 'year', 'value_fob_aud', 'gross_weight_tonnes']
                if all(col in sample_df.columns for col in required_cols):
                    # File exists, is fresh, and has required columns - use it
                    return cleaned_path
                else:
                    # Missing required columns, need to regenerate
                    st.info("Updating data format (adding missing columns)...")
            except Exception:
                # File might be corrupted, will regenerate
                pass
        else:
            # File is older than 30 days
            st.info(f"Data is {int(file_age_days)} days old. Refreshing from ABS website...")
    
    # If we get here, we need to regenerate
    needs_regeneration = True

    # Try to generate raw exports and produce a minimal cleaned file
    extract_func = None
    try:
        import importlib.util, pathlib
        extractor_path = pathlib.Path(__file__).parent / "2024_2025_extractor.py"
        if extractor_path.exists():
            spec = importlib.util.spec_from_file_location("extractor_module", str(extractor_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                extract_func = getattr(module, "extract_2024_2025", None)
    except Exception:
        extract_func = None

    try:
        df = extract_func() if callable(extract_func) else None
    except Exception:
        df = None

    if df is None:
        # As a fallback, try reading previously saved raw file
        raw_path = 'data/exports_2024_2025.csv'
        if os.path.exists(raw_path):
            import pandas as pd
            df = pd.read_csv(raw_path)

    if df is None:
        raise FileNotFoundError("No data available and automatic download failed")

    # Comprehensive cleaning to match notebook process
    import pandas as pd
    
    # Rename columns if needed (match notebook)
    if 'sitc' in df.columns and 'product_description' not in df.columns:
        df = df.rename(columns={'sitc': 'product_description'})
    if 'sitc_code' in df.columns and 'prod_descpt_code' not in df.columns:
        df = df.rename(columns={'sitc_code': 'prod_descpt_code'})
    
    # Convert numeric columns and handle missing/negative values (match notebook)
    numeric_columns = ['quantity', 'gross_weight_tonnes', 'value_fob_aud']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill missing with 0 (match notebook)
            df[col] = df[col].fillna(0)
            # Convert negative values to 0 (match notebook)
            df[col] = df[col].clip(lower=0)
    
    # Convert text columns to string type first (match notebook)
    text_columns = ['country_of_destination', 'product_description']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Fill missing text values (match notebook)
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].replace('nan', 'Unknown').replace('NaN', 'Unknown').fillna('Unknown')
    
    # Clean product descriptions (match notebook)
    if 'product_description' in df.columns:
        df['product_description'] = df['product_description'].astype(str).str.strip()
        df['product_description'] = df['product_description'].str.replace(r'\s+', ' ', regex=True)
        df['product_description'] = df['product_description'].str.replace(r'[\n\r\t]+', ' ', regex=True)
        df['product_description'] = df['product_description'].str.replace('"', '', regex=False)
        df['product_description'] = df['product_description'].str.replace("'", '', regex=False)
    
    # Create derived features (match notebook - must happen before duplicate removal)
    import numpy as np
    from datetime import datetime
    
    # Extract year and month_number (match notebook)
    if 'month' in df.columns:
        # Extract year (4 digits at the end)
        if 'year' not in df.columns:
            df['year'] = df['month'].astype(str).str.extract(r'(\d{4})')[0]
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Extract month name and convert to month_number (match notebook)
        if 'month_number' not in df.columns:
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            # Extract month name (everything before the year) - match notebook format
            month_name_temp = df['month'].astype(str).str.extract(r'^([A-Za-z]+)')[0]
            df['month_number'] = month_name_temp.map(month_map)
            
            # Update month column to just month name (match notebook behavior)
            # But keep original format for compatibility, just store month name in a temp column
            # We'll use month_number for calculations
    
    # Create value_per_tonne metric (match notebook)
    if 'value_fob_aud' in df.columns and 'gross_weight_tonnes' in df.columns:
        df['value_per_tonne'] = df['value_fob_aud'] / df['gross_weight_tonnes'].replace(0, np.nan)
    
    # Add data processing date (match notebook)
    df['data_processed_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Remove duplicates (match notebook - happens after derived features but before mappings)
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        st.info(f"Removed {duplicates_removed:,} duplicate rows")
    
    # Apply SITC code mapping to unclassified products (match notebook - after duplicates)
    try:
        from sitc_mapping import map_sitc_to_product, get_unclassified_patterns
        if 'product_description' in df.columns and 'prod_descpt_code' in df.columns:
            unclassified_patterns = '|'.join(get_unclassified_patterns())
            mask_unclassified = df['product_description'].astype(str).str.contains(unclassified_patterns, case=False, na=False)
            if mask_unclassified.any():
                df.loc[mask_unclassified, 'product_description'] = df.loc[mask_unclassified, 'prod_descpt_code'].apply(map_sitc_to_product)
    except ImportError:
        # If sitc_mapping not available, skip this step
        pass
    
    # Apply country code mapping for problematic countries (match notebook - after SITC mapping)
    try:
        from country_mapping import is_problematic_country_name, map_country_code_to_name
        if 'country_of_destination' in df.columns and 'country_of_destination_code' in df.columns:
            # Map problematic country names using country_of_destination_code (match notebook)
            problematic_mask = df['country_of_destination'].apply(is_problematic_country_name)
            if problematic_mask.any():
                df.loc[problematic_mask, 'country_of_destination'] = df.loc[problematic_mask, 'country_of_destination_code'].apply(map_country_code_to_name)
    except ImportError:
        # If country_mapping not available, skip this step
        pass
    
    # Fix "Country Code XXX" entries (match notebook - happens AFTER problematic country mapping)
    try:
        from country_mapping import map_country_code_to_name
        if 'country_of_destination' in df.columns:
            country_code_mask = df['country_of_destination'].astype(str).str.contains('Country Code', na=False)
            if country_code_mask.any():
                # Extract the country code (remove "Country Code " prefix)
                df.loc[country_code_mask, 'country_of_destination'] = df.loc[country_code_mask, 'country_of_destination'].str.replace('Country Code ', '')
                # Apply proper mapping
                df.loc[country_code_mask, 'country_of_destination'] = df.loc[country_code_mask, 'country_of_destination'].apply(map_country_code_to_name)
    except ImportError:
        # If country_mapping not available, skip this step
        pass
    
    # Final normalization: strip whitespace from country names (affects unique count)
    if 'country_of_destination' in df.columns:
        df['country_of_destination'] = df['country_of_destination'].astype(str).str.strip()

    df.to_csv(cleaned_path, index=False)
    st.success("Data loaded and processed successfully!")
    return cleaned_path


@st.cache_data(ttl=60, max_entries=1, show_spinner=False)  # Very short TTL to free memory quickly
def load_exports_cleaned(path: str) -> pd.DataFrame:
    """Load all columns in chunks, add compatibility shims, and downcast numerics to save RAM."""
    # Load in optimized chunks - balance between memory and performance
    chunk_size = 75000  # 75k rows at a time - optimized for full dataset loading
    chunks = []
    df_combined = None
    
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            # Apply transformations to each chunk
            # Ensure month_number exists
            if 'month_number' not in chunk.columns and 'month' in chunk.columns:
                month_map = {
                    'January': 1, 'February': 2, 'March': 3, 'April': 4,
                    'May': 5, 'June': 6, 'July': 7, 'August': 8,
                    'September': 9, 'October': 10, 'November': 11, 'December': 12,
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                chunk['month_name'] = chunk['month'].astype(str).str.split().str[0]
                chunk['month_number'] = chunk['month_name'].map(month_map).fillna(1)
                chunk = chunk.drop(columns=['month_name'], errors='ignore')
            
            # Ensure product_description exists
            if 'product_description' not in chunk.columns:
                import re
                def _norm(s: str) -> str:
                    return re.sub(r"[^a-z0-9]", "", str(s).lower())
                
                candidates = []
                for col in chunk.columns:
                    col_norm = _norm(col)
                    if col_norm in {'productdescription', 'product_description', 'sitc'}:
                        candidates.append(col)
                
                if candidates:
                    chunk['product_description'] = chunk[candidates[0]].astype(str)
                else:
                    chunk['product_description'] = 'All Products'
            
            # Ensure code column exists
            if 'prod_descpt_code' not in chunk.columns and 'sitc_code' in chunk.columns:
                chunk['prod_descpt_code'] = chunk['sitc_code'].astype(str)
            
            # Downcast numerics
            for col in chunk.select_dtypes(include=['int64','float64']).columns:
                if pd.api.types.is_integer_dtype(chunk[col]):
                    chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
                else:
                    chunk[col] = pd.to_numeric(chunk[col], downcast='float')
            
            # Derived fields
            if 'year' in chunk.columns and 'month_number' in chunk.columns:
                chunk['date'] = pd.to_datetime(chunk['year'].astype(str) + '-' + chunk['month_number'].astype(str).str.zfill(2) + '-01')
            if {'value_fob_aud','gross_weight_tonnes'}.issubset(chunk.columns):
                chunk['value_per_tonne'] = chunk['value_fob_aud'] / chunk['gross_weight_tonnes']
            
            chunks.append(chunk)
            # Combine chunks in smaller batches to reduce memory spikes
            # Process and combine more frequently to avoid memory buildup
            if len(chunks) >= 3:  # Combine every 3 chunks (150k rows) - balanced approach
                if df_combined is None:
                    df_combined = pd.concat(chunks, ignore_index=True)
                else:
                    df_combined = pd.concat([df_combined] + chunks, ignore_index=True)
                chunks = []
                gc.collect()
        
        # Combine remaining chunks
        if chunks:
            if df_combined is None:
                df_combined = pd.concat(chunks, ignore_index=True)
            else:
                df_combined = pd.concat([df_combined] + chunks, ignore_index=True)
        
        df = df_combined if df_combined is not None else pd.DataFrame()
        del chunks, df_combined
        gc.collect()
        
        return df
    except Exception as e:
        st.error(f"Error loading data file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()  # Return empty dataframe on error


@st.cache_data(ttl=60, max_entries=1)  # Very short TTL
def compute_kpis_chunked(file_path: str) -> dict:
    """Compute KPIs by processing in chunks to avoid loading full dataset into memory."""
    total_value = 0.0
    total_weight = 0.0
    total_records = 0
    value_list = []  # For median calculation
    
    chunk_size = 50000  # Process 50k rows at a time
    
    try:
        # Process in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if 'value_fob_aud' in chunk.columns:
                total_value += float(chunk['value_fob_aud'].sum())
                # Sample values for median (every 100th record to save memory)
                value_list.extend(chunk['value_fob_aud'].iloc[::100].tolist())
            if 'gross_weight_tonnes' in chunk.columns:
                total_weight += float(chunk['gross_weight_tonnes'].sum())
            total_records += len(chunk)
            del chunk  # Explicitly delete chunk
            gc.collect()
        
        # Calculate median from sampled values
        median_val = float(pd.Series(value_list).median()) if value_list else 0.0
        del value_list
        gc.collect()
        
        return {
            'total_value': total_value,
            'total_weight': total_weight,
            'total_records': total_records,
            'total_shipments': total_records,
            'avg_shipment_value': (total_value / total_records) if total_records > 0 else 0.0,
            'median_shipment_value': median_val
        }
    except Exception as e:
        st.error(f"Error computing KPIs: {str(e)}")
        return None


@st.cache_data(ttl=300, max_entries=1)  # Moderate TTL for full dataset
def load_data_full(file_path: str) -> pd.DataFrame:
    """Load full dataset using optimized chunked loading - memory efficient."""
    # Use the existing chunked loading function which is already optimized
    return load_exports_cleaned(file_path)

@st.cache_data(ttl=60, max_entries=1)  # Very short TTL
def load_data():
    """Efficiently load dataset - use lazy loading for large datasets."""
    try:
        file_path = ensure_data_file()
        
        # Compute KPIs from chunks first (doesn't load full dataset)
        accurate_kpis = compute_kpis_chunked(file_path)
        if accurate_kpis is None:
            return None, None
        
        # Check if user wants full dataset or fast mode
        load_full = st.session_state.get('load_full_dataset', True)  # Default to full dataset
        
        if load_full:
            # Load FULL dataset using optimized chunked loading
            # This uses the load_exports_cleaned function which loads in chunks efficiently
            df = load_data_full(file_path)
        else:
            # Fast mode: Load only essential columns with row limit
            try:
                sample_df = pd.read_csv(file_path, nrows=1000)
                required_cols = ['year', 'month_number', 'value_fob_aud', 'gross_weight_tonnes', 
                               'country_of_destination', 'product_description', 'state_of_origin',
                               'port_of_loading', 'mode_of_transport']
                cols_to_load = [col for col in required_cols if col in sample_df.columns]
                
                import os
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                if file_size_mb > 100:
                    max_rows = 500000
                    df = pd.read_csv(file_path, usecols=cols_to_load, nrows=max_rows, low_memory=False)
                else:
                    df = pd.read_csv(file_path, usecols=cols_to_load, low_memory=False)
                
                # Add derived columns
                if 'date' not in df.columns and 'year' in df.columns and 'month_number' in df.columns:
                    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_number'].astype(str).str.zfill(2) + '-01')
                if 'value_per_tonne' not in df.columns and 'value_fob_aud' in df.columns and 'gross_weight_tonnes' in df.columns:
                    df['value_per_tonne'] = df['value_fob_aud'] / df['gross_weight_tonnes'].replace(0, np.nan)
                
                # Downcast to save memory
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                
                del sample_df
                gc.collect()
            except Exception as e:
                st.error(f"Error loading data file: {str(e)}")
                return None, None
        
        gc.collect()
        return df, accurate_kpis
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# Initialize session state for data
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
    st.session_state.accurate_kpis = None
    st.session_state.data_file_path = None
    st.session_state.load_full_dataset = True  # Default to full dataset

# Sidebar option for data loading mode (before loading data)
st.sidebar.header("Dashboard Controls")

# Data loading mode toggle
load_mode = st.sidebar.radio(
    "Data Loading Mode",
    ["Full Dataset (Recommended)", "Fast Mode (Limited Rows)"],
    index=0 if st.session_state.load_full_dataset else 1,
    help="Full Dataset: Loads all records using efficient chunked processing. Fast Mode: Limits to 500k rows for faster loading."
)

# Update session state
st.session_state.load_full_dataset = (load_mode == "Full Dataset (Recommended)")

# Check if we need to reload data (mode changed)
if st.session_state.df_loaded is not None:
    # Check if mode changed
    current_mode_full = st.session_state.load_full_dataset
    # If mode changed, clear cache and reload
    if hasattr(st.session_state, 'last_load_mode'):
        if st.session_state.last_load_mode != current_mode_full:
            st.session_state.df_loaded = None
            st.session_state.accurate_kpis = None
            st.cache_data.clear()
            st.rerun()
    st.session_state.last_load_mode = current_mode_full

# Load data with error handling - use session state to avoid reloading
if st.session_state.df_loaded is None:
    try:
        mode_text = "full dataset" if st.session_state.load_full_dataset else "sample data"
        with st.spinner(f'Loading {mode_text}... This may take a moment.'):
            df, accurate_kpis = load_data()
            if df is not None:
                st.session_state.df_loaded = df
                st.session_state.accurate_kpis = accurate_kpis
                st.session_state.last_load_mode = st.session_state.load_full_dataset
                if st.session_state.load_full_dataset:
                    st.success(f"✅ Loaded full dataset: {len(df):,} records")
                else:
                    st.info(f"⚡ Fast mode: Loaded {len(df):,} records")
            else:
                st.error("Failed to load data")
                st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()
else:
    df = st.session_state.df_loaded
    accurate_kpis = st.session_state.accurate_kpis

if df is not None and accurate_kpis is not None:
    # Helper function to safely execute sections
    def safe_execute(func, section_name):
        """Execute a function with error handling to prevent crashes."""
        try:
            func()
            gc.collect()  # Clean up after each section
        except Exception as e:
            st.error(f"Error in {section_name}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Final compatibility guard: ensure 'product_description' exists even if cached data is old
    if 'product_description' not in df.columns:
        import re
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(s).lower())
        candidates = ['product_description', 'product description', 'product', 'sitc', 'commodity', 'sitc description', 'sitc_description']
        norm_to_col = { _norm(c): c for c in df.columns }
        src = None
        for cand in candidates:
            key = _norm(cand)
            if key in norm_to_col:
                src = norm_to_col[key]
                break
        if src:
            df['product_description'] = df[src].astype(str)
        else:
            df['product_description'] = 'All Products'
    # Clean presentation - no status messages
    
    # Cache clear button (moved after data loading mode)
    if st.sidebar.button("Clear Cache & Reload Data", help="Clear cached data and force fresh data reload"):
        st.cache_data.clear()
        st.success("Cache cleared! Refreshing...")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Date range filter
    st.sidebar.subheader("Date Range")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    else:
        df_filtered = df
    
    # Country filter
    st.sidebar.subheader("Country Filter")
    all_countries = ['All Countries'] + sorted(df['country_of_destination'].unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=all_countries,
        default=['All Countries']
    )
    
    if 'All Countries' not in selected_countries and selected_countries:
        df_filtered = df_filtered[df_filtered['country_of_destination'].isin(selected_countries)]
    
    # Product filter
    st.sidebar.subheader("Product Filter")
    all_products = ['All Products'] + sorted(df['product_description'].unique().tolist())
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=all_products,
        default=['All Products']
    )
    
    if 'All Products' not in selected_products and selected_products:
        df_filtered = df_filtered[df_filtered['product_description'].isin(selected_products)]
    
    # Main dashboard content
    
    # 1. DATASET SUMMARY (from your notebook Cell 4)
    st.markdown('<h2 class="section-header">Dataset Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Records</h3>
            <h2>{len(df_filtered):,}</h2>
            <p>Individual Shipments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Countries</h3>
            <h2>{df_filtered['country_of_destination'].nunique()}</h2>
            <p>Export Destinations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Products</h3>
            <h2>{df_filtered['product_description'].nunique()}</h2>
            <p>Product Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>States</h3>
            <h2>{df_filtered['state_of_origin'].nunique()}</h2>
            <p>Export States</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Financial Summary (from your notebook Cell 4)
    st.markdown('<h2 class="section-header">Financial Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate from filtered dataset to respect date range selection
        total_value = df_filtered['value_fob_aud'].sum()
        if total_value >= 1e9:
            value_display = f"${total_value/1e9:.1f}B"
        elif total_value >= 1e6:
            value_display = f"${total_value/1e6:.1f}M"
        elif total_value >= 1e3:
            value_display = f"${total_value/1e3:.1f}K"
        else:
            value_display = f"${total_value:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Export Value</h3>
            <h2>{value_display}</h2>
            <p>Australian Dollars </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate from filtered dataset to respect date range selection
        avg_shipment = df_filtered['value_fob_aud'].mean()
        if avg_shipment >= 1e6:
            avg_display = f"${avg_shipment/1e6:.1f}M"
        elif avg_shipment >= 1e3:
            avg_display = f"${avg_shipment/1e3:.1f}K"
        else:
            avg_display = f"${avg_shipment:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Shipment Value</h3>
            <h2>{avg_display}</h2>
            <p>Australian Dollars </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate from filtered dataset to respect date range selection
        median_shipment = df_filtered['value_fob_aud'].median()
        if median_shipment >= 1e6:
            median_display = f"${median_shipment/1e6:.1f}M"
        elif median_shipment >= 1e3:
            median_display = f"${median_shipment/1e3:.1f}K"
        else:
            median_display = f"${median_shipment:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Median Shipment</h3>
            <h2>{median_display}</h2>
            <p>Australian Dollars </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate from filtered dataset to respect date range selection
        total_weight = df_filtered['gross_weight_tonnes'].sum()
        if total_weight >= 1e9:
            weight_display = f"{total_weight/1e9:.1f}B"
        elif total_weight >= 1e6:
            weight_display = f"{total_weight/1e6:.1f}M"
        elif total_weight >= 1e3:
            weight_display = f"{total_weight/1e3:.1f}K"
        else:
            weight_display = f"{total_weight:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Weight</h3>
            <h2>{weight_display}</h2>
            <p>Tonnes </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Force cleanup after overview
    gc.collect()
    
    # 2. TIME SERIES ANALYSIS (from your notebook Cell 6) - LAZY LOADED
    try:
        st.markdown('<h2 class="section-header">Time Series Analysis</h2>', unsafe_allow_html=True)
        
        # Use filtered dataset for time series analysis - avoid copy
        df_ts_raw = df_filtered  # Use view to save memory
        
        if df_ts_raw.empty:
            st.warning("No data available for time series analysis.")
        else:
            # Check required columns exist
            required_cols = ['year', 'month_number', 'value_fob_aud', 'gross_weight_tonnes']
            missing_cols = [col for col in required_cols if col not in df_ts_raw.columns]
            if not missing_cols:
                # Ensure month_number and year are not NaN for grouping - create copy only when needed
                mask = df_ts_raw['month_number'].notna() & df_ts_raw['year'].notna()
                df_ts = df_ts_raw[mask].copy()  # Only copy filtered subset
                del mask
                gc.collect()
                
                if len(df_ts) == 0:
                    st.warning("No valid date data available for time series analysis.")
                else:
                    # Monthly trends - handle month column (may be missing or in different format)
                    if 'month' in df_ts.columns:
                        group_cols = ['year', 'month_number', 'month']
                    else:
                        # Create month name from month_number if month column is missing
                        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                                       5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                       9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                        df_ts['month'] = df_ts['month_number'].map(month_names).fillna('Unknown')
                        group_cols = ['year', 'month_number', 'month']
                    
                    # Process aggregation efficiently
                    monthly = df_ts.groupby(group_cols).agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum'
                    }).reset_index().sort_values(['year', 'month_number'])
                    
                    # Clean up df_ts immediately after use
                    del df_ts
                    gc.collect()
                    
                    # Check if we have data for the selected date range
                    if len(monthly) > 0:
                        # Safely create period string
                        monthly['period'] = monthly['month'].astype(str) + ' ' + monthly['year'].astype(str)
                        # Handle division by zero and infinite values
                        monthly['value_per_tonne'] = monthly['value_fob_aud'] / monthly['gross_weight_tonnes'].replace(0, np.nan)
                        monthly['value_per_tonne'] = monthly['value_per_tonne'].replace([np.inf, -np.inf], np.nan).fillna(0)
                        
                        # Ensure all numeric columns are valid
                        monthly = monthly[monthly['value_fob_aud'].notna() & monthly['gross_weight_tonnes'].notna()]
                        
                        if len(monthly) == 0:
                            st.warning("No valid data after cleaning for time series analysis.")
                        else:
                            # Chart 1: Export Value Over Time (Interactive)
                            st.subheader("Monthly Export Value Trend")
                            
                            # Smart formatting for export values
                            def format_export_value(value):
                                if value >= 1e9:
                                    return f"${value/1e9:.1f}B"
                                elif value >= 1e6:
                                    return f"${value/1e6:.1f}M"
                                elif value >= 1e3:
                                    return f"${value/1e3:.1f}K"
                                else:
                                    return f"${value:,.0f}"
                            
                            fig1 = px.line(monthly, x='period', y=monthly['value_fob_aud'] / 1e9,
                                           title='Monthly Export Value Trend (2024-2025)',
                                           labels={'y': 'Export Value (Billion AUD)', 'x': 'Month'},
                                           markers=True)
                            fig1.update_traces(
                                line_color='#2E86AB', 
                                line_width=3, 
                                marker_size=8,
                                mode='lines+markers+text',
                                text=[format_export_value(value) for value in monthly['value_fob_aud']],
                                textposition='top center'
                            )
                            fig1.update_layout(
                                title_font_size=16,
                                title_font_color='#2c3e50',
                                xaxis_title_font_size=14,
                                yaxis_title_font_size=14,
                                hovermode='x unified',
                                template='plotly_white',
                                yaxis=dict(tickformat='.1f')
                            )
                            fig1.update_xaxes(tickangle=45)
                            st.plotly_chart(fig1)

                            # Chart 2: Export Weight Over Time (Interactive)
                            st.subheader("Monthly Export Weight Trend")
                            
                            # Smart formatting for weight values
                            def format_weight_value(value):
                                if value >= 1e9:
                                    return f"{value/1e9:.1f}B"
                                elif value >= 1e6:
                                    return f"{value/1e6:.1f}M"
                                elif value >= 1e3:
                                    return f"{value/1e3:.1f}K"
                                else:
                                    return f"{value:,.0f}"
                            
                            fig2 = px.line(monthly, x='period', y=monthly['gross_weight_tonnes'] / 1e6,
                                           title='Monthly Export Weight Trend (2024-2025)',
                                           labels={'y': 'Export Weight (Million Tonnes)', 'x': 'Month'},
                                           markers=True)
                            fig2.update_traces(
                                line_color='#F18F01', 
                                line_width=3, 
                                marker_size=8,
                                marker_symbol='square',
                                mode='lines+markers+text',
                                text=[format_weight_value(value) for value in monthly['gross_weight_tonnes']],
                                textposition='top center'
                            )
                            fig2.update_layout(
                                title_font_size=16,
                                title_font_color='#2c3e50',
                                xaxis_title_font_size=14,
                                yaxis_title_font_size=14,
                                hovermode='x unified',
                                template='plotly_white',
                                yaxis=dict(tickformat='.1f')
                            )
                            fig2.update_xaxes(tickangle=45)
                            st.plotly_chart(fig2)

                            # Chart 3: Value per Tonne Over Time (Interactive) - KEY METRIC FOR LOGISTICS!
                            st.subheader("Average Value per Tonne Trend")
                            
                            # Smart formatting for value per tonne
                            def format_value_per_tonne(value):
                                if value >= 1e6:
                                    return f"${value/1e6:.1f}M"
                                elif value >= 1e3:
                                    return f"${value/1e3:.1f}K"
                                else:
                                    return f"${value:,.0f}"
                            
                            fig3 = px.line(monthly, x='period', y=monthly['value_per_tonne'],
                                           title='Average Value per Tonne Trend (2024-2025)',
                                           labels={'y': 'Value per Tonne (AUD)', 'x': 'Month'},
                                           markers=True)
                            fig3.update_traces(
                                line_color='#06A77D', 
                                line_width=3, 
                                marker_size=8,
                                marker_symbol='diamond',
                                mode='lines+markers+text',
                                text=[format_value_per_tonne(value) for value in monthly['value_per_tonne']],
                                textposition='top center'
                            )
                            fig3.update_layout(
                                title_font_size=16,
                                title_font_color='#2c3e50',
                                xaxis_title_font_size=14,
                                yaxis_title_font_size=14,
                                hovermode='x unified',
                                template='plotly_white',
                                yaxis=dict(tickformat=',.0f')
                            )
                            fig3.update_xaxes(tickangle=45)
                            st.plotly_chart(fig3)
                    else:
                        st.warning("No data available for the selected date range. Please adjust your date filter.")
            else:
                st.warning(f"Missing required columns for time series analysis: {missing_cols}")
                
                # Clean up memory after time series section
                del df_ts_raw
                if 'monthly' in locals():
                    del monthly
                gc.collect()
    except Exception as e:
        st.error(f"Error in Time Series Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        gc.collect()
    
    # Force cleanup after time series
    gc.collect()
    
    # 3. COUNTRY ANALYSIS (from your notebook Cell 9) - LAZY LOADED
    st.markdown('<h2 class="section-header">Country Analysis</h2>', unsafe_allow_html=True)
    
    # Use filtered dataset for country analysis - avoid copy
    df_country = df_filtered  # Use view to save memory
    
    if not df_country.empty:
        # Top export destinations - process efficiently
        top_countries = df_country.groupby('country_of_destination').agg({
            'value_fob_aud': 'sum',
            'gross_weight_tonnes': 'sum'
        }).sort_values('value_fob_aud', ascending=False)
        
        # Clean up df_country reference immediately
        del df_country
        gc.collect()
        
        # Check if we have data for the selected date range
        if len(top_countries) > 0:
            top_countries['value_billions'] = top_countries['value_fob_aud'] / 1e9
            top_countries['pct'] = (top_countries['value_fob_aud'] / top_countries['value_fob_aud'].sum() * 100)
            
            # Display top 15 countries
            # Clean presentation - visualization shows the data
            
            # Interactive Visualization with Value Labels (No Percentage)
            top_15 = top_countries.head(15).reset_index()
            
            fig = px.bar(top_15, x='value_billions', y='country_of_destination',
                         orientation='h',
                         title='Top 15 Countries by Export Value',
                         labels={'value_billions': 'Export Value (Billion AUD)', 'country_of_destination': 'Country'},
                         color='value_billions',
                         color_continuous_scale='Greens')
            fig.update_traces(
                text=[f"${value:.1f}B" for value in top_15['value_billions']],
                textposition='outside'
            )
            
            fig.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                template='plotly_white',
                height=600,
                showlegend=False
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(tickformat='$,.1f')
            
            # Update text positioning and styling
            fig.update_traces(
                textposition='outside',
                textfont=dict(size=10, color='#2c3e50'),
                hovertemplate='<b>%{y}</b><br>Export Value: $%{x:.1f}B<extra></extra>'
            )
            
            st.plotly_chart(fig)
            
            # Clean up memory after country analysis
            del top_countries, top_15, fig
            gc.collect()
        else:
            st.warning("No data available for the selected date range. Please adjust your date filter.")
    else:
        st.warning("No data available for country analysis")
    
    # Force cleanup
    gc.collect()
    
    # 4. PRODUCT ANALYSIS (from your notebook Cell 11) - LAZY LOADED
    st.markdown('<h2 class="section-header">Product Analysis</h2>', unsafe_allow_html=True)
    
    # Use filtered dataset for product analysis - avoid copy
    df_product = df_filtered  # Use view to save memory
    
    if not df_product.empty:
        # Top 20 products by value - process efficiently
        top_products = df_product.groupby('product_description').agg({
            'value_fob_aud': 'sum',
            'gross_weight_tonnes': 'sum'
        }).sort_values('value_fob_aud', ascending=False)
        
        # Clean up df_product reference immediately
        del df_product
        gc.collect()

        # Create clean display columns (rounded to 2 decimal places)
        top_products['Value ($B)'] = (top_products['value_fob_aud'] / 1e9).round(2)
        top_products['Weight (M Tonnes)'] = (top_products['gross_weight_tonnes'] / 1e6).round(2)
        top_products['% Total'] = ((top_products['value_fob_aud'] / top_products['value_fob_aud'].sum() * 100)).round(2)

        # Check if we have data for the selected date range
        if len(top_products) > 0:
            # Get top 20 with FULL product names (no truncation)
            top_20_display = top_products.head(20).copy()
            top_20_display['Product'] = top_20_display.index  # Full product name

            # Reset index and add rank
            top_20_display = top_20_display.reset_index(drop=True)
            if len(top_20_display) > 0:
                top_20_display.index = range(1, min(21, len(top_20_display) + 1))
                top_20_display.index.name = 'Rank'
            
            # Create final display DataFrame
            products_df = top_20_display[['Product', 'Value ($B)', 'Weight (M Tonnes)', '% Total']].copy()
            
            # Display summary
            st.subheader("Top 20 Products by Export Value")
            
            # Interactive Product Visualization with Value Labels (No Percentage)
            top_20_products = top_products.head(20).reset_index()
            
            fig = px.bar(top_20_products, x='value_fob_aud', y='product_description',
                         orientation='h',
                         title='Top 20 Products by Export Value',
                         labels={'value_fob_aud': 'Export Value (AUD)', 'product_description': 'Product'},
                         color='value_fob_aud',
                         color_continuous_scale='Blues')
            fig.update_traces(
                text=[f"${value/1e9:.1f}B" for value in top_20_products['value_fob_aud']],
                textposition='outside',
                textfont=dict(size=9, color='#2c3e50'),
                hovertemplate='<b>%{y}</b><br>Export Value: $%{x:,.0f}<extra></extra>'
            )
            
            fig.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                template='plotly_white',
                height=800,
                showlegend=False
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(tickformat='$,.0f')
            
            st.plotly_chart(fig)
        else:
            st.warning("No data available for the selected date range. Please adjust your date filter.")
        
        # Clean up memory after product analysis
        del top_products, top_20_display, top_20_products, products_df, fig
        gc.collect()
    else:
        st.warning("No data available for product analysis")
    
    # Force cleanup
    gc.collect()
    
    # 4.5. INDUSTRY CATEGORY ANALYSIS (EXACT from your notebook) - LAZY LOADED
    st.markdown('<h2 class="section-header">Industry Category Analysis</h2>', unsafe_allow_html=True)
    
    # Use the main dataset for accurate industry analysis (same as other sections)
    
    # Use the filtered dataset to respect date range selection - avoid copy
    df_full_industry = df_filtered  # Use view to save memory
    
    # SITC Code-based Product Categorization - Clean presentation

    # Ensure the expected code column exists; map from 'sitc_code' when available
    if 'prod_descpt_code' not in df_full_industry.columns:
        if 'sitc_code' in df_full_industry.columns:
            df_full_industry['prod_descpt_code'] = df_full_industry['sitc_code'].astype(str)
        else:
            df_full_industry['prod_descpt_code'] = ''

    # Import SITC mapping and create sitc_category column
    try:
        from sitc_mapping import SITC_MAPPING
        
        def get_sitc_section(sitc_code):
            if pd.isna(sitc_code) or sitc_code == '':
                return 'Other Commodities'
            sitc_str = str(sitc_code).strip()
            if len(sitc_str) >= 2:
                section_code = sitc_str[:2]
                return SITC_MAPPING.get(section_code, 'Other Commodities')
            return 'Other Commodities'
        
        df_full_industry['sitc_category'] = df_full_industry['prod_descpt_code'].apply(get_sitc_section)
        # Clean dashboard - no unnecessary text
    except ImportError:
        st.warning("SITC mapping module not found, using fallback categorization")
        df_full_industry['sitc_category'] = 'Other Commodities'
    
    # Create Stakeholder-Friendly Industry Categories (EXACT from your notebook)
    # Clean dashboard - no unnecessary text
    
    # Map SITC code to industry category using first digit
    def get_industry_category(sitc_code):
        if pd.isna(sitc_code) or sitc_code == '':
            return 'Other Commodities'
        first_digit = str(sitc_code).strip()[0] if len(str(sitc_code).strip()) >= 1 else '9'
        mapping = {'0': 'Food & Agriculture', '1': 'Beverages & Tobacco', '2': 'Raw Materials & Mining', 
                   '3': 'Energy & Petroleum', '4': 'Food Processing', '5': 'Chemicals & Pharmaceuticals',
                   '6': 'Manufactured Goods and materials', '7': 'Machinery & Equipment', 
                   '8': 'Consumer Goods', '9': 'Other Commodities'}
        return mapping.get(first_digit, 'Other Commodities')
    
    df_full_industry['industry_category'] = df_full_industry['prod_descpt_code'].apply(get_industry_category)
    # Clean dashboard - no unnecessary text
    
    # Analyze by industry category - process efficiently
    industry_analysis = df_full_industry.groupby('industry_category').agg({
        'value_fob_aud': ['sum', 'count', 'mean'],
        'gross_weight_tonnes': 'sum',
        'value_per_tonne': 'mean'
    }).round(2)
    
    # Clean up df_full_industry immediately after aggregation
    del df_full_industry
    gc.collect()
    
    # Flatten column names
    industry_analysis.columns = ['Total_Value', 'Shipment_Count', 'Avg_Value', 'Total_Weight', 'Avg_Value_per_Tonne']
    industry_analysis = industry_analysis.sort_values('Total_Value', ascending=False)
    
    # Calculate percentages
    total_value = industry_analysis['Total_Value'].sum()
    industry_analysis['Value_Percentage'] = (industry_analysis['Total_Value'] / total_value * 100).round(1)
    
    # Clean presentation - show only visualizations
    
    # Create individual visualizations (separate charts)
    plt.rcParams['figure.dpi'] = 80  # Standard DPI for individual charts
    
    # Chart 1: Industry Categories by Export Value (Horizontal Bar)
    st.subheader("Australian Export Value by Industry Category")
    
    top_industries = industry_analysis.head(8).reset_index()
    
    # Add value labels with percentage
    fig1 = px.bar(top_industries, x='Total_Value', y='industry_category',
                  orientation='h',
                  title='Australian Export Value by Industry Category',
                  labels={'Total_Value': 'Export Value (AUD)', 'industry_category': 'Industry'},
                  color='Total_Value',
                  color_continuous_scale='viridis')
    fig1.update_traces(
        text=[f"${value/1e9:.1f}B<br>({pct:.1f}%)" for value, pct in zip(top_industries['Total_Value'], top_industries['Value_Percentage'])],
        textposition='outside'
    )
    
    fig1.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600,
        showlegend=False
    )
    fig1.update_yaxes(autorange="reversed")
    fig1.update_xaxes(tickformat='$,.0f')
    
    # Update text positioning and styling
    fig1.update_traces(
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{y}</b><br>Export Value: $%{x:,.0f}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=top_industries['Value_Percentage']
    )
    
    st.plotly_chart(fig1)
    
    # Chart 2: Industry Value Density (Interactive) - EXACT from your notebook
    st.subheader("Industry Value Density (Value per Tonne Shipped)")
    
    # Calculate ratio of totals (Total Value ÷ Total Weight) per industry - CORRECT method
    industry_analysis['Value_per_Tonne_Ratio'] = industry_analysis['Total_Value'] / industry_analysis['Total_Weight']
    value_density_industry = industry_analysis.sort_values('Value_per_Tonne_Ratio', ascending=False).reset_index()
    
    fig2 = px.bar(value_density_industry, x='industry_category', y='Value_per_Tonne_Ratio',
                  title='Industry Value Density (Value per Tonne Shipped)',
                  labels={'Value_per_Tonne_Ratio': 'Value per Tonne (AUD)', 'industry_category': 'Industry'},
                  color='Value_per_Tonne_Ratio',
                  color_continuous_scale='viridis')
    fig2.update_traces(
        text=[f"${value:,.0f}/tonne" for value in value_density_industry['Value_per_Tonne_Ratio']],
        textposition='outside'
    )
    
    fig2.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600,
        showlegend=False
    )
    fig2.update_xaxes(tickangle=45)
    fig2.update_yaxes(tickformat='$,.0f')
    
    # Update text positioning and styling
    fig2.update_traces(
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{x}</b><br>Value per Tonne: $%{y:,.0f}<extra></extra>'
    )
    
    st.plotly_chart(fig2)
    
    # Chart 3: Dynamic Interactive Performance Summary Table
    st.subheader("Industry Performance Summary Table")
    
    # Create a comprehensive industry performance table (without Value/Tonne) - Dynamic and updates with filters
    table_data = industry_analysis[['Total_Value', 'Shipment_Count', 'Avg_Value', 'Value_Percentage']].copy()
    
    # Format the data for better readability
    table_data['Total_Value_B'] = (table_data['Total_Value'] / 1e9).round(2)
    table_data['Avg_Value_K'] = (table_data['Avg_Value'] / 1e3).round(0)
    table_data['Shipment_Count_K'] = (table_data['Shipment_Count'] / 1e3).round(0)
    
    # Create display table with formatted columns (removed Value/Tonne)
    display_table = pd.DataFrame({
        'Industry': table_data.index,
        'Total Value ($B)': table_data['Total_Value_B'],
        'Market Share (%)': table_data['Value_Percentage'],
        'Shipments (K)': table_data['Shipment_Count_K'],
        'Avg Value/Shipment ($K)': table_data['Avg_Value_K']
    })
    
    # Reset index to make Industry a regular column for better sorting/filtering
    display_table = display_table.reset_index(drop=True)
    
    # Apply styling with lighter colors - alternating rows with soft pastel colors
    def style_table(row):
        """Apply alternating row colors with lighter pastel shades"""
        # Soft pastel colors for alternating rows
        colors = ['#F5F7FA', '#FFFFFF']  # Very light grey-blue and white
        return [f'background-color: {colors[row.name % 2]}' for _ in row]
    
    # Style the table with lighter pastel colors
    styled_table = display_table.style.apply(style_table, axis=1).format({
        'Total Value ($B)': '{:.2f}',
        'Market Share (%)': '{:.1f}',
        'Shipments (K)': '{:.0f}',
        'Avg Value/Shipment ($K)': '{:.0f}'
    }).background_gradient(
        subset=['Total Value ($B)'],
        cmap='Blues',
        vmin=display_table['Total Value ($B)'].min(),
        vmax=display_table['Total Value ($B)'].max()
    ).set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#E8F4F8'),  # Very light blue header
                ('color', '#2C5282'),  # Soft blue text
                ('font-weight', '600'),
                ('text-align', 'center'),
                ('padding', '12px 8px'),
                ('border-bottom', '2px solid #BEE3F8'),
                ('font-size', '14px')
            ]
        },
        {
            'selector': 'tbody tr:hover',
            'props': [
                ('background-color', '#E0F2FE'),  # Very light cyan on hover
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('padding', '10px 8px'),
                ('text-align', 'center'),
                ('border-bottom', '1px solid #E2E8F0'),  # Light grey border
                ('font-size', '13px')
            ]
        },
        {
            'selector': 'tbody tr:nth-of-type(even)',
            'props': [
                ('background-color', '#F7FAFC')  # Very light grey-blue for even rows
            ]
        },
        {
            'selector': 'tbody tr:nth-of-type(odd)',
            'props': [
                ('background-color', '#FFFFFF')  # Pure white for odd rows
            ]
        }
    ]).set_properties(**{
        'font-size': '13px',
        'font-family': 'Segoe UI, Arial, sans-serif'
    })
    
    # Display styled table using Streamlit
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True
    )
    
    # Clean dashboard - no unnecessary text
    
    # 4.6. PRODUCT-MARKET ANALYSIS (EXACT from your notebook)
    st.markdown('<h2 class="section-header">Product-Market Analysis</h2>', unsafe_allow_html=True)
    
    # Use filtered dataset for Product-Market Analysis - columns already exist (date and value_per_tonne already calculated)
    df_full_product_market = df_filtered  # Use view to save memory
    
    # Function to format large numbers with proper suffixes
    def format_number(value):
        """Format numbers with B, M, K suffixes and appropriate decimal places"""
        if value >= 1e9:
            return f"{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{value/1e6:.2f}M"
        elif value >= 1e3:
            return f"{value/1e3:.2f}K"
        else:
            return f"{value:.2f}"
    
    # Clean presentation - show only visualizations
    
    # Calculate ALL data for visualizations - process efficiently before deleting
    top_products = df_full_product_market.groupby('product_description').agg({
        'value_fob_aud': 'sum',
        'gross_weight_tonnes': 'sum',
        'country_of_destination': 'nunique'
    }).round(2)
    
    top_products.columns = ['Total_Value', 'Total_Weight', 'Countries_Served']
    top_products = top_products.sort_values('Total_Value', ascending=False).head(10)
    
    top_countries = df_full_product_market.groupby('country_of_destination').agg({
        'value_fob_aud': 'sum',
        'gross_weight_tonnes': 'sum',
        'product_description': 'nunique'
    }).round(2)
    
    top_countries.columns = ['Total_Value', 'Total_Weight', 'Products_Imported']
    top_countries = top_countries.sort_values('Total_Value', ascending=False).head(15)
    
    # Calculate product diversification BEFORE deleting df_full_product_market
    country_product_diversity = df_full_product_market.groupby('country_of_destination').agg({
        'value_fob_aud': 'sum',
        'product_description': 'nunique'
    }).round(2).reset_index()
    
    # NOW we can safely delete df_full_product_market
    del df_full_product_market
    gc.collect()
    
    # 4. PRODUCT-MARKET VISUALIZATIONS (Interactive Individual Charts)
    
    # 1. TOP PRODUCTS BY EXPORT VALUE (Interactive)
    st.subheader("Top 10 Export Products by Value")
    top_10_products = top_products.head(10).reset_index()
    fig1 = px.bar(top_10_products, x='Total_Value', y='product_description',
                  orientation='h',
                  title='TOP 10 EXPORT PRODUCTS BY VALUE',
                  labels={'Total_Value': 'Export Value (AUD)', 'product_description': 'Product'},
                  color='Total_Value',
                  color_continuous_scale='viridis')
    fig1.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600
    )
    fig1.update_yaxes(autorange="reversed")
    fig1.update_xaxes(tickformat='$,.0f')
    st.plotly_chart(fig1)
    
    # 2. TOP DESTINATION COUNTRIES (Interactive)
    st.subheader("Top 10 Destination Countries")
    top_10_countries = top_countries.head(10).reset_index()
    fig2 = px.bar(top_10_countries, x='Total_Value', y='country_of_destination',
                  orientation='h',
                  title='TOP 10 DESTINATION COUNTRIES',
                  labels={'Total_Value': 'Import Value (AUD)', 'country_of_destination': 'Country'},
                  color='Total_Value',
                  color_continuous_scale='plasma')
    fig2.update_traces(
        text=[f"${value/1e9:.1f}B" for value in top_10_countries['Total_Value']],
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{y}</b><br>Import Value: $%{x:,.0f}<extra></extra>'
    )
    fig2.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600
    )
    fig2.update_yaxes(autorange="reversed")
    fig2.update_xaxes(tickformat='$,.0f')
    st.plotly_chart(fig2)
    
    # 3. PRODUCT DIVERSIFICATION BY COUNTRY (Interactive)
    st.subheader("Product Diversification by Country")
    # country_product_diversity already calculated above before deleting df_full_product_market
    
    fig3 = px.scatter(country_product_diversity, x='product_description', y='value_fob_aud',
                      size='value_fob_aud',
                      color='value_fob_aud',
                      title='PRODUCT DIVERSIFICATION BY COUNTRY',
                      labels={'product_description': 'Number of Products Imported', 'value_fob_aud': 'Total Import Value (AUD)'},
                      hover_name='country_of_destination',
                      color_continuous_scale='viridis')
    fig3.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600
    )
    fig3.update_yaxes(tickformat='$,.0f')
    st.plotly_chart(fig3)
    
    # 4. MARKET CONCENTRATION ANALYSIS (Interactive)
    st.subheader("Market Concentration Analysis")
    # Top 10 countries market share
    top_10_share = top_countries.head(10)['Total_Value']
    others_share = top_countries.iloc[10:]['Total_Value'].sum()
    market_share_data = list(top_10_share.values) + [others_share]
    market_share_labels = list(top_10_share.index) + ['Others']
    
    market_share_df = pd.DataFrame({
        'Country': market_share_labels,
        'Value': market_share_data
    })
    
    fig4 = px.pie(market_share_df, values='Value', names='Country',
                   title='MARKET CONCENTRATION (Top 10 Countries vs Others)',
                   color_discrete_sequence=px.colors.qualitative.Set3)
    fig4.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        template='plotly_white',
        height=500
    )
    fig4.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig4)
    
    # Clean dashboard - no unnecessary text
    
    # 4.7. TOP 15 PORTS BY TONNAGE (EXACT from your notebook)
    st.markdown('<h2 class="section-header">Top 15 Ports by Tonnage</h2>', unsafe_allow_html=True)
    
    # Use filtered dataset for Port Analysis - columns already exist (date and value_per_tonne already calculated)
    df_full_ports = df_filtered  # Use view to save memory
    
    # Clean dashboard - no unnecessary text
    
    # Group by port of loading and calculate total tonnage
    port_tonnage = df_full_ports.groupby('port_of_loading')['gross_weight_tonnes'].sum().reset_index()
    
    # Clean up df_full_ports immediately after aggregation
    del df_full_ports
    gc.collect()
    
    port_tonnage = port_tonnage.sort_values('gross_weight_tonnes', ascending=False)
    port_tonnage['tonnage_millions'] = port_tonnage['gross_weight_tonnes'] / 1e6
    top_15_ports = port_tonnage.head(15)
    
    # Clean dashboard - no unnecessary text
    
    # Create interactive lollipop chart
    fig = go.Figure()
    
    # Add horizontal lines (sticks)
    for i, (_, row) in enumerate(top_15_ports.iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row['tonnage_millions']],
            y=[i, i],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add dots (lollipops) at the end of each line
    fig.add_trace(go.Scatter(
        x=top_15_ports['tonnage_millions'],
        y=list(range(len(top_15_ports))),
        mode='markers',
        marker=dict(
            size=15,
            color=top_15_ports['tonnage_millions'],
            colorscale='viridis',
            line=dict(color='black', width=1)
        ),
        text=[f"{port}<br>Tonnage: {tonnage:.1f}M tonnes" for port, tonnage in zip(top_15_ports['port_of_loading'], top_15_ports['tonnage_millions'])],
        hovertemplate='%{text}<extra></extra>',
        name='Ports'
    ))
    
    # Update layout
    fig.update_layout(
        title='TOP 15 PORTS BY TONNAGE',
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title='Tonnage (Million Tonnes)',
        yaxis_title='Port',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600,
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_15_ports))),
            ticktext=top_15_ports['port_of_loading'].tolist(),
            autorange="reversed"
        ),
        xaxis=dict(tickformat=',.1f'),
        showlegend=False
    )
    
    st.plotly_chart(fig)
    
    # 4.8. VOLUME VS VALUE ANALYSIS (EXACT from your notebook)
    st.markdown('<h2 class="section-header">Volume vs Value Analysis</h2>', unsafe_allow_html=True)
    
    # Volume vs. Value Analysis by Product (EXACT from your notebook)
    
    # st.write("**=== VOLUME VS. VALUE ANALYSIS BY PRODUCT (INDUSTRY-IDENTIFIED) ===**")
    
    # Use filtered dataset for Volume vs Value Analysis - columns already exist (date and value_per_tonne already calculated)
    df_full_volume_value = df_filtered  # Use view to save memory
    
    # Add industry category to full dataset
    # Ensure code column exists for this section as well
    if 'prod_descpt_code' not in df_full_volume_value.columns:
        if 'sitc_code' in df_full_volume_value.columns:
            df_full_volume_value['prod_descpt_code'] = df_full_volume_value['sitc_code'].astype(str)
        else:
            df_full_volume_value['prod_descpt_code'] = ''
    def get_industry_category(sitc_code):
        if pd.isna(sitc_code) or sitc_code == '':
            return 'Other Commodities'
        first_digit = str(sitc_code).strip()[0] if len(str(sitc_code).strip()) >= 1 else '9'
        mapping = {'0': 'Food & Agriculture', '1': 'Beverages & Tobacco', '2': 'Raw Materials & Mining', 
                   '3': 'Energy & Petroleum', '4': 'Food Processing', '5': 'Chemicals & Pharmaceuticals',
                   '6': 'Manufactured Goods and materials', '7': 'Machinery & Equipment', 
                   '8': 'Consumer Goods', '9': 'Other Commodities'}
        return mapping.get(first_digit, 'Other Commodities')
    
    df_full_volume_value['industry_category'] = df_full_volume_value['prod_descpt_code'].apply(get_industry_category)
    
    # Calculate volume and value metrics for each product
    product_analysis = df_full_volume_value.groupby('product_description').agg({
        'value_fob_aud': 'sum',
        'gross_weight_tonnes': 'sum',
        'value_per_tonne': 'mean',
        'industry_category': 'first'  # Get the industry category for each product
    }).round(2)
    
    # Reset index to make product_description a column (not just an index)
    product_analysis = product_analysis.reset_index()
    
    # Calculate additional metrics
    shipment_counts = df_full_volume_value.groupby('product_description').size()
    product_analysis['shipment_count'] = product_analysis['product_description'].map(shipment_counts).fillna(0)
    product_analysis['avg_shipment_value'] = (product_analysis['value_fob_aud'] / product_analysis['shipment_count']).round(2)
    product_analysis['avg_shipment_weight'] = (product_analysis['gross_weight_tonnes'] / product_analysis['shipment_count']).round(2)
    
    # Clean up df_full_volume_value immediately after aggregations
    del df_full_volume_value, shipment_counts
    gc.collect()
    
    # Sort by total value
    product_analysis = product_analysis.sort_values('value_fob_aud', ascending=False)
    
    # Create volume and value percentiles for classification
    product_analysis['volume_percentile'] = product_analysis['gross_weight_tonnes'].rank(pct=True) * 100
    product_analysis['value_percentile'] = product_analysis['value_fob_aud'].rank(pct=True) * 100
    product_analysis['shipment_count_percentile'] = product_analysis['shipment_count'].rank(pct=True) * 100
    
    # Classify products based on volume vs value (removed medium category)
    def classify_product(row):
        volume_pct = row['volume_percentile']
        value_pct = row['value_percentile']
        
        if volume_pct >= 70 and value_pct >= 70:
            return 'High Volume - High Value'
        elif volume_pct >= 70 and value_pct <= 30:
            return 'High Volume - Low Value'
        elif volume_pct <= 30 and value_pct >= 70:
            return 'Low Volume - High Value'
        else:  # volume_pct <= 30 and value_pct <= 30
            return 'Low Volume - Low Value'
    
    product_analysis['volume_value_category'] = product_analysis.apply(classify_product, axis=1)
    
    # Focus on interesting categories
    high_volume_low_value = product_analysis[product_analysis['volume_value_category'] == 'High Volume - Low Value'].head(10)
    low_volume_high_value = product_analysis[product_analysis['volume_value_category'] == 'Low Volume - High Value'].head(10)
    high_volume_high_value = product_analysis[product_analysis['volume_value_category'] == 'High Volume - High Value'].head(10)
    low_volume_low_value = product_analysis[product_analysis['volume_value_category'] == 'Low Volume - Low Value'].head(10)
    
    # Clean presentation - show only visualizations
    
    # Define strategic categories with descriptive titles and colors
    categories = {
        'High Volume - High Value': ('#2E8B57', 'High Volume - High Value (Market Leaders)'),
        'High Volume - Low Value': ('#DC143C', 'High Volume - Low Value (Market Opportunities)'),
        'Low Volume - High Value': ('#4169E1', 'Low Volume - High Value (Premium Products)'),
        'Low Volume - Low Value': ('#FF8C00', 'Low Volume - Low Value (Niche Markets)')
    }
    
    # Create individual interactive charts for each strategic category
    for category_name, (color, title) in categories.items():
        st.subheader(f"{title}")
        
        # Get products in this category
        category_products = product_analysis[product_analysis['volume_value_category'] == category_name]
        
        if len(category_products) > 0:
            # Group by industry for cleaner visualization
            industry_summary = category_products.groupby('industry_category').agg({
                'value_fob_aud': 'sum',
                'gross_weight_tonnes': 'sum'
            }).round(2).reset_index()
            
            # Create interactive bar chart with short form value labels
            def format_short_value(value):
                if value >= 1e9:
                    return f"${value/1e9:.1f}B"
                elif value >= 1e6:
                    return f"${value/1e6:.1f}M"
                elif value >= 1e3:
                    return f"${value/1e3:.1f}K"
                else:
                    return f"${value:.0f}"
            
            fig = px.bar(industry_summary, x='value_fob_aud', y='industry_category',
                        orientation='h',
                        title=title,
                        labels={'value_fob_aud': 'Export Value (AUD)', 'industry_category': 'Industry'},
                        color='value_fob_aud',
                        color_continuous_scale=[(0, color), (1, color)])  # Use single color
            fig.update_traces(
                text=[format_short_value(value) for value in industry_summary['value_fob_aud']],
                textposition='outside',
                textfont=dict(size=10, color=color),
                hovertemplate='<b>%{y}</b><br>Export Value: $%{x:,.0f}<extra></extra>'
            )
            
            fig.update_layout(
                title_font_size=16,
                title_font_color=color,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                template='plotly_white',
                height=500,
                showlegend=False
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(tickformat='$,.0f')
            
            # Add summary information as annotation
            total_value = category_products['value_fob_aud'].sum()
            total_volume = category_products['gross_weight_tonnes'].sum()
            product_count = len(category_products)
            
            # Format numbers
            def format_number(value):
                if value >= 1e9:
                    return f"{value/1e9:.2f}B"
                elif value >= 1e6:
                    return f"{value/1e6:.2f}M"
                elif value >= 1e3:
                    return f"{value/1e3:.2f}K"
                else:
                    return f"{value:.2f}"
            
            def format_volume(value):
                if value >= 1e9:
                    return f"{value/1e9:.2f}B tonnes"
                elif value >= 1e6:
                    return f"{value/1e6:.2f}M tonnes"
                elif value >= 1e3:
                    return f"{value/1e3:.2f}K tonnes"
                else:
                    return f"{value:.2f} tonnes"
            
            # Add summary text
            summary_text = f'Total Products: {product_count}<br>Total Value: ${format_number(total_value)}<br>Total Volume: {format_volume(total_volume)}'
            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.98, y=0.02,
                showarrow=False,
                font=dict(size=12, color=color),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1
            )
            
            st.plotly_chart(fig)
        else:
            # No products in this category
            st.info(f"No products found in the {category_name} category.")
    
    # Clean presentation - no unnecessary text
    
    # 5. STATE & TRANSPORT ANALYSIS (from your notebook Cells 13-16)
    st.markdown('<h2 class="section-header">State & Transport Analysis</h2>', unsafe_allow_html=True)
    
    # State Analysis (from your notebook Cell 13)
    states = df_filtered.groupby('state_of_origin')['value_fob_aud'].sum().sort_values(ascending=False)
    states_pct = (states / states.sum() * 100).round(1)
    
    st.subheader("Export Value by State")
    
    # Clean dashboard - no unnecessary text
    
    # Create interactive state visualization with Plotly
    states_sorted = states.sort_values(ascending=False)
    states_pct_sorted = states_pct.reindex(states_sorted.index)
    
    # Prepare data for Plotly
    states_df = pd.DataFrame({
        'state': states_sorted.index,
        'value_billions': states_sorted.values / 1e9,
        'percentage': states_pct_sorted.values
    })
    
    # Create interactive horizontal bar chart
    fig = px.bar(states_df, x='value_billions', y='state',
                 orientation='h',
                 title='Export Value by State (2024-2025)',
                 labels={'value_billions': 'Export Value (Billion AUD)', 'state': 'State'},
                 color='value_billions',
                 color_continuous_scale='viridis')
    fig.update_traces(
        text=[f"${value:.1f}B<br>({pct:.1f}%)" for value, pct in zip(states_df['value_billions'], states_df['percentage'])],
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{y}</b><br>Export Value: $%{x:.1f}B<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=states_df['percentage']
    )
    
    # Update layout for better presentation
    fig.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    # Update y-axis to show states in descending order (highest at top)
    fig.update_yaxes(autorange="reversed")
    
    # Update x-axis formatting
    fig.update_xaxes(tickformat='$,.1f')
    
    st.plotly_chart(fig)
    
    # Transport Mode Analysis (from your notebook Cell 15)
    st.subheader("Export Value by Transport Mode")
    
    transport = df_filtered.groupby('mode_of_transport')['value_fob_aud'].sum().sort_values(ascending=False)
    transport_pct = (transport / transport.sum() * 100)
    
    # Clean dashboard - no unnecessary text
    
    # Create interactive transport mode visualization with Plotly
    transport_sorted = transport.sort_values(ascending=False)
    transport_pct_sorted = transport_pct.reindex(transport_sorted.index)
    
    # Prepare data for Plotly
    transport_df = pd.DataFrame({
        'transport_mode': transport_sorted.index,
        'value_billions': transport_sorted.values / 1e9,
        'percentage': transport_pct_sorted.values
    })
    
    # Create interactive vertical bar chart
    fig = px.bar(transport_df, x='transport_mode', y='value_billions',
                 title='Export Value by Transport Mode (2024-2025)',
                 labels={'value_billions': 'Export Value (Billion AUD)', 'transport_mode': 'Transport Mode'},
                 color='value_billions',
                 color_continuous_scale='viridis')
    fig.update_traces(
        text=[f"${value:.1f}B<br>({pct:.1f}%)" for value, pct in zip(transport_df['value_billions'], transport_df['percentage'])],
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{x}</b><br>Export Value: $%{y:.1f}B<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=transport_df['percentage']
    )
    
    # Update layout for better presentation
    fig.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    # Update x-axis labels rotation
    fig.update_xaxes(tickangle=15)
    
    # Update y-axis formatting
    fig.update_yaxes(tickformat='$,.1f')
    
    st.plotly_chart(fig)
    
    # 6. CUSTOM ANALYSIS (from your notebook)
    st.markdown('<h2 class="section-header">Custom Analysis</h2>', unsafe_allow_html=True)
    
    # Volume vs Value Analysis section removed as requested
    
    # Product Categories Analysis
    st.subheader("Product Categories Analysis")
    
    # Calculate value per tonne for products
    product_analysis = df_filtered.groupby('product_description').agg({
        'value_fob_aud': 'sum',
        'gross_weight_tonnes': 'sum'
    }).reset_index()
    
    product_analysis['value_per_tonne'] = product_analysis['value_fob_aud'] / product_analysis['gross_weight_tonnes']
    product_analysis = product_analysis.sort_values('value_per_tonne', ascending=False)
    
    # Clean presentation - visualizations show the data
    
    # 7. PORT ANALYSIS (from your notebook)
    st.markdown('<h2 class="section-header">Port Analysis</h2>', unsafe_allow_html=True)
    
    # Top 15 ports by tonnage
    st.subheader("Top 15 Ports by Tonnage")
    
    port_tonnage = df_filtered.groupby('port_of_loading')['gross_weight_tonnes'].sum().sort_values(ascending=False).head(15)
    
    # Create interactive lollipop chart
    port_tonnage_df = port_tonnage.reset_index()
    port_tonnage_df.columns = ['port_of_loading', 'gross_weight_tonnes']
    
    fig = px.bar(port_tonnage_df, x='gross_weight_tonnes', y='port_of_loading',
                 orientation='h',
                 title='Top 15 Ports by Tonnage',
                 labels={'gross_weight_tonnes': 'Total Tonnage', 'port_of_loading': 'Port'},
                 color='gross_weight_tonnes',
                 color_continuous_scale='viridis')
    fig.update_traces(
        text=[f"{value/1e6:.1f}M" for value in port_tonnage_df['gross_weight_tonnes']],
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{y}</b><br>Total Tonnage: %{x:,.0f} tonnes<extra></extra>'
    )
    fig.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=',.0f')
    st.plotly_chart(fig)
    
    # Port Efficiency Analysis (from your notebook) - Using FULL dataset for accuracy
    st.subheader("Port Efficiency Analysis")
    
    # Use filtered dataset for accurate port analysis to respect date range selection
    port_efficiency = df_filtered.groupby('port_of_loading').agg({
        'value_fob_aud': 'sum',
        'country_of_destination': 'count'
    }).reset_index()
    
    port_efficiency = port_efficiency.rename(columns={'country_of_destination': 'shipment_count'})
    
    # Calculate key efficiency metrics
    port_efficiency['avg_value_per_shipment'] = port_efficiency['value_fob_aud'] / port_efficiency['shipment_count']
    port_efficiency['total_value_millions'] = port_efficiency['value_fob_aud'] / 1e6
    port_efficiency['shipments_per_day'] = port_efficiency['shipment_count'] / 730  # 2 years of data
    
    # Filter for ports with significant activity (at least 50 shipments)
    significant_ports = port_efficiency[port_efficiency['shipment_count'] >= 50].copy()
    significant_ports = significant_ports.sort_values('avg_value_per_shipment', ascending=False)
    
    # High-value ports
    # Clean presentation - visualizations show the data
    
    # Port Value Visualizations (Interactive)
    st.subheader("Port Value Visualizations")
    
    # 1. TOP 15 HIGH-VALUE PORTS (Interactive)
    top_15_high_value = significant_ports.head(15)
    # Create short form value labels for high-value ports
    def format_short_value(value):
        if value >= 1e6:
            return f"${value/1e6:.1f}M"
        elif value >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.0f}"
    
    fig1 = px.bar(top_15_high_value, x='avg_value_per_shipment', y='port_of_loading',
                  orientation='h',
                  title='TOP 15 HIGH-VALUE PORTS (Average Value per Shipment)',
                  labels={'avg_value_per_shipment': 'Average Value per Shipment ($)', 'port_of_loading': 'Port'},
                  color='avg_value_per_shipment',
                  color_continuous_scale='Greens',
                  text=[format_short_value(value) for value in top_15_high_value['avg_value_per_shipment']])
    fig1.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600
    )
    fig1.update_yaxes(autorange="reversed")
    fig1.update_xaxes(tickformat='$,.0f')
    fig1.update_traces(
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{y}</b><br>Average Value per Shipment: $%{x:,.0f}<extra></extra>'
    )
    st.plotly_chart(fig1)
    
    # 2. LOWEST 15 VALUE PORTS (Interactive)
    lowest_15_value = significant_ports.nsmallest(15, 'avg_value_per_shipment')
    fig2 = px.bar(lowest_15_value, x='avg_value_per_shipment', y='port_of_loading',
                  orientation='h',
                  title='LOWEST 15 VALUE PORTS (Potential Congestion Risk)',
                  labels={'avg_value_per_shipment': 'Average Value per Shipment ($)', 'port_of_loading': 'Port'},
                  color='avg_value_per_shipment',
                  color_continuous_scale='Reds',
                  text=[format_short_value(value) for value in lowest_15_value['avg_value_per_shipment']])
    fig2.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=600
    )
    fig2.update_yaxes(autorange="reversed")
    fig2.update_xaxes(tickformat='$,.0f')
    fig2.update_traces(
        textposition='outside',
        textfont=dict(size=10, color='#2c3e50'),
        hovertemplate='<b>%{y}</b><br>Average Value per Shipment: $%{x:,.0f}<extra></extra>'
    )
    st.plotly_chart(fig2)
    
    # 8. REGIONAL ANALYSIS (EXACT from your notebook) - Using FULL dataset
    st.markdown('<h2 class="section-header">Regional Analysis</h2>', unsafe_allow_html=True)
    
    # Use filtered dataset for accurate regional analysis to respect date range selection
    df_full = df_filtered  # Use view to save memory
    
    # Add calculated fields to match your notebook
    df_full['date'] = pd.to_datetime(df_full['year'].astype(str) + '-' + df_full['month_number'].astype(str).str.zfill(2) + '-01')
    df_full['value_per_tonne'] = df_full['value_fob_aud'] / df_full['gross_weight_tonnes']
    
    # EXACT CODE FROM YOUR NOTEBOOK - Regional Mapping using region_mapping.py
    try:
        from region_mapping import add_region_to_dataframe
        # Add region column to existing dataframe
        df_full = add_region_to_dataframe(df_full, 'country_of_destination', 'region')
    except ImportError:
        # Fallback if region_mapping.py is not available
        def get_region(country):
            asia_pacific = ['China', 'Japan', 'Korea, Republic of (South)', 'Singapore', 'India', 'Taiwan', 
                           'Indonesia', 'Malaysia', 'Hong Kong', 'Vietnam', 'Thailand', 'Philippines']
            europe = ['United Kingdom, Channel Islands and Isle of Man, nfd', 'Germany', 'France', 'Italy']
            americas = ['United States of America', 'Canada', 'Brazil', 'Mexico']
            middle_east = ['United Arab Emirates', 'Saudi Arabia', 'Turkey']
            
            if country in asia_pacific:
                return 'Asia-Pacific'
            elif country in europe:
                return 'Europe'
            elif country in americas:
                return 'Americas'
            elif country in middle_east:
                return 'Middle East'
            else:
                return 'Other'
        
        # Add region column
        df_full['region'] = df_full['country_of_destination'].apply(get_region)
    
    # Check which countries are mapped to "Other" (EXACT from your notebook)
    other_countries = df_full[df_full['region'] == 'Other']['country_of_destination'].unique()
    # Clean dashboard - no unnecessary text
    
    # 2. REGIONAL EXPORT SHARE ANALYSIS (EXACT from your notebook)
    regional_analysis = df_full.groupby('region')['value_fob_aud'].sum().reset_index()
    regional_analysis['value_billions'] = regional_analysis['value_fob_aud'] / 1e9
    regional_analysis['market_share_pct'] = (regional_analysis['value_fob_aud'] / regional_analysis['value_fob_aud'].sum()) * 100
    
    # Sort by market share
    regional_analysis = regional_analysis.sort_values('market_share_pct', ascending=False)
    
    # 3. COMBINE SMALL REGIONS WITH MAIN "OTHER" CATEGORY (EXACT from your notebook)
    small_regions = regional_analysis[regional_analysis['market_share_pct'] <= 1.0]
    other_mask = regional_analysis['region'] == 'Other'
    
    if other_mask.any() and len(small_regions) > 0:
        # Add small regions to the main "Other" category
        regional_analysis.loc[other_mask, 'market_share_pct'] += small_regions['market_share_pct'].sum()
        regional_analysis.loc[other_mask, 'value_fob_aud'] += small_regions['value_fob_aud'].sum()
        regional_analysis.loc[other_mask, 'value_billions'] = regional_analysis.loc[other_mask, 'value_fob_aud'] / 1e9
        
        # Remove small regions from the data
        regional_analysis = regional_analysis[~regional_analysis.index.isin(small_regions.index)]
        
        # Clean dashboard - no unnecessary text
    
    # Re-sort after combining
    regional_analysis = regional_analysis.sort_values('market_share_pct', ascending=False)
    
    # Clean dashboard - no unnecessary text
    
    # 4. REGIONAL VISUALIZATIONS (Interactive) - EXACT from your notebook
    st.subheader("Regional Visualizations")
    
    # 4.1 Regional Market Share Pie Chart (Interactive)
    fig1 = px.pie(regional_analysis, values='market_share_pct', names='region',
                  title='Australian Export Share by Region (Combined Other Categories)',
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig1.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        template='plotly_white',
        height=500
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1)
    
    # 4.2 Regional Value Comparison Bar Chart (Interactive)
    fig2 = px.bar(regional_analysis, x='region', y='value_billions',
                  title='Export Value by Region (Billions AUD)',
                  labels={'value_billions': 'Export Value (Billions AUD)', 'region': 'Region'},
                  color='value_billions',
                  color_continuous_scale='viridis',
                  text=[f"${value:.1f}B" for value in regional_analysis['value_billions']])
    fig2.update_layout(
        title_font_size=16,
        title_font_color='#2c3e50',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template='plotly_white',
        height=500
    )
    fig2.update_xaxes(tickangle=45)
    fig2.update_yaxes(tickformat='$,.1f')
    fig2.update_traces(
        textposition='outside',
        textfont=dict(size=12, color='#2c3e50'),
        hovertemplate='<b>%{x}</b><br>Export Value: $%{y:.1f}B<extra></extra>'
    )
    st.plotly_chart(fig2)
    
    # Clean dashboard - no unnecessary text
    
    # 9. GROWING & DECLINING MARKETS ANALYSIS (EXACT from your notebook)
    try:
        st.markdown('<h2 class="section-header">Growing & Declining Markets Analysis</h2>', unsafe_allow_html=True)
        
        # Clean presentation - show only visualizations
        
        # Filter for Q1 months (January, February, March, April)
        # Handle different month column formats
        if 'month' in df_filtered.columns:
            # If month column contains full date strings like "January 2024", extract just the month name
            if df_filtered['month'].dtype == 'object':
                month_names = df_filtered['month'].astype(str).str.split().str[0]
                q1_months = ['January', 'February', 'March', 'April']
                df_q1 = df_filtered[month_names.isin(q1_months)].copy()
            else:
                # If month is already just month names
                q1_months = ['January', 'February', 'March', 'April']
                df_q1 = df_filtered[df_filtered['month'].isin(q1_months)].copy()
        elif 'month_number' in df_filtered.columns:
            # If only month_number exists, filter for Q1 (months 1-4)
            df_q1 = df_filtered[df_filtered['month_number'].isin([1, 2, 3, 4])].copy()
        else:
            df_q1 = pd.DataFrame()
            st.warning("Month information not available for Q1 filtering.")
        
        if len(df_q1) > 0 and 'year' in df_q1.columns and 'country_of_destination' in df_q1.columns:
            # Ensure year is numeric for proper grouping
            df_q1['year'] = pd.to_numeric(df_q1['year'], errors='coerce')
            
            # Calculate export values by country and year for Q1 only
            country_yearly = df_q1.groupby(['country_of_destination', 'year'])['value_fob_aud'].sum().unstack(fill_value=0)
            
            # Convert year columns to numeric if they're strings
            country_yearly.columns = pd.to_numeric(country_yearly.columns, errors='coerce')
            
            # Calculate YoY growth
            if 2024 in country_yearly.columns and 2025 in country_yearly.columns:
                # Handle division by zero
                country_yearly = country_yearly[country_yearly[2024] > 0].copy()  # Remove countries with zero 2024 value
                
                country_yearly['YoY_Growth_%'] = ((country_yearly[2025] - country_yearly[2024]) / country_yearly[2024] * 100).round(2)
                country_yearly['YoY_Growth_Absolute'] = (country_yearly[2025] - country_yearly[2024]) / 1e9  # in billions
                country_yearly['Q1_2024_Value'] = country_yearly[2024] / 1e9  # in billions
                country_yearly['Q1_2025_Value'] = country_yearly[2025] / 1e9  # in billions
                
                # Filter countries with significant trade volume (at least $100M in Q1 2024)
                significant_countries = country_yearly[country_yearly[2024] >= 1e8].copy()  # $100M threshold
                
                if len(significant_countries) > 0:
                    # Sort by YoY growth
                    significant_countries = significant_countries.sort_values('YoY_Growth_%', ascending=False)
                    
                    # Clean presentation - show only visualizations
                    
                    # 6.1 Top 15 Growing Markets (Interactive)
                    top_growing = significant_countries.head(15).reset_index()
                    fig1 = px.bar(top_growing, x='YoY_Growth_%', y='country_of_destination',
                                  orientation='h',
                                  title='Top 15 Growing Markets (Q1 2024 → Q1 2025)',
                                  labels={'YoY_Growth_%': 'YoY Growth (%)', 'country_of_destination': 'Country'},
                                  color='YoY_Growth_%',
                                  color_continuous_scale='viridis',
                                  text=[f"{value:.1f}%" for value in top_growing['YoY_Growth_%']])
                    fig1.update_layout(
                        title_font_size=16,
                        title_font_color='#2c3e50',
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        template='plotly_white',
                        height=600
                    )
                    fig1.update_yaxes(autorange="reversed")
                    fig1.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)
                    fig1.update_traces(
                        textposition='outside',
                        textfont=dict(size=10, color='#2c3e50'),
                        hovertemplate='<b>%{y}</b><br>YoY Growth: %{x:.1f}%<extra></extra>'
                    )
                    st.plotly_chart(fig1)
                    
                    # 6.2 Top 15 Declining Markets (Interactive)
                    top_declining = significant_countries.tail(15).reset_index()
                    fig2 = px.bar(top_declining, x='YoY_Growth_%', y='country_of_destination',
                                  orientation='h',
                                  title='Top 15 Declining Markets (Q1 2024 → Q1 2025)',
                                  labels={'YoY_Growth_%': 'YoY Growth (%)', 'country_of_destination': 'Country'},
                                  color='YoY_Growth_%',
                                  color_continuous_scale='plasma',
                                  text=[f"{value:.1f}%" for value in top_declining['YoY_Growth_%']])
                    fig2.update_layout(
                        title_font_size=16,
                        title_font_color='#2c3e50',
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        template='plotly_white',
                        height=600
                    )
                    fig2.update_yaxes(autorange="reversed")
                    fig2.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)
                    fig2.update_traces(
                        textposition='outside',
                        textfont=dict(size=10, color='#2c3e50'),
                        hovertemplate='<b>%{y}</b><br>YoY Growth: %{x:.1f}%<extra></extra>'
                    )
                    st.plotly_chart(fig2)
                else:
                    st.warning("No countries meet the significant trade volume threshold ($100M in Q1 2024) for growth analysis.")
            else:
                st.warning("Data for both 2024 and 2025 Q1 periods is required for YoY growth analysis. Current data may only cover one year.")
        else:
            st.warning("Insufficient data available for Q1 analysis. Please check your date range filters.")
    except Exception as e:
        st.error(f"Error in Growing & Declining Markets Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 10. EXECUTIVE SUMMARY (from your notebook Cell 22)
    st.markdown('<h2 class="section-header">Executive Summary</h2>', unsafe_allow_html=True)
    
    # Generate executive summary report (from your notebook)
    st.markdown("""
    <div class="summary-box">
        <h3>EXECUTIVE SUMMARY - AUSTRALIAN FREIGHT EXPORTS 2024-2025</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Overview")
        # Clean dashboard - no unnecessary text
    
    with col2:
        st.subheader("Top 3 Destinations")
        for i, (country, value) in enumerate(top_countries['Total_Value'].head(3).items(), 1):
            st.write(f"{i}. {country}: ${value/1e9:.2f}B")
    
    # Footer
    st.markdown("---")
    st.markdown("**Australian Freight Export Analysis Dashboard** | **Data Source:** Australian Bureau of Statistics | **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

else:
    st.error("Unable to load data. Please check your data file and try again.")

