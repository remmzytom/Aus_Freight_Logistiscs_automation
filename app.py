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
    - If missing: generate immediately
    - If older than 30 days: regenerate from ABS website
    Returns the relative path to the cleaned CSV.
    """
    import os
    import time
    os.makedirs('data', exist_ok=True)

    cleaned_path = 'data/exports_cleaned.csv'
    
    # Check if file exists and if it needs refresh (>30 days old) or is missing required columns
    needs_regeneration = False
    if os.path.exists(cleaned_path):
        # Check file age
        file_age_days = (time.time() - os.path.getmtime(cleaned_path)) / (60 * 60 * 24)
        if file_age_days > 30:
            st.info(f"🔄 Data is {int(file_age_days)} days old. Refreshing from ABS website...")
            needs_regeneration = True
        else:
            # Check if file has required columns (month_number might be missing from old files)
            try:
                import pandas as pd
                sample_df = pd.read_csv(cleaned_path, nrows=1)
                if 'month_number' not in sample_df.columns:
                    st.info("🔄 Updating data format (adding missing columns)...")
                    needs_regeneration = True
                elif not needs_regeneration:
                    # File exists, is fresh, and has required columns - return it
                    return cleaned_path
            except Exception:
                # File might be corrupted, regenerate
                needs_regeneration = True
    else:
        # File doesn't exist - need to generate
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

    # Minimal cleaning compatible with dashboard expectations
    import pandas as pd
    for col in ['quantity', 'gross_weight_tonnes', 'value_fob_aud']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'month' in df.columns and 'year' not in df.columns:
        df['year'] = df['month'].astype(str).str.extract(r'(\d{4})')
    
    # Create month_number column (required by dashboard)
    if 'month' in df.columns and 'month_number' not in df.columns:
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        # Extract month name from month column (format: "January 2024")
        df['month_name'] = df['month'].astype(str).str.split().str[0]
        df['month_number'] = df['month_name'].map(month_map).fillna(1)
        df = df.drop(columns=['month_name'], errors='ignore')

    df.to_csv(cleaned_path, index=False)
    st.success("✅ Data loaded and processed successfully!")
    return cleaned_path


@st.cache_data(ttl=3600, show_spinner=False)
def load_exports_cleaned(path: str) -> pd.DataFrame:
    """Load ALL data efficiently using chunked processing with aggressive memory management."""
    import os
    
    # Check if running on Streamlit Cloud for optimized chunk sizes
    is_streamlit_cloud = (
        os.environ.get('STREAMLIT_SHARING', '').lower() == 'true' or
        'streamlit.app' in os.environ.get('SERVER_NAME', '') or
        os.path.exists('/app')
    )
    
    # Use smaller chunks for cloud to reduce peak memory usage
    chunk_size = 30000 if is_streamlit_cloud else 100000
    chunks = []
    df_combined = None
    
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
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
            
            # Derived fields (before downcasting to preserve precision)
            if 'year' in chunk.columns and 'month_number' in chunk.columns:
                chunk['date'] = pd.to_datetime(chunk['year'].astype(str) + '-' + chunk['month_number'].astype(str).str.zfill(2) + '-01', errors='coerce')
            if {'value_fob_aud','gross_weight_tonnes'}.issubset(chunk.columns):
                chunk['value_per_tonne'] = (chunk['value_fob_aud'] / chunk['gross_weight_tonnes']).replace([np.inf, -np.inf], np.nan)
            
            # Aggressively downcast numerics to save memory (critical for large datasets)
            for col in chunk.select_dtypes(include=['int64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='integer', errors='ignore')
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='float', errors='ignore')
            
            chunks.append(chunk)
            
            # Combine chunks in smaller batches on cloud to reduce memory spikes
            batch_size = 2 if is_streamlit_cloud else 5
            if len(chunks) >= batch_size:
                if df_combined is None:
                    df_combined = pd.concat(chunks, ignore_index=True)
                else:
                    df_combined = pd.concat([df_combined] + chunks, ignore_index=True)
                chunks = []
                # Aggressive memory cleanup
                import gc
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


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=600)
def load_kpis_only():
    """Load only KPIs (no full dataset) - efficient for memory."""
    try:
        file_path = ensure_data_file()
        accurate_kpis = compute_kpis_chunked(file_path)
        return accurate_kpis
    except Exception as e:
        st.error(f"Error computing KPIs: {str(e)}")
        return None


@st.cache_data(ttl=600)
def load_section_data_chunked(file_path: str, section: str, filters: dict = None):
    """Lazy load data for specific sections - processes chunks and returns only what's needed."""
    import os
    
    is_streamlit_cloud = (
        os.environ.get('STREAMLIT_SHARING', '').lower() == 'true' or
        'streamlit.app' in os.environ.get('SERVER_NAME', '') or
        os.path.exists('/app')
    )
    
    chunk_size = 30000 if is_streamlit_cloud else 100000
    
    try:
        # Process chunks based on section needs
        if section == 'time_series':
            # Aggregate by month - much smaller result
            monthly_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # Apply compatibility shims
                if 'month_number' not in chunk.columns and 'month' in chunk.columns:
                    month_map = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12,
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    chunk['month_name'] = chunk['month'].astype(str).str.split().str[0]
                    chunk['month_number'] = chunk['month_name'].map(month_map).fillna(1)
                    chunk = chunk.drop(columns=['month_name'], errors='ignore')
                
                # Aggregate by year, month_number, month
                if {'year', 'month_number', 'month', 'value_fob_aud', 'gross_weight_tonnes'}.issubset(chunk.columns):
                    grouped = chunk.groupby(['year', 'month_number', 'month']).agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        key = (row['year'], row['month_number'], row['month'])
                        if key not in monthly_data:
                            monthly_data[key] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0}
                        monthly_data[key]['value_fob_aud'] += row['value_fob_aud']
                        monthly_data[key]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                
                del chunk
                gc.collect()
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {'year': k[0], 'month_number': k[1], 'month': k[2], 
                 'value_fob_aud': v['value_fob_aud'], 'gross_weight_tonnes': v['gross_weight_tonnes']}
                for k, v in monthly_data.items()
            ])
            df = df.sort_values(['year', 'month_number'])
            if len(df) > 0:
                df['value_per_tonne'] = df['value_fob_aud'] / df['gross_weight_tonnes']
            return df
        
        elif section == 'country_analysis':
            # Aggregate by country
            country_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                if 'country_of_destination' in chunk.columns:
                    grouped = chunk.groupby('country_of_destination').agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        country = row['country_of_destination']
                        if country not in country_data:
                            country_data[country] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0}
                        country_data[country]['value_fob_aud'] += row['value_fob_aud']
                        country_data[country]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'country_of_destination': k, 'value_fob_aud': v['value_fob_aud'], 
                 'gross_weight_tonnes': v['gross_weight_tonnes']}
                for k, v in country_data.items()
            ])
            return df.sort_values('value_fob_aud', ascending=False)
        
        elif section == 'product_analysis':
            # Aggregate by product
            product_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # Ensure product_description exists
                if 'product_description' not in chunk.columns:
                    import re
                    def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
                    candidates = [c for c in chunk.columns if _norm(c) in {'productdescription', 'product_description', 'sitc'}]
                    if candidates:
                        chunk['product_description'] = chunk[candidates[0]].astype(str)
                    else:
                        chunk['product_description'] = 'All Products'
                
                if 'product_description' in chunk.columns:
                    grouped = chunk.groupby('product_description').agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        product = row['product_description']
                        if product not in product_data:
                            product_data[product] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0}
                        product_data[product]['value_fob_aud'] += row['value_fob_aud']
                        product_data[product]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'product_description': k, 'value_fob_aud': v['value_fob_aud'], 
                 'gross_weight_tonnes': v['gross_weight_tonnes']}
                for k, v in product_data.items()
            ])
            return df.sort_values('value_fob_aud', ascending=False)
        
        elif section == 'industry_analysis':
            # Aggregate by industry category
            industry_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # Ensure prod_descpt_code exists
                if 'prod_descpt_code' not in chunk.columns and 'sitc_code' in chunk.columns:
                    chunk['prod_descpt_code'] = chunk['sitc_code'].astype(str)
                elif 'prod_descpt_code' not in chunk.columns:
                    chunk['prod_descpt_code'] = ''
                
                # Map to industry category
                def get_industry_category(sitc_code):
                    if pd.isna(sitc_code) or sitc_code == '':
                        return 'Other Commodities'
                    first_digit = str(sitc_code).strip()[0] if len(str(sitc_code).strip()) >= 1 else '9'
                    mapping = {'0': 'Food & Agriculture', '1': 'Beverages & Tobacco', '2': 'Raw Materials & Mining', 
                               '3': 'Energy & Petroleum', '4': 'Food Processing', '5': 'Chemicals & Pharmaceuticals',
                               '6': 'Manufactured Goods and materials', '7': 'Machinery & Equipment', 
                               '8': 'Consumer Goods', '9': 'Other Commodities'}
                    return mapping.get(first_digit, 'Other Commodities')
                
                chunk['industry_category'] = chunk['prod_descpt_code'].apply(get_industry_category)
                
                grouped = chunk.groupby('industry_category').agg({
                    'value_fob_aud': 'sum',
                    'gross_weight_tonnes': 'sum'
                }).reset_index()
                
                for _, row in grouped.iterrows():
                    industry = row['industry_category']
                    if industry not in industry_data:
                        industry_data[industry] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0}
                    industry_data[industry]['value_fob_aud'] += row['value_fob_aud']
                    industry_data[industry]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'industry_category': k, 'value_fob_aud': v['value_fob_aud'], 
                 'gross_weight_tonnes': v['gross_weight_tonnes']}
                for k, v in industry_data.items()
            ])
            if len(df) > 0:
                df['value_per_tonne'] = df['value_fob_aud'] / df['gross_weight_tonnes']
            return df.sort_values('value_fob_aud', ascending=False)
        
        elif section == 'state_analysis':
            # Aggregate by state
            state_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                if 'state_of_origin' in chunk.columns:
                    grouped = chunk.groupby('state_of_origin').agg({
                        'value_fob_aud': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        state = row['state_of_origin']
                        if state not in state_data:
                            state_data[state] = {'value_fob_aud': 0}
                        state_data[state]['value_fob_aud'] += row['value_fob_aud']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'state_of_origin': k, 'value_fob_aud': v['value_fob_aud']}
                for k, v in state_data.items()
            ])
            return df.sort_values('value_fob_aud', ascending=False)
        
        elif section == 'transport_analysis':
            # Aggregate by transport mode
            transport_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                if 'mode_of_transport' in chunk.columns:
                    grouped = chunk.groupby('mode_of_transport').agg({
                        'value_fob_aud': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        transport = row['mode_of_transport']
                        if transport not in transport_data:
                            transport_data[transport] = {'value_fob_aud': 0}
                        transport_data[transport]['value_fob_aud'] += row['value_fob_aud']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'mode_of_transport': k, 'value_fob_aud': v['value_fob_aud']}
                for k, v in transport_data.items()
            ])
            return df.sort_values('value_fob_aud', ascending=False)
        
        elif section == 'port_analysis':
            # Aggregate by port
            port_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                if 'port_of_loading' in chunk.columns:
                    grouped = chunk.groupby('port_of_loading').agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum'
                    }).reset_index()
                    
                    # Count shipments by port
                    port_counts = chunk.groupby('port_of_loading').size()
                    for port, count in port_counts.items():
                        if port not in port_data:
                            port_data[port] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0, 'shipment_count': 0}
                        port_data[port]['shipment_count'] += count
                    
                    for _, row in grouped.iterrows():
                        port = row['port_of_loading']
                        if port not in port_data:
                            port_data[port] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0, 'shipment_count': 0}
                        port_data[port]['value_fob_aud'] += row['value_fob_aud']
                        port_data[port]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'port_of_loading': k, 'value_fob_aud': v['value_fob_aud'], 
                 'gross_weight_tonnes': v['gross_weight_tonnes'], 'shipment_count': v['shipment_count']}
                for k, v in port_data.items()
            ])
            if len(df) > 0:
                df['avg_value_per_shipment'] = df['value_fob_aud'] / df['shipment_count']
            return df.sort_values('gross_weight_tonnes', ascending=False)
        
        elif section == 'regional_analysis':
            # Aggregate by region
            regional_data = {}
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                if 'country_of_destination' in chunk.columns:
                    # Map country to region
                    try:
                        from region_mapping import add_region_to_dataframe
                        chunk = add_region_to_dataframe(chunk, 'country_of_destination', 'region')
                    except ImportError:
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
                        chunk['region'] = chunk['country_of_destination'].apply(get_region)
                    
                    grouped = chunk.groupby('region').agg({
                        'value_fob_aud': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        region = row['region']
                        if region not in regional_data:
                            regional_data[region] = {'value_fob_aud': 0}
                        regional_data[region]['value_fob_aud'] += row['value_fob_aud']
                
                del chunk
                gc.collect()
            
            df = pd.DataFrame([
                {'region': k, 'value_fob_aud': v['value_fob_aud']}
                for k, v in regional_data.items()
            ])
            if len(df) > 0:
                total_value = df['value_fob_aud'].sum()
                df['market_share_pct'] = (df['value_fob_aud'] / total_value * 100)
            return df.sort_values('value_fob_aud', ascending=False)
        
        elif section == 'product_market_analysis':
            # Aggregate product-country combinations
            product_country_data = {}
            product_data = {}
            country_data = {}
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # Ensure product_description exists
                if 'product_description' not in chunk.columns:
                    import re
                    def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
                    candidates = [c for c in chunk.columns if _norm(c) in {'productdescription', 'product_description', 'sitc'}]
                    if candidates:
                        chunk['product_description'] = chunk[candidates[0]].astype(str)
                    else:
                        chunk['product_description'] = 'All Products'
                
                # Products
                if 'product_description' in chunk.columns:
                    grouped_products = chunk.groupby('product_description').agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum',
                        'country_of_destination': 'nunique'
                    }).reset_index()
                    for _, row in grouped_products.iterrows():
                        product = row['product_description']
                        if product not in product_data:
                            product_data[product] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0, 'countries_served': 0}
                        product_data[product]['value_fob_aud'] += row['value_fob_aud']
                        product_data[product]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                        product_data[product]['countries_served'] = max(product_data[product]['countries_served'], row['country_of_destination'])
                
                # Countries
                if 'country_of_destination' in chunk.columns:
                    grouped_countries = chunk.groupby('country_of_destination').agg({
                        'value_fob_aud': 'sum',
                        'gross_weight_tonnes': 'sum',
                        'product_description': 'nunique'
                    }).reset_index()
                    for _, row in grouped_countries.iterrows():
                        country = row['country_of_destination']
                        if country not in country_data:
                            country_data[country] = {'value_fob_aud': 0, 'gross_weight_tonnes': 0, 'products_imported': 0}
                        country_data[country]['value_fob_aud'] += row['value_fob_aud']
                        country_data[country]['gross_weight_tonnes'] += row['gross_weight_tonnes']
                        country_data[country]['products_imported'] = max(country_data[country]['products_imported'], row['product_description'])
                
                del chunk
                gc.collect()
            
            return {
                'products': pd.DataFrame([
                    {'product_description': k, 'value_fob_aud': v['value_fob_aud'], 
                     'gross_weight_tonnes': v['gross_weight_tonnes'], 'countries_served': v['countries_served']}
                    for k, v in product_data.items()
                ]).sort_values('value_fob_aud', ascending=False),
                'countries': pd.DataFrame([
                    {'country_of_destination': k, 'value_fob_aud': v['value_fob_aud'], 
                     'gross_weight_tonnes': v['gross_weight_tonnes'], 'products_imported': v['products_imported']}
                    for k, v in country_data.items()
                ]).sort_values('value_fob_aud', ascending=False)
            }
        
        elif section == 'growing_declining_markets':
            # Q1 comparison by country
            country_yearly_data = {}
            q1_months = ['January', 'February', 'March', 'April']
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # Filter Q1 months
                if 'month' in chunk.columns:
                    chunk_q1 = chunk[chunk['month'].isin(q1_months)]
                else:
                    chunk_q1 = chunk
                
                if 'country_of_destination' in chunk_q1.columns and 'year' in chunk_q1.columns:
                    grouped = chunk_q1.groupby(['country_of_destination', 'year']).agg({
                        'value_fob_aud': 'sum'
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        country = row['country_of_destination']
                        year = row['year']
                        if country not in country_yearly_data:
                            country_yearly_data[country] = {2024: 0, 2025: 0}
                        if year in country_yearly_data[country]:
                            country_yearly_data[country][year] += row['value_fob_aud']
                
                del chunk, chunk_q1
                gc.collect()
            
            # Calculate growth
            growth_data = []
            for country, years in country_yearly_data.items():
                val_2024 = years.get(2024, 0)
                val_2025 = years.get(2025, 0)
                if val_2024 >= 1e8:  # At least $100M in 2024
                    growth_pct = ((val_2025 - val_2024) / val_2024 * 100) if val_2024 > 0 else 0
                    growth_data.append({
                        'country_of_destination': country,
                        'Q1_2024_Value': val_2024 / 1e9,
                        'Q1_2025_Value': val_2025 / 1e9,
                        'YoY_Growth_%': growth_pct,
                        'YoY_Growth_Absolute': (val_2025 - val_2024) / 1e9
                    })
            
            df = pd.DataFrame(growth_data)
            return df.sort_values('YoY_Growth_%', ascending=False)
        
        # Default: return empty for unsupported sections
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading {section} data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


# Load only KPIs initially (lightweight)
try:
    with st.spinner('Computing key metrics...'):
        accurate_kpis = load_kpis_only()
except Exception as e:
    st.error(f"Failed to load KPIs: {str(e)}")
    st.stop()

if accurate_kpis is not None:
    # Get file path for lazy loading
    try:
        file_path = ensure_data_file()
    except Exception as e:
        st.error(f"Failed to ensure data file: {str(e)}")
        st.stop()
    
    # Store file_path in session state for lazy loading
    if 'data_file_path' not in st.session_state:
        st.session_state.data_file_path = file_path
    
    # Helper function to get filter options (loads minimal metadata)
    @st.cache_data(ttl=3600)
    def get_filter_options(file_path: str):
        """Get unique values for filters without loading full dataset."""
        countries = set()
        products = set()
        min_date = None
        max_date = None
        
        chunk_size = 50000
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, usecols=[
            'country_of_destination', 'product_description', 'month', 'year'
        ] if all(c in pd.read_csv(file_path, nrows=0).columns for c in ['country_of_destination', 'product_description', 'month', 'year']) 
        else None):
            if 'country_of_destination' in chunk.columns:
                countries.update(chunk['country_of_destination'].dropna().unique())
            if 'product_description' in chunk.columns:
                products.update(chunk['product_description'].dropna().unique())
            # Get date range
            if 'year' in chunk.columns and 'month' in chunk.columns:
                chunk_years = chunk['year'].dropna().unique()
                if len(chunk_years) > 0:
                    if min_date is None or min(chunk_years) < min_date:
                        min_date = min(chunk_years)
                    if max_date is None or max(chunk_years) > max_date:
                        max_date = max(chunk_years)
            
            if len(countries) > 1000:  # Limit to prevent memory issues
                break
        
        return {
            'countries': sorted(list(countries))[:500],  # Limit to top 500
            'products': sorted(list(products))[:500],
            'min_year': int(min_date) if min_date else 2024,
            'max_year': int(max_date) if max_date else 2025
        }
    
    # Get filter options
    filter_options = get_filter_options(file_path)
    
    # Sidebar controls (work without full dataset)
    st.sidebar.header("Dashboard Controls")
    
    # Date range filter (using year only for simplicity)
    st.sidebar.subheader("Date Range")
    year_range = st.sidebar.select_slider(
        "Select Year",
        options=list(range(filter_options['min_year'], filter_options['max_year'] + 1)),
        value=(filter_options['min_year'], filter_options['max_year'])
    )
    
    # Country filter
    st.sidebar.subheader("Country Filter")
    all_countries = ['All Countries'] + filter_options['countries']
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=all_countries[:100],  # Limit dropdown size
        default=['All Countries']
    )
    
    # Product filter
    st.sidebar.subheader("Product Filter")
    all_products = ['All Products'] + filter_options['products']
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=all_products[:100],  # Limit dropdown size
        default=['All Products']
    )
    
    # Store filters for lazy loading
    filters = {
        'year_range': year_range,
        'countries': selected_countries if 'All Countries' not in selected_countries else [],
        'products': selected_products if 'All Products' not in selected_products else []
    }
    
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
    
    # Main dashboard content
    
    # 1. DATASET SUMMARY (from your notebook Cell 4) - Uses KPIs computed from full dataset
    st.markdown('<h2 class="section-header">Dataset Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Records</h3>
            <h2>{accurate_kpis['total_records']:,}</h2>
            <p>Individual Shipments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Countries</h3>
            <h2>{len(filter_options['countries'])}</h2>
            <p>Export Destinations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Products</h3>
            <h2>{len(filter_options['products'])}</h2>
            <p>Product Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Date Range</h3>
            <h2>{filter_options['max_year'] - filter_options['min_year'] + 1}</h2>
            <p>Years of Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Financial Summary (from your notebook Cell 4) - Uses KPIs from full dataset
    st.markdown('<h2 class="section-header">Financial Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = accurate_kpis['total_value']
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
            <p>Australian Dollars (Full Dataset)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_shipment = accurate_kpis['avg_shipment_value']
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
            <p>Australian Dollars</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        median_shipment = accurate_kpis['median_shipment_value']
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
            <p>Australian Dollars</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_weight = accurate_kpis['total_weight']
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
            <p>Tonnes</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. TIME SERIES ANALYSIS - Lazy loaded from chunks
    try:
        st.markdown('<h2 class="section-header">Time Series Analysis</h2>', unsafe_allow_html=True)
        
        # Lazy load time series data (aggregated by month)
        with st.spinner('Loading time series data...'):
            monthly = load_section_data_chunked(file_path, 'time_series', filters)
            
            # Check if we have data
            if len(monthly) > 0:
                monthly['period'] = monthly['month'] + ' ' + monthly['year'].astype(str)
                # value_per_tonne already computed in load_section_data_chunked
                
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
    except Exception as e:
        st.error(f"Error in Time Series Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Clean up memory after time series section
    gc.collect()
    
    # Clean presentation - no redundant tables
    
    # 3. COUNTRY ANALYSIS - Lazy loaded from chunks
    st.markdown('<h2 class="section-header">Country Analysis</h2>', unsafe_allow_html=True)
    
    # Lazy load country data (aggregated by country)
    with st.spinner('Loading country analysis data...'):
        top_countries_df = load_section_data_chunked(file_path, 'country_analysis', filters)
        top_countries = top_countries_df.set_index('country_of_destination')
    
    # Check if we have data for the selected date range
    if len(top_countries) > 0:
        top_countries['value_billions'] = top_countries['value_fob_aud'] / 1e9
        top_countries['pct'] = (top_countries['value_fob_aud'] / top_countries['value_fob_aud'].sum() * 100)
    else:
        st.warning("No data available for the selected date range. Please adjust your date filter.")
    
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
    
    # 4. PRODUCT ANALYSIS - Lazy loaded from chunks
    st.markdown('<h2 class="section-header">Product Analysis</h2>', unsafe_allow_html=True)
    
    # Lazy load product data (aggregated by product)
    with st.spinner('Loading product analysis data...'):
        top_products_df = load_section_data_chunked(file_path, 'product_analysis', filters)
        top_products = top_products_df.set_index('product_description')

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
    else:
        st.warning("No data available for the selected date range. Please adjust your date filter.")

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
    
    # 4.5. INDUSTRY CATEGORY ANALYSIS - Lazy loaded
    try:
        st.markdown('<h2 class="section-header">Industry Category Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner('Loading industry analysis data...'):
            industry_df = load_section_data_chunked(file_path, 'industry_analysis', filters)
        
        if len(industry_df) > 0:
            industry_df['Value_Percentage'] = (industry_df['value_fob_aud'] / industry_df['value_fob_aud'].sum() * 100).round(1)
            industry_df['value_billions'] = industry_df['value_fob_aud'] / 1e9
            
            # Top industries chart
            top_industries = industry_df.head(8).reset_index()
            
            fig1 = px.bar(top_industries, x='value_billions', y='industry_category',
                         orientation='h',
                         title='Australian Export Value by Industry Category',
                         labels={'value_billions': 'Export Value (Billion AUD)', 'industry_category': 'Industry'},
                         color='value_billions',
                         color_continuous_scale='viridis')
            fig1.update_traces(
                text=[f"${value:.1f}B<br>({pct:.1f}%)" for value, pct in zip(top_industries['value_billions'], top_industries['Value_Percentage'])],
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
            fig1.update_xaxes(tickformat='$,.1f')
            st.plotly_chart(fig1)
            
            # Value density chart
            industry_df['Value_per_Tonne_Ratio'] = industry_df['value_fob_aud'] / industry_df['gross_weight_tonnes']
            value_density = industry_df.sort_values('Value_per_Tonne_Ratio', ascending=False).reset_index()
            
            st.subheader("Industry Value Density (Value per Tonne Shipped)")
            fig2 = px.bar(value_density, x='industry_category', y='Value_per_Tonne_Ratio',
                         title='Industry Value Density (Value per Tonne Shipped)',
                         labels={'Value_per_Tonne_Ratio': 'Value per Tonne (AUD)', 'industry_category': 'Industry'},
                         color='Value_per_Tonne_Ratio',
                         color_continuous_scale='viridis')
            fig2.update_traces(
                text=[f"${value:,.0f}/tonne" for value in value_density['Value_per_Tonne_Ratio']],
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
            st.plotly_chart(fig2)
            
        gc.collect()
    except Exception as e:
        st.error(f"Error in Industry Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 4.6. PRODUCT-MARKET ANALYSIS - Lazy loaded
    try:
        st.markdown('<h2 class="section-header">Product-Market Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner('Loading product-market analysis data...'):
            pm_data = load_section_data_chunked(file_path, 'product_market_analysis', filters)
        
        if isinstance(pm_data, dict) and 'products' in pm_data and 'countries' in pm_data:
            top_products_pm = pm_data['products'].head(10)
            top_countries_pm = pm_data['countries'].head(10)
            
            # Top products
            st.subheader("Top 10 Export Products by Value")
            fig1 = px.bar(top_products_pm, x='value_fob_aud', y='product_description',
                         orientation='h',
                         title='TOP 10 EXPORT PRODUCTS BY VALUE',
                         labels={'value_fob_aud': 'Export Value (AUD)', 'product_description': 'Product'},
                         color='value_fob_aud',
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
            
            # Top countries
            st.subheader("Top 10 Destination Countries")
            fig2 = px.bar(top_countries_pm, x='value_fob_aud', y='country_of_destination',
                         orientation='h',
                         title='TOP 10 DESTINATION COUNTRIES',
                         labels={'value_fob_aud': 'Import Value (AUD)', 'country_of_destination': 'Country'},
                         color='value_fob_aud',
                         color_continuous_scale='plasma')
            fig2.update_traces(
                text=[f"${value/1e9:.1f}B" for value in top_countries_pm['value_fob_aud']],
                textposition='outside'
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
            
        gc.collect()
    except Exception as e:
        st.error(f"Error in Product-Market Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 4.7. PORT ANALYSIS - Lazy loaded
    try:
        st.markdown('<h2 class="section-header">Port Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner('Loading port analysis data...'):
            port_df = load_section_data_chunked(file_path, 'port_analysis', filters)
        
        if len(port_df) > 0:
            # Top ports by tonnage
            top_15_ports = port_df.head(15).reset_index()
            top_15_ports['tonnage_millions'] = top_15_ports['gross_weight_tonnes'] / 1e6
            
            st.subheader("Top 15 Ports by Tonnage")
            fig = px.bar(top_15_ports, x='tonnage_millions', y='port_of_loading',
                        orientation='h',
                        title='TOP 15 PORTS BY TONNAGE',
                        labels={'tonnage_millions': 'Tonnage (Million Tonnes)', 'port_of_loading': 'Port'},
                        color='tonnage_millions',
                        color_continuous_scale='viridis')
            fig.update_traces(
                text=[f"{value:.1f}M" for value in top_15_ports['tonnage_millions']],
                textposition='outside'
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
            fig.update_xaxes(tickformat=',.1f')
            st.plotly_chart(fig)
            
            # Port efficiency
            significant_ports = port_df[port_df['shipment_count'] >= 50].copy()
            significant_ports = significant_ports.sort_values('avg_value_per_shipment', ascending=False)
            
            if len(significant_ports) > 0:
                st.subheader("Port Value Visualizations")
                
                top_15_high_value = significant_ports.head(15)
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
                fig1.update_traces(textposition='outside')
                st.plotly_chart(fig1)
            
        gc.collect()
    except Exception as e:
        st.error(f"Error in Port Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 5. STATE & TRANSPORT ANALYSIS - Lazy loaded
    try:
        st.markdown('<h2 class="section-header">State & Transport Analysis</h2>', unsafe_allow_html=True)
        
        # State Analysis
        with st.spinner('Loading state analysis data...'):
            state_df = load_section_data_chunked(file_path, 'state_analysis', filters)
        
        if len(state_df) > 0:
            state_df['value_billions'] = state_df['value_fob_aud'] / 1e9
            state_df['percentage'] = (state_df['value_fob_aud'] / state_df['value_fob_aud'].sum() * 100).round(1)
            
            st.subheader("Export Value by State")
            fig = px.bar(state_df, x='value_billions', y='state_of_origin',
                        orientation='h',
                        title='Export Value by State (2024-2025)',
                        labels={'value_billions': 'Export Value (Billion AUD)', 'state_of_origin': 'State'},
                        color='value_billions',
                        color_continuous_scale='viridis')
            fig.update_traces(
                text=[f"${value:.1f}B<br>({pct:.1f}%)" for value, pct in zip(state_df['value_billions'], state_df['percentage'])],
                textposition='outside'
            )
            fig.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                template='plotly_white',
                height=500,
                showlegend=False
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(tickformat='$,.1f')
            st.plotly_chart(fig)
        
        # Transport Analysis
        with st.spinner('Loading transport analysis data...'):
            transport_df = load_section_data_chunked(file_path, 'transport_analysis', filters)
        
        if len(transport_df) > 0:
            transport_df['value_billions'] = transport_df['value_fob_aud'] / 1e9
            transport_df['percentage'] = (transport_df['value_fob_aud'] / transport_df['value_fob_aud'].sum() * 100).round(1)
            
            st.subheader("Export Value by Transport Mode")
            fig = px.bar(transport_df, x='mode_of_transport', y='value_billions',
                        title='Export Value by Transport Mode (2024-2025)',
                        labels={'value_billions': 'Export Value (Billion AUD)', 'mode_of_transport': 'Transport Mode'},
                        color='value_billions',
                        color_continuous_scale='viridis')
            fig.update_traces(
                text=[f"${value:.1f}B<br>({pct:.1f}%)" for value, pct in zip(transport_df['value_billions'], transport_df['percentage'])],
                textposition='outside'
            )
            fig.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                template='plotly_white',
                height=500,
                showlegend=False
            )
            fig.update_xaxes(tickangle=15)
            fig.update_yaxes(tickformat='$,.1f')
            st.plotly_chart(fig)
        
        gc.collect()
    except Exception as e:
        st.error(f"Error in State & Transport Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 6. REGIONAL ANALYSIS - Lazy loaded
    try:
        st.markdown('<h2 class="section-header">Regional Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner('Loading regional analysis data...'):
            regional_df = load_section_data_chunked(file_path, 'regional_analysis', filters)
        
        if len(regional_df) > 0:
            regional_df['value_billions'] = regional_df['value_fob_aud'] / 1e9
            
            # Pie chart
            st.subheader("Regional Market Share")
            fig1 = px.pie(regional_df, values='market_share_pct', names='region',
                         title='Australian Export Share by Region',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig1.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                template='plotly_white',
                height=500
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1)
            
            # Bar chart
            st.subheader("Regional Value Comparison")
            fig2 = px.bar(regional_df, x='region', y='value_billions',
                         title='Export Value by Region (Billions AUD)',
                         labels={'value_billions': 'Export Value (Billions AUD)', 'region': 'Region'},
                         color='value_billions',
                         color_continuous_scale='viridis',
                         text=[f"${value:.1f}B" for value in regional_df['value_billions']])
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
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2)
            
        gc.collect()
    except Exception as e:
        st.error(f"Error in Regional Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 7. GROWING & DECLINING MARKETS - Lazy loaded
    try:
        st.markdown('<h2 class="section-header">Growing & Declining Markets Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner('Loading growth analysis data...'):
            growth_df = load_section_data_chunked(file_path, 'growing_declining_markets', filters)
        
        if len(growth_df) > 0:
            # Top growing markets
            top_growing = growth_df.head(15).reset_index()
            
            st.subheader("Top 15 Growing Markets (Q1 2024 → Q1 2025)")
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
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1)
            
            # Top declining markets
            top_declining = growth_df.tail(15).reset_index()
            
            st.subheader("Top 15 Declining Markets (Q1 2024 → Q1 2025)")
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
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2)
            
        gc.collect()
    except Exception as e:
        st.error(f"Error in Growing & Declining Markets Analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Footer - Show footer
    st.markdown("---")
    st.markdown("**Australian Freight Export Analysis Dashboard** | **Data Source:** Australian Bureau of Statistics | **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Note: Additional sections (Industry, State, Transport, Port, Regional analyses) are being optimized
    # for lazy loading to handle large datasets efficiently on Streamlit Cloud.

else:
    st.error("Unable to load data. Please check your data file and try again.")

# Note: Additional sections (Industry, State, Transport, Port, Regional analyses) will be
# restored with lazy loading implementations for memory-efficient processing on Streamlit Cloud.
