"""Shared data pipeline for generating cleaned export dataset."""
from __future__ import annotations

import importlib.util
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import numpy as np

# Type alias for optional callback dict
CallbackMap = Optional[Dict[str, Callable[[str], None]]]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_EXPORTS_PATH = DATA_DIR / "exports_2024_2025.csv"
CLEAN_EXPORTS_PATH = DATA_DIR / "exports_cleaned.csv"
EXTRACTOR_PATH = PROJECT_ROOT / "2024_2025_extractor.py"


def _emit(callbacks: CallbackMap, level: str, message: str) -> None:
    """Send message to callback if provided, otherwise print."""
    func = None
    if callbacks:
        func = callbacks.get(level)
    if callable(func):
        func(message)
    else:
        # Fallback to simple print for automation/logging purposes
        print(f"[{level.upper()}] {message}")


def _load_extractor() -> Optional[Callable[[], pd.DataFrame]]:
    """Dynamically load the extractor function if available."""
    if not EXTRACTOR_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("extractor_module", str(EXTRACTOR_PATH))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return getattr(module, "extract_2024_2025", None)
    return None


def _download_raw_data(callbacks: CallbackMap) -> pd.DataFrame:
    """Fetch raw export data (2024-2025) using the shared extractor."""
    extractor = _load_extractor()
    df = None

    if callable(extractor):
        _emit(callbacks, "info", "Downloading raw export data from ABS (2024-2025)...")
        try:
            df = extractor()
        except Exception as exc:  # pragma: no cover - defensive
            _emit(callbacks, "warning", f"Extractor raised an error: {exc}")
            df = None

    if df is None and RAW_EXPORTS_PATH.exists():
        _emit(callbacks, "info", "Using previously downloaded raw dataset from disk")
        df = pd.read_csv(RAW_EXPORTS_PATH)

    if df is None:
        raise FileNotFoundError("No raw export data available. Extraction failed.")

    return df


def _clean_raw_data(df: pd.DataFrame, callbacks: CallbackMap) -> pd.DataFrame:
    """Apply the full cleaning pipeline (port of notebook logic)."""
    _emit(callbacks, "info", "Cleaning raw dataset to notebook standards...")

    # Rename columns consistent with notebook expectations
    if 'sitc' in df.columns and 'product_description' not in df.columns:
        df = df.rename(columns={'sitc': 'product_description'})
    if 'sitc_code' in df.columns and 'prod_descpt_code' not in df.columns:
        df = df.rename(columns={'sitc_code': 'prod_descpt_code'})

    # Numeric columns
    numeric_columns = ['quantity', 'gross_weight_tonnes', 'value_fob_aud']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)

    # Text columns
    text_columns = ['country_of_destination', 'product_description']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'nan': 'Unknown', 'NaN': 'Unknown'}).fillna('Unknown')

    if 'product_description' in df.columns:
        df['product_description'] = (
            df['product_description']
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[\n\r\t]+", " ", regex=True)
            .str.replace('"', '', regex=False)
            .str.replace("'", '', regex=False)
        )

    # Derived features
    if 'month' in df.columns:
        if 'year' not in df.columns:
            df['year'] = df['month'].astype(str).str.extract(r"(\d{4})")[0]
            df['year'] = pd.to_numeric(df['year'], errors='coerce')

        if 'month_number' not in df.columns:
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            month_name = df['month'].astype(str).str.extract(r'^([A-Za-z]+)')[0]
            df['month_number'] = month_name.map(month_map)

    if {'value_fob_aud', 'gross_weight_tonnes'}.issubset(df.columns):
        denominator = df['gross_weight_tonnes'].replace(0, np.nan)
        df['value_per_tonne'] = df['value_fob_aud'] / denominator

    df['data_processed_date'] = datetime.now().strftime('%Y-%m-%d')

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        _emit(callbacks, "info", f"Removed {removed:,} duplicate rows")

    # Optional mappings (same as notebook) wrapped in try/except to avoid hard deps
    try:
        from sitc_mapping import map_sitc_to_product, get_unclassified_patterns
        if {'product_description', 'prod_descpt_code'}.issubset(df.columns):
            patterns = '|'.join(get_unclassified_patterns())
            mask = df['product_description'].astype(str).str.contains(patterns, case=False, na=False)
            if mask.any():
                df.loc[mask, 'product_description'] = df.loc[mask, 'prod_descpt_code'].apply(map_sitc_to_product)
    except ImportError:
        pass

    try:
        from country_mapping import is_problematic_country_name, map_country_code_to_name
        if {'country_of_destination', 'country_of_destination_code'}.issubset(df.columns):
            mask = df['country_of_destination'].apply(is_problematic_country_name)
            if mask.any():
                df.loc[mask, 'country_of_destination'] = df.loc[mask, 'country_of_destination_code'].apply(map_country_code_to_name)
        if 'country_of_destination' in df.columns:
            mask_code = df['country_of_destination'].astype(str).str.contains('Country Code', na=False)
            if mask_code.any():
                cleaned = df.loc[mask_code, 'country_of_destination'].str.replace('Country Code ', '')
                df.loc[mask_code, 'country_of_destination'] = cleaned.apply(map_country_code_to_name)
    except ImportError:
        pass

    if 'country_of_destination' in df.columns:
        df['country_of_destination'] = df['country_of_destination'].astype(str).str.strip()

    return df


def ensure_clean_data(
    max_age_days: int = 30,
    callbacks: CallbackMap = None,
    force_refresh: bool = False,
) -> Path:
    """Ensure we have a fresh cleaned dataset and return its path.

    Parameters
    ----------
    max_age_days: int
        Regenerate data if the existing cleaned file is older than this many days.
    callbacks: dict
        Optional mapping of {'info': callable, 'warning': callable, 'success': callable}.
        Useful to pipe messages to Streamlit or logging.
    force_refresh: bool
        Regenerate regardless of file age.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_EXPORTS_PATH.exists() and not force_refresh:
        age_days = (time.time() - CLEAN_EXPORTS_PATH.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            _emit(callbacks, "info", f"Using cached cleaned dataset (age: {age_days:.1f} days)")
            return CLEAN_EXPORTS_PATH
        else:
            _emit(callbacks, "info", f"Clean dataset is {age_days:.0f} days old; regenerating...")
    else:
        if CLEAN_EXPORTS_PATH.exists():
            _emit(callbacks, "info", "Force refresh requested. Regenerating clean dataset...")
        else:
            _emit(callbacks, "info", "No cleaned dataset found. Generating now...")

    # Download + clean
    raw_df = _download_raw_data(callbacks)
    clean_df = _clean_raw_data(raw_df, callbacks)

    clean_df.to_csv(CLEAN_EXPORTS_PATH, index=False)
    _emit(callbacks, "success", f"Clean dataset saved to {CLEAN_EXPORTS_PATH}")

    return CLEAN_EXPORTS_PATH

__all__ = ["ensure_clean_data", "CLEAN_EXPORTS_PATH", "RAW_EXPORTS_PATH"]
