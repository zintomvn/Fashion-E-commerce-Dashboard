from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_TABLE_FILES: Dict[str, str] = {
    "customers": "customers.csv",
    "geography": "geography.csv",
    "inventory": "inventory.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "payments.csv",
    "products": "products.csv",
    "promotions": "promotions.csv",
    "returns": "returns.csv",
    "reviews": "reviews.csv",
    "sales": "sales.csv",
    "sample_submission": "sample_submission.csv",
    "shipments": "shipments.csv",
    "web_traffic": "web_traffic.csv",
}

TABLE_DATE_COLUMNS: Dict[str, Iterable[str]] = {
    "customers": ["signup_date"],
    "inventory": ["snapshot_date"],
    "orders": ["order_date"],
    "promotions": ["start_date", "end_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "sales": ["Date"],
    "sample_submission": ["Date"],
    "shipments": ["ship_date", "delivery_date"],
    "web_traffic": ["date"],
}


def load_table(
    table: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    parse_dates: Optional[Iterable[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a raw CSV table by logical name.

    Parameters
    ----------
    table:
        One of the keys in RAW_TABLE_FILES.
    raw_dir:
        Directory containing the raw CSV files.
    parse_dates:
        Columns to parse as datetimes. If None, uses TABLE_DATE_COLUMNS when available.
    read_csv_kwargs:
        Additional keyword args passed to pandas.read_csv.
    """

    if table not in RAW_TABLE_FILES:
        raise KeyError(f"Unknown table: {table}. Expected one of: {sorted(RAW_TABLE_FILES)}")

    file_path = raw_dir / RAW_TABLE_FILES[table]
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    if parse_dates is None:
        parse_dates = TABLE_DATE_COLUMNS.get(table)

    parse_dates_list = list(parse_dates) if parse_dates else None
    return pd.read_csv(file_path, parse_dates=parse_dates_list, **read_csv_kwargs)


def load_all_raw(raw_dir: Path = DEFAULT_RAW_DIR) -> Dict[str, pd.DataFrame]:
    """Load all raw tables into a dict keyed by logical table name."""

    tables: Dict[str, pd.DataFrame] = {}
    for table in RAW_TABLE_FILES:
        tables[table] = load_table(table, raw_dir=raw_dir)
    return tables


def load_sales(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load and standardize the sales (train) table."""

    df = load_table("sales", raw_dir=raw_dir)
    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    expected = {"Date", "Revenue", "COGS"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"sales.csv is missing columns: {sorted(missing)}")

    df = df[["Date", "Revenue", "COGS"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["COGS"] = pd.to_numeric(df["COGS"], errors="coerce")

    return df


def load_submission_template(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the sample submission template (test dates).

    Note: The competition may provide a separate sales_test.csv later; in that case,
    prefer reading that file directly.
    """

    df = load_table("sample_submission", raw_dir=raw_dir)
    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df
