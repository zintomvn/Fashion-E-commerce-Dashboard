from __future__ import annotations

from typing import Tuple

import pandas as pd


def clean_sales(sales: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Basic cleaning for the daily sales table."""

    df = sales.copy()

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    for col in ["Revenue", "COGS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate duplicates (if any) by summing numeric targets.
    numeric_cols = [c for c in ["Revenue", "COGS"] if c in df.columns]
    if df.duplicated(subset=[date_col]).any() and numeric_cols:
        df = df.groupby(date_col, as_index=False)[numeric_cols].sum()

    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def ensure_daily_continuity(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Validate that the time series has no missing dates."""

    start = df[date_col].min()
    end = df[date_col].max()
    full = pd.date_range(start=start, end=end, freq="D")

    observed = pd.DatetimeIndex(df[date_col].unique())
    missing = full.difference(observed)
    if len(missing) > 0:
        raise ValueError(
            f"Sales time series is missing {len(missing)} days between {start.date()} and {end.date()}."
        )

    return df


def train_val_split_time(
    df: pd.DataFrame, val_days: int = 365, date_col: str = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: last `val_days` are validation."""

    cutoff = df[date_col].max() - pd.Timedelta(days=val_days)
    train_df = df[df[date_col] <= cutoff].copy()
    val_df = df[df[date_col] > cutoff].copy()
    return train_df, val_df
