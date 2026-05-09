from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    date = pd.to_datetime(out[date_col], errors="coerce")

    out["year"] = date.dt.year.astype(int)
    out["month"] = date.dt.month.astype(int)
    out["day"] = date.dt.day.astype(int)
    out["dayofweek"] = date.dt.dayofweek.astype(int)
    out["weekofyear"] = date.dt.isocalendar().week.astype(int)
    out["dayofyear"] = date.dt.dayofyear.astype(int)

    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = date.dt.is_month_start.astype(int)
    out["is_month_end"] = date.dt.is_month_end.astype(int)
    out["is_quarter_start"] = date.dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = date.dt.is_quarter_end.astype(int)
    out["is_year_start"] = date.dt.is_year_start.astype(int)
    out["is_year_end"] = date.dt.is_year_end.astype(int)

    # Cyclical encodings
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

    return out


def add_lag_features(df: pd.DataFrame, value_col: str, lags: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{value_col}_lag_{lag}"] = out[value_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, value_col: str, windows: Iterable[int]
) -> pd.DataFrame:
    out = df.copy()

    # Shift by 1 day so features for day t only use values <= t-1.
    shifted = out[value_col].shift(1)
    for window in windows:
        out[f"{value_col}_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"{value_col}_roll_std_{window}"] = shifted.rolling(window).std()

    return out


def build_supervised_frame(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = "Date",
    lags: Iterable[int] = (1, 7, 14, 28, 56, 365),
    windows: Iterable[int] = (7, 14, 28),
) -> pd.DataFrame:
    """Build a modeling frame with time + lag + rolling features.

    The returned frame still contains NaNs for the first max(lag/window) rows.
    Drop them at training time.
    """

    out = df[[date_col, target_col]].copy()
    out = add_time_features(out, date_col=date_col)
    out = add_lag_features(out, target_col, lags)
    out = add_rolling_features(out, target_col, windows)
    return out


def get_feature_columns(frame: pd.DataFrame, target_col: str, date_col: str = "Date") -> List[str]:
    return [c for c in frame.columns if c not in {date_col, target_col}]
