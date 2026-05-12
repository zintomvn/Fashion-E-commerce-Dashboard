"""
data.py — Data loading and caching layer for DATATHON-2026 Dashboard.
All loaders use @st.cache_data to avoid re-reading large files on reruns.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Root paths (resolved relative to this file so the app works from any cwd)
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent          # src/streamlit/
_ROOT = _HERE.parent.parent                    # project root
DATA_RAW = _ROOT / "data" / "raw"
OUTPUTS   = _ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _read(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    """Read CSV with error handling."""
    if not path.exists():
        st.warning(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# RAW DATA LOADERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sales() -> pd.DataFrame:
    """Daily Revenue & COGS time series (2012–2022)."""
    df = _read(DATA_RAW / "sales.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["GrossProfit"] = df["Revenue"] - df["COGS"]
    df["GrossMarginPct"] = (df["GrossProfit"] / df["Revenue"].replace(0, np.nan)) * 100
    df["Rolling7_Revenue"] = df["Revenue"].rolling(7, min_periods=1).mean()
    df["Rolling7_GP"] = df["GrossProfit"].rolling(7, min_periods=1).mean()
    return df


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    """XGBoost Revenue & COGS forecast (2023+)."""
    # df = _read(OUTPUTS / "submission_xgb_fold5_tuned.csv", parse_dates=["Date"])
    df = _read(OUTPUTS / "submission_test.csv", parse_dates=["Date"])
    if df.empty:
        df = _read(OUTPUTS / "submission_xgb_fold5.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "GrossProfit" not in df.columns:
        df["GrossProfit"] = df["Revenue"] - df["COGS"]
    return df


@st.cache_data(show_spinner=False)
def load_orders() -> pd.DataFrame:
    """Order headers with status, payment, device, source."""
    df = _read(DATA_RAW / "orders.csv", parse_dates=["order_date"])
    return df


@st.cache_data(show_spinner=False)
def load_order_items() -> pd.DataFrame:
    """Order items with product_id, quantity, unit_price."""
    df = _read(DATA_RAW / "order_items.csv")
    return df


@st.cache_data(show_spinner=False)
def load_shipments() -> pd.DataFrame:
    """Shipment dates and fees; derive lead_time_days."""
    df = _read(DATA_RAW / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
    if not df.empty and "delivery_date" in df.columns and "ship_date" in df.columns:
        df["lead_time_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
        df["lead_time_days"] = df["lead_time_days"].clip(lower=0)
    return df


@st.cache_data(show_spinner=False)
def load_returns() -> pd.DataFrame:
    """Returns with reason and refund amount."""
    df = _read(DATA_RAW / "returns.csv")
    return df


@st.cache_data(show_spinner=False)
def load_geography() -> pd.DataFrame:
    """Zip → city / region / district mapping."""
    df = _read(DATA_RAW / "geography.csv")
    return df


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    """Product catalog: category, segment, size, color, price, cogs."""
    df = _read(DATA_RAW / "products.csv")
    return df


@st.cache_data(show_spinner=False)
def load_customers() -> pd.DataFrame:
    """Customer demographics (light cols only to keep memory low)."""
    cols = ["customer_id", "zip", "gender", "age_group", "acquisition_channel",
            "device_type", "order_source", "payment_method"]
    available = [c for c in cols if True]  # will filter after load
    df = _read(DATA_RAW / "customers.csv")
    keep = [c for c in cols if c in df.columns]
    return df[keep] if not df.empty else df


# ---------------------------------------------------------------------------
# OUTPUT / AI ARTIFACT LOADERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_uplift(sample_n: int | None = 30_000) -> pd.DataFrame:
    """Uplift predictions test set (large — sampled for perf)."""
    df = _read(OUTPUTS / "uplift_predictions_test_set.csv")
    if df.empty:
        return df
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_rfm() -> pd.DataFrame:
    """Customer RFM segments."""
    df = _read(OUTPUTS / "customer_rfm_segments.csv")
    return df


@st.cache_data(show_spinner=False)
def load_counterfactual() -> pd.DataFrame:
    """Profit optimization counterfactual actions (segment × action)."""
    df = _read(OUTPUTS / "profit_optimization_counterfactual_actions.csv")
    return df


@st.cache_data(show_spinner=False)
def load_best_actions() -> pd.DataFrame:
    """Best action per segment from profit optimization."""
    df = _read(OUTPUTS / "profit_optimization_best_actions_by_segment.csv")
    return df


@st.cache_data(show_spinner=False)
def load_component_models() -> pd.DataFrame:
    """Component model feature importances per segment."""
    df = _read(OUTPUTS / "profit_optimization_component_models.csv")
    return df


@st.cache_data(show_spinner=False)
def load_product_portfolio() -> pd.DataFrame:
    """BCG-style product portfolio segments (Stars/Cash Cows/Dogs/Question Marks)."""
    df = _read(OUTPUTS / "product_profit_volume_segments.csv")
    return df


# ---------------------------------------------------------------------------
# JOINED / DERIVED DATASETS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_orders_enriched() -> pd.DataFrame:
    """Orders joined with geography, shipment lead time."""
    orders = load_orders()
    geo    = load_geography()
    ships  = load_shipments()

    if orders.empty:
        return orders

    # Join geography
    if not geo.empty and "zip" in orders.columns:
        orders = orders.merge(geo[["zip", "city", "region"]], on="zip", how="left")

    # Join shipment lead time
    if not ships.empty:
        orders = orders.merge(
            ships[["order_id", "lead_time_days", "shipping_fee"]],
            on="order_id", how="left"
        )

    return orders


@st.cache_data(show_spinner=False)
def load_uplift_enriched() -> pd.DataFrame:
    """Uplift predictions joined with RFM customer segments."""
    uplift = load_uplift()
    rfm    = load_rfm()
    if uplift.empty:
        return uplift
    if not rfm.empty and "customer_id" in uplift.columns and "customer_id" in rfm.columns:
        seg_map = rfm[["customer_id", "customer_segment"]].drop_duplicates("customer_id")
        # If uplift already has customer_segment from its own columns, skip; else join
        if "customer_segment" not in uplift.columns:
            uplift = uplift.merge(seg_map, on="customer_id", how="left")
    return uplift


# ---------------------------------------------------------------------------
# SUMMARY STATS  (used by KPI cards)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_kpis(sales: pd.DataFrame) -> dict:
    """Compute top-level KPIs from sales dataframe."""
    if sales.empty:
        return {}

    total_revenue   = sales["Revenue"].sum()
    total_cogs      = sales["COGS"].sum()
    total_gp        = sales["GrossProfit"].sum()
    avg_gm_pct      = (total_gp / total_revenue * 100) if total_revenue else 0

    # YoY growth: compare last full calendar year vs prior year
    sales_yr = sales.copy()
    sales_yr["year"] = sales_yr["Date"].dt.year
    yearly = sales_yr.groupby("year")["Revenue"].sum()
    yoy_growth = None
    if len(yearly) >= 2:
        last_yr, prev_yr = yearly.index[-1], yearly.index[-2]
        if yearly[prev_yr] > 0:
            yoy_growth = (yearly[last_yr] - yearly[prev_yr]) / yearly[prev_yr] * 100

    # 30-day momentum
    last_30  = sales.tail(30)["Revenue"].sum()
    prev_30  = sales.iloc[-60:-30]["Revenue"].sum() if len(sales) >= 60 else None
    mom_pct  = ((last_30 - prev_30) / prev_30 * 100) if prev_30 else None

    # Avg daily revenue
    avg_daily = sales["Revenue"].mean()

    return {
        "total_revenue":   total_revenue,
        "total_cogs":      total_cogs,
        "total_gp":        total_gp,
        "avg_gm_pct":      avg_gm_pct,
        "yoy_growth":      yoy_growth,
        "mom_pct":         mom_pct,
        "avg_daily":       avg_daily,
    }
