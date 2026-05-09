from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_sales
from src.data.preprocess import clean_sales, ensure_daily_continuity
from src.features.build_features import add_time_features


def _make_feature_row(
    date: pd.Timestamp,
    history: List[float],
    target_col: str,
    lags: List[int],
    windows: List[int],
    feature_cols: List[str],
) -> pd.DataFrame:
    row = pd.DataFrame({"Date": [date]})
    row = add_time_features(row, date_col="Date")

    for lag in lags:
        row[f"{target_col}_lag_{lag}"] = history[-lag]

    for window in windows:
        window_vals = history[-window:]
        row[f"{target_col}_roll_mean_{window}"] = float(np.mean(window_vals))
        row[f"{target_col}_roll_std_{window}"] = float(np.std(window_vals, ddof=1))

    return row[feature_cols]


def _forecast_autoreg(
    model,
    history_values: List[float],
    future_dates: List[pd.Timestamp],
    target_col: str,
    lags: List[int],
    windows: List[int],
    feature_cols: List[str],
) -> List[float]:
    history = list(history_values)
    preds: List[float] = []

    for d in future_dates:
        X_row = _make_feature_row(
            date=d,
            history=history,
            target_col=target_col,
            lags=lags,
            windows=windows,
            feature_cols=feature_cols,
        )
        pred = float(model.predict(X_row)[0])
        pred = max(0.0, pred)

        preds.append(pred)
        history.append(pred)

    return preds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate submission predictions.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data") / "raw" / "sample_submission.csv",
        help="Submission template with Date column (default: data/raw/sample_submission.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs") / "models" / "model_package.joblib",
        help="Path to the trained model package.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs") / "submissions" / "submission.csv",
        help="Where to write submission CSV.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model package not found: {args.model_path}. Run `python -m src.models.train` first."
        )

    package: Dict = joblib.load(args.model_path)
    config = package["config"]
    lags: List[int] = list(config["lags"])
    windows: List[int] = list(config["windows"])

    template = pd.read_csv(args.template)
    if "Date" not in template.columns and "date" in template.columns:
        template = template.rename(columns={"date": "Date"})

    if "Date" not in template.columns:
        raise ValueError("Template must contain a Date column.")

    template["Date"] = pd.to_datetime(template["Date"], errors="coerce").dt.normalize()
    template = template.dropna(subset=["Date"]).reset_index(drop=True)

    template_dates = template["Date"]
    future_dates_sorted = sorted(pd.DatetimeIndex(template_dates.unique()))

    sales = load_sales(raw_dir=args.raw_dir)
    sales = clean_sales(sales, date_col="Date")
    sales = ensure_daily_continuity(sales, date_col="Date")
    sales = sales.sort_values("Date").reset_index(drop=True)

    train_end = pd.Timestamp(sales["Date"].max()).normalize()
    if future_dates_sorted and future_dates_sorted[0] <= train_end:
        raise ValueError(
            f"Template dates must be strictly after train end date ({train_end.date()}). "
            f"Got first template date: {future_dates_sorted[0].date()}"
        )

    if future_dates_sorted:
        expected = pd.date_range(future_dates_sorted[0], future_dates_sorted[-1], freq="D")
        if len(expected) != len(future_dates_sorted) or not (expected == pd.DatetimeIndex(future_dates_sorted)).all():
            raise ValueError(
                "This baseline assumes the template contains a continuous daily range. "
                "(Dates are missing or duplicated.)"
            )

    out = pd.DataFrame({"Date": template_dates.dt.strftime("%Y-%m-%d")})

    for target in ["Revenue", "COGS"]:
        model = package["models"][target]
        feature_cols = package["feature_cols_by_target"][target]
        history_values = sales[target].astype(float).tolist()

        preds = _forecast_autoreg(
            model=model,
            history_values=history_values,
            future_dates=list(future_dates_sorted),
            target_col=target,
            lags=lags,
            windows=windows,
            feature_cols=feature_cols,
        )
        pred_map = {d: p for d, p in zip(future_dates_sorted, preds)}
        out[target] = template_dates.map(pred_map).astype(float)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)
    print(f"Wrote submission -> {args.out_path}")


if __name__ == "__main__":
    main()
