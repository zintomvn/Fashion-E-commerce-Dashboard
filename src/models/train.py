from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.load_data import load_sales
from src.data.preprocess import clean_sales, ensure_daily_continuity
from src.features.build_features import build_supervised_frame, get_feature_columns
from src.utils.metrics import mae, r2, rmse


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _build_X_y(
    sales: pd.DataFrame,
    target_col: str,
    date_col: str,
    lags: List[int],
    windows: List[int],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    frame = build_supervised_frame(
        sales,
        target_col=target_col,
        date_col=date_col,
        lags=lags,
        windows=windows,
    )

    feature_cols = get_feature_columns(frame, target_col=target_col, date_col=date_col)
    frame = frame.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X = frame[feature_cols]
    y = frame[target_col]
    dates = frame[date_col]
    return X, y, dates, feature_cols


def _fit_and_eval(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    val_days: int,
    random_state: int,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, float]]:
    cutoff = dates.max() - pd.Timedelta(days=val_days)
    train_mask = dates <= cutoff

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[~train_mask], y.loc[~train_mask]

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    metrics = {
        "mae": mae(y_val, pred),
        "rmse": rmse(y_val, pred),
        "r2": r2(y_val, pred),
    }

    # Refit on all data for final model.
    model.fit(X, y)
    return model, metrics


def train(
    raw_dir: Path,
    model_out: Path,
    val_days: int,
    lags: List[int],
    windows: List[int],
    random_state: int,
) -> Path:
    sales = load_sales(raw_dir=raw_dir)
    sales = clean_sales(sales, date_col="Date")
    sales = ensure_daily_continuity(sales, date_col="Date")

    targets = ["Revenue", "COGS"]
    models: Dict[str, HistGradientBoostingRegressor] = {}
    feature_cols_by_target: Dict[str, List[str]] = {}
    metrics_by_target: Dict[str, Dict[str, float]] = {}

    for target in targets:
        X, y, dates, feature_cols = _build_X_y(
            sales=sales,
            target_col=target,
            date_col="Date",
            lags=lags,
            windows=windows,
        )

        model, metrics = _fit_and_eval(
            X=X,
            y=y,
            dates=dates,
            val_days=val_days,
            random_state=random_state,
        )

        models[target] = model
        feature_cols_by_target[target] = feature_cols
        metrics_by_target[target] = metrics

        print(f"[{target}] val_mae={metrics['mae']:.2f} val_rmse={metrics['rmse']:.2f} val_r2={metrics['r2']:.4f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "models": models,
        "feature_cols_by_target": feature_cols_by_target,
        "config": {
            "date_col": "Date",
            "val_days": val_days,
            "lags": lags,
            "windows": windows,
            "random_state": random_state,
        },
        "metrics": metrics_by_target,
        "train_end_date": str(sales["Date"].max().date()),
    }

    joblib.dump(package, model_out)
    print(f"Saved model package -> {model_out}")
    return model_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline revenue/COGS forecaster.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("outputs") / "models" / "model_package.joblib",
        help="Where to save the trained model package.",
    )
    parser.add_argument("--val-days", type=int, default=365, help="Validation window (days).")
    parser.add_argument(
        "--lags",
        type=str,
        default="1,7,14,28,56,365",
        help="Comma-separated lag days.",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="7,14,28",
        help="Comma-separated rolling window sizes.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    lags = _parse_int_list(args.lags)
    windows = _parse_int_list(args.windows)

    train(
        raw_dir=args.raw_dir,
        model_out=args.model_out,
        val_days=args.val_days,
        lags=lags,
        windows=windows,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
