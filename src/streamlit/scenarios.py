"""
scenarios.py — What-if computation engines for DATATHON-2026 Dashboard.

Promo Engine  : computes expected incremental profit from uplift predictions.
Ops Engine    : looks up counterfactual profit impact from pre-computed CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PROMO ENGINE
# ---------------------------------------------------------------------------

class PromoEngine:
    """
    Simulate promo P&L impact using uplift model predictions.

    Parameters
    ----------
    uplift_df : DataFrame with columns
        [P_Y_given_Promo, P_Y_given_NoPromo, Uplift_Score, price, cogs,
         return_rate, customer_segment, category, acquisition_channel, ...]
    """

    def __init__(self, uplift_df: pd.DataFrame) -> None:
        self.df = uplift_df.copy()

    def _apply_promo_price(
        self,
        price: pd.Series,
        promo_type: str,
        discount_value: float,
    ) -> pd.Series:
        """Calculate post-promo price. Clamp to >= 0."""
        if promo_type == "percentage":
            promo_price = price * (1 - discount_value / 100)
        else:  # fixed
            promo_price = price - discount_value
        return promo_price.clip(lower=0)

    def compute(
        self,
        promo_type: str = "percentage",
        discount_value: float = 10.0,
        penalise_returns: bool = True,
        top_n: int | None = None,
        uplift_threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute expected_uplift_profit per (customer, product) row.

        Returns a sorted DataFrame with the simulation results.
        """
        df = self.df.copy()
        required = {"price", "cogs", "P_Y_given_Promo", "P_Y_given_NoPromo"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        # Margins
        promo_price  = self._apply_promo_price(df["price"], promo_type, discount_value)
        base_margin  = df["price"] - df["cogs"]
        promo_margin = promo_price - df["cogs"]

        # Expected profits
        df["expected_base"]  = df["P_Y_given_NoPromo"] * base_margin
        df["expected_promo"] = df["P_Y_given_Promo"]  * promo_margin

        # Optional: penalise high-return segments
        if penalise_returns and "return_rate" in df.columns:
            df["expected_promo"] = df["expected_promo"] * (1 - df["return_rate"].fillna(0))

        df["expected_uplift_profit"] = df["expected_promo"] - df["expected_base"]
        df["promo_price"] = promo_price

        # Filter by uplift threshold
        if uplift_threshold is not None and "Uplift_Score" in df.columns:
            df = df[df["Uplift_Score"] >= uplift_threshold]

        # Sort descending by expected uplift profit
        df = df.sort_values("expected_uplift_profit", ascending=False).reset_index(drop=True)

        # Top-N filter
        if top_n:
            df = df.head(top_n)

        return df

    def summary_stats(self, computed: pd.DataFrame) -> dict:
        """Aggregate P&L summary over targeted rows."""
        if computed.empty:
            return {}
        total_base  = computed["expected_base"].sum()
        total_promo = computed["expected_promo"].sum()
        total_uplift = computed["expected_uplift_profit"].sum()
        n_targeted  = len(computed)
        return {
            "n_targeted":    n_targeted,
            "total_base":    total_base,
            "total_promo":   total_promo,
            "total_uplift":  total_uplift,
            "avg_uplift":    computed["expected_uplift_profit"].mean(),
            "pct_positive":  (computed["expected_uplift_profit"] > 0).mean() * 100,
        }


# ---------------------------------------------------------------------------
# OPS ENGINE
# ---------------------------------------------------------------------------

class OpsEngine:
    """
    Look up counterfactual profit impact from pre-computed table.

    Parameters
    ----------
    cf_df : DataFrame with columns
        [customer_segment, action, feature, baseline_profit, scenario_profit,
         delta_profit, delta_profit_pct, baseline_revenue, scenario_revenue,
         delta_revenue, baseline_cost, scenario_cost, delta_cost]
    """

    def __init__(self, cf_df: pd.DataFrame) -> None:
        self.df = cf_df.copy()

    @property
    def segments(self) -> list[str]:
        return sorted(self.df["customer_segment"].unique().tolist())

    @property
    def actions(self) -> list[str]:
        return sorted(self.df["action"].unique().tolist())

    def filter(
        self, segment: str | None = None, action: str | None = None
    ) -> pd.DataFrame:
        df = self.df.copy()
        if segment:
            df = df[df["customer_segment"] == segment]
        if action:
            df = df[df["action"] == action]
        return df.reset_index(drop=True)

    def best_action(self, segment: str) -> pd.Series | None:
        """Return the single best action row for a segment by delta_profit_pct."""
        df = self.filter(segment=segment)
        if df.empty:
            return None
        return df.loc[df["delta_profit_pct"].idxmax()]

    def leaderboard(self) -> pd.DataFrame:
        """Best action per segment ranked by delta_profit_pct."""
        rows = []
        for seg in self.segments:
            best = self.best_action(seg)
            if best is not None:
                rows.append(best)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("delta_profit_pct", ascending=False).reset_index(drop=True)

    def waterfall_inputs(self, segment: str, action: str | None = None) -> pd.Series | None:
        """Return the row to feed into chart_ops_waterfall."""
        df = self.filter(segment=segment, action=action)
        if df.empty:
            return None
        # If multiple rows, pick the one with highest delta_profit
        return df.loc[df["delta_profit"].idxmax()]


# ---------------------------------------------------------------------------
# DIAGNOSTICS ENGINE (Page D)
# ---------------------------------------------------------------------------

def compute_confounders(
    df: pd.DataFrame,
    treatment_col: str = "True_T",
    outcome_col: str = "True_Y",
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute confounder scores for each numeric feature:
        assoc_T = |corr(f, treatment)|
        assoc_Y = |corr(f, outcome)|
        confounder_score = assoc_T * assoc_Y
    """
    if df.empty:
        return pd.DataFrame()

    if numeric_cols is None:
        exclude = {treatment_col, outcome_col, "customer_id", "product_id"}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        sub = df[[col, treatment_col, outcome_col]].dropna()
        if len(sub) < 10:
            continue
        assoc_t = abs(sub[col].corr(sub[treatment_col]))
        assoc_y = abs(sub[col].corr(sub[outcome_col]))
        rows.append({
            "feature": col,
            "assoc_T": assoc_t,
            "assoc_Y": assoc_y,
            "confounder_score": assoc_t * assoc_y,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("confounder_score", ascending=False).reset_index(drop=True)


# Pipeline categorical features (mirrors notebook feature engineering)
_PIPELINE_CAT_FEATURES = [
    "age_group", "gender", "city", "region", "acquisition_channel",
    "device_type", "order_source", "payment_method",
    "category", "segment", "size", "color",
]


def _encode_categoricals(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Label-encode pipeline categorical features present in the dataframe.
    Returns (encoded_df_subset, encoded_col_names).
    """
    from sklearn.preprocessing import LabelEncoder
    encoded_frames = []
    encoded_names = []
    for col in _PIPELINE_CAT_FEATURES:
        if col in sample.columns:
            enc_col = f"{col}_enc"
            le = LabelEncoder()
            vals = sample[col].fillna("Unknown").astype(str)
            encoded_frames.append(
                pd.Series(le.fit_transform(vals), index=sample.index, name=enc_col)
            )
            encoded_names.append(enc_col)
    if encoded_frames:
        return pd.concat(encoded_frames, axis=1), encoded_names
    return pd.DataFrame(index=sample.index), []


def compute_shap_surrogate(
    df: pd.DataFrame,
    target_col: str = "Uplift_Score",
    feature_cols: list[str] | None = None,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """
    Fit a LightGBM surrogate and compute mean |SHAP| values.
    Includes both numeric and label-encoded categorical pipeline features.
    Returns DataFrame with [feature, mean_abs_shap].
    """
    if df.empty or target_col not in df.columns:
        return pd.DataFrame()

    # Subsample for performance
    sample = df.sample(min(max_rows, len(df)), random_state=42)

    # --- FIX lỗi 4: exclude promo_order_rate when target=True_Y (treatment leakage)
    # --- FIX lỗi 3: align exclude set with pipeline
    if feature_cols is None:
        base_exclude = {target_col, "True_T", "True_Y", "customer_id", "product_id",
                        "P_Y_given_Promo", "P_Y_given_NoPromo", "Uplift_Score"}
        if target_col == "True_Y":
            base_exclude.add("promo_order_rate")  # encodes treatment history → leakage

        num_cols = [
            c for c in sample.select_dtypes(include=[np.number]).columns
            if c not in base_exclude
        ]
        # Add label-encoded categoricals (FIX lỗi 3)
        cat_df, cat_cols = _encode_categoricals(sample)
        X_num = sample[num_cols].fillna(0)
        X = pd.concat([X_num, cat_df], axis=1) if not cat_df.empty else X_num
        feature_cols_final = num_cols + cat_cols
    else:
        X = sample[feature_cols].fillna(0)
        feature_cols_final = feature_cols

    y = sample[target_col].fillna(0)

    # --- FIX lỗi 2: balanced training when target is binary
    is_binary = (y.nunique() == 2 and set(y.dropna().unique()).issubset({0, 1}))

    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
        if is_binary:
            neg = int((y == 0).sum()); pos = int((y == 1).sum())
            spw = float(neg / pos) if pos > 0 else 1.0
            model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1,
                                   scale_pos_weight=spw)
        else:
            model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X, y)

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            # For classifiers shap may return list of arrays
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            mean_abs = np.abs(shap_vals).mean(axis=0)
        except Exception:
            mean_abs = model.feature_importances_

    except ImportError:
        try:
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            if is_binary:
                model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            mean_abs = model.feature_importances_
        except Exception:
            return pd.DataFrame()

    result = pd.DataFrame({
        "feature": feature_cols_final,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return result


def compute_surrogate_metrics(
    df: pd.DataFrame,
    target_col: str = "Uplift_Score",
    feature_cols: list[str] | None = None,
    max_rows: int = 3000,
) -> dict:
    """
    Fit a surrogate and evaluate on holdout to get R², MAE, RMSE.
    For binary targets (True_Y): uses class-balanced model + optimal threshold.
    Includes both numeric and label-encoded categorical pipeline features.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import math

    if df.empty or target_col not in df.columns:
        return {}

    sample = df.sample(min(max_rows, len(df)), random_state=42)

    # --- FIX lỗi 3 + 4: align exclude set, block treatment-leaking feature for True_Y
    if feature_cols is None:
        base_exclude = {target_col, "True_T", "True_Y", "customer_id", "product_id",
                        "P_Y_given_Promo", "P_Y_given_NoPromo", "Uplift_Score"}
        if target_col == "True_Y":
            base_exclude.add("promo_order_rate")  # encodes treatment history → leakage

        num_cols = [
            c for c in sample.select_dtypes(include=[np.number]).columns
            if c not in base_exclude
        ]
        # FIX lỗi 3: include label-encoded categoricals from pipeline
        cat_df, cat_cols = _encode_categoricals(sample)
        X = pd.concat([sample[num_cols].fillna(0), cat_df], axis=1) \
            if not cat_df.empty else sample[num_cols].fillna(0)
        all_feature_cols = num_cols + cat_cols
    else:
        X = sample[feature_cols].fillna(0)
        all_feature_cols = feature_cols

    y = sample[target_col].fillna(0)

    if len(X) < 20:
        return {}

    # Detect binary target
    is_binary = (y.nunique() == 2 and set(y.dropna().unique()).issubset({0, 1}))

    # FIX lỗi 2: stratify split for binary target
    strat = y if is_binary else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    # FIX lỗi 2: class-balanced model for imbalanced binary target
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
        if is_binary:
            neg = int((y_train == 0).sum()); pos = int((y_train == 1).sum())
            spw = float(neg / pos) if pos > 0 else 1.0
            model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1,
                                   scale_pos_weight=spw)
        else:
            model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        if is_binary:
            model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    model.fit(X_train, y_train)

    if is_binary:
        from sklearn.metrics import (
            roc_auc_score, balanced_accuracy_score, log_loss,
            roc_curve, confusion_matrix,
        )
        y_prob = model.predict_proba(X_test)[:, 1]

        # FIX lỗi 2: optimal threshold via Youden's J (not 0.5 on imbalanced data)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        youden_j = tpr - fpr
        best_thresh = float(thresholds[np.argmax(youden_j)])
        y_pred_bal = (y_prob >= best_thresh).astype(int)

        try:
            auc = round(roc_auc_score(y_test, y_prob), 4)
        except Exception:
            auc = 0.5
        bal_acc = round(balanced_accuracy_score(y_test, y_pred_bal), 4)
        try:
            ll = round(log_loss(y_test, y_prob), 4)
        except Exception:
            ll = None
        cm = confusion_matrix(y_test, y_pred_bal)

        return {
            "auc":               auc,
            "balanced_acc":      bal_acc,
            "log_loss":          ll,
            "optimal_threshold": round(best_thresh, 4),
            "cm":                cm,          # ← confusion matrix for display
            # Keep r2/mae/rmse keys for display compatibility
            "r2":   auc,
            "mae":  round(1 - bal_acc, 4),
            "rmse": ll if ll is not None else 0.0,
        }
    else:
        y_pred = model.predict(X_test)
        return {
            "r2":   round(r2_score(y_test, y_pred), 4),
            "mae":  round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(math.sqrt(mean_squared_error(y_test, y_pred)), 4),
        }
