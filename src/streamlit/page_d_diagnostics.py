"""
page_d_diagnostics.py — Page D: Regression & Confounders (Diagnostics + XAI)
Covers: KPI R²/MAE/RMSE, Correlation heatmap, Scatter + regression, Confounder ranking, SHAP.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from src.streamlit.components import (
    render_page_header, render_section_header, render_metric_card,
    render_insight, render_divider, render_assumption_box,
)
from src.streamlit import charts, data as D
from src.streamlit.scenarios import compute_confounders, compute_shap_surrogate, compute_surrogate_metrics


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
NUMERIC_FEATURE_CANDIDATES = [
    "price", "cogs", "Recency", "Frequency", "Monetary",
    "promo_order_rate", "return_rate", "customer_tenure_days",
    "gross_profit_historical", "P_Y_given_Promo", "P_Y_given_NoPromo",
]

TARGET_OPTIONS = ["Uplift_Score", "gross_profit_historical", "True_Y"]


# ---------------------------------------------------------------------------
# MAIN RENDER
# ---------------------------------------------------------------------------

def render(filters: dict) -> None:
    render_page_header(
        title="Phân tích Hồi Quy và Giải Thích Model",
        subtitle="Đánh giá mô hình · Biến nhiễu · Feature importance (SHAP) · Giải thích Model",
        # subtitle="",
        icon="",
        # timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

    # --- Load data ---
    with st.spinner("Loading uplift data for diagnostics…"):
        uplift_raw = D.load_uplift_enriched()

    if uplift_raw.empty:
        st.error("⚠️ Uplift predictions not found.")
        return

    # =========================================================
    # CONTROLS
    # =========================================================
    # render_section_header("Điều Khiển Phân Tích", "")

    

    # render_assumption_box(
    #     "Mô hình thay thế (Surrogate model) = LightGBM (hoặc GradientBoosting) được huấn luyện trên mẫu ≤3,000 dòng. Các chỉ số được đánh giá trên 20% dữ liệu kiểm thử. Giá trị SHAP tính trên mẫu ≤2,000 dòng. "
    #     "Điểm biến nhiễu (Confounder score) = |corr(f, T)| × |corr(f, Y)|."
    # )

    render_divider()


    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        available_targets = [t for t in TARGET_OPTIONS if t in uplift_raw.columns]
        target_y = st.selectbox("Biến mục tiêu (y)", available_targets, key="diag_target")

    with col_c2:
        x_candidates = [c for c in NUMERIC_FEATURE_CANDIDATES if c in uplift_raw.columns]
        x_axis = st.selectbox("Trục X (biểu đồ phân tán)", x_candidates, key="diag_xaxis")

    with col_c3:
        treatment_col = "True_T" if "True_T" in uplift_raw.columns else None
        outcome_col   = "True_Y" if "True_Y" in uplift_raw.columns else None
        confounder_k  = st.slider("Top-K Biến nhiễu", 5, 20, 10, key="diag_confk")

    # =========================================================
    # APPLY FILTERS
    # =========================================================
    df = uplift_raw.copy()
    if filters.get("segments") and "customer_segment" in df.columns:
        df = df[df["customer_segment"].isin(filters["segments"])]
    if filters.get("categories") and "category" in df.columns:
        df = df[df["category"].isin(filters["categories"])]

    n_total = len(df)
    if df.empty:
        st.warning("Không có dữ liệu sau khi lọc.")
        return

        

    # =========================================================
    # SURROGATE METRICS (R², MAE, RMSE)
    # =========================================================
    # render_section_header(f"Chỉ Số Đánh Giá Mô Hình Thay Thế — Mục tiêu: {target_y}", "")

    with st.spinner("Fitting surrogate model…"):
        if target_y == "True_Y" and treatment_col and outcome_col:
            # Use observed P_hat for True_Y (binary)
            sub = df[[treatment_col, "P_Y_given_Promo", "P_Y_given_NoPromo", outcome_col]].dropna()
            if not sub.empty:
                sub["p_hat"] = sub[treatment_col] * sub["P_Y_given_Promo"] + \
                               (1 - sub[treatment_col]) * sub["P_Y_given_NoPromo"]
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                import math
                r2   = round(r2_score(sub[outcome_col], sub["p_hat"]), 4)
                mae  = round(mean_absolute_error(sub[outcome_col], sub["p_hat"]), 4)
                rmse = round(math.sqrt(mean_squared_error(sub[outcome_col], sub["p_hat"])), 4)
                metrics = {"r2": r2, "mae": mae, "rmse": rmse}
            else:
                metrics = {}
        else:
            metrics = compute_surrogate_metrics(df, target_col=target_y)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        r2_val = metrics.get("r2", None)
        render_metric_card("R²", f"{r2_val:.4f}" if r2_val is not None else "N/A",
                           delta_positive=(r2_val or 0) > 0.5, icon="", accent="#4285F4")
    with k2:
        render_metric_card("MAE", f"{metrics.get('mae', 'N/A')}", icon="", accent="#FBBC05")
    with k3:
        render_metric_card("RMSE", f"{metrics.get('rmse', 'N/A')}", icon="", accent="#EA4335")
    with k4:
        render_metric_card("Cỡ Mẫu", f"{n_total:,}", icon="", accent="#34A853")

    render_divider()



    # =========================================================
    # SCATTER + REGRESSION
    # =========================================================
    col_sc, col_dist = st.columns([6, 4])
    with col_sc:
        # render_section_header(f"Biểu đồ phân tán: {x_axis} và {target_y}", "")
        sample_scatter = df[[x_axis, target_y]].dropna().sample(
            min(3000, len(df)), random_state=42
        )
        st.plotly_chart(
            charts.chart_scatter_regression(sample_scatter, x_axis, target_y),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_dist:
        # render_section_header(f"Phân phối: {target_y}", "")
        st.plotly_chart(
            charts.chart_distribution(df, target_y),
            use_container_width=True, config={"displayModeBar": False},
        )

    # render_divider()

    # =========================================================
    # CORRELATION HEATMAP
    # =========================================================
    # render_section_header("Bản Đồ Nhiệt Tương Quan", "")
    heatmap_cols = [c for c in [
        target_y, treatment_col, outcome_col,
        "price", "cogs", "Recency", "Frequency", "Monetary",
        "promo_order_rate", "return_rate", "Uplift_Score", "gross_profit_historical",
    ] if c and c in df.columns]

    st.plotly_chart(
        charts.chart_correlation_heatmap(df, heatmap_cols),
        use_container_width=True, config={"displayModeBar": False},
    )

    # render_divider()


    # =========================================================
    # CONFOUNDER RANKING
    # =========================================================
    # render_section_header("Xếp Hạng Biến Nhiễu (Liên kết Treatment × Outcome)", "")
    if treatment_col and outcome_col:
        with st.spinner("Computing confounder scores…"):
            confounder_df = compute_confounders(
                df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                numeric_cols=[c for c in NUMERIC_FEATURE_CANDIDATES if c in df.columns and c not in {treatment_col, outcome_col}],
            )
        confounder_df = confounder_df.head(confounder_k)
        col_cf1, col_cf2 = st.columns([6, 4])
        with col_cf1:
            st.plotly_chart(
                charts.chart_confounder_bar(confounder_df),
                use_container_width=True, config={"displayModeBar": False},
            )
        with col_cf2:
            st.dataframe(
                confounder_df.style.format({
                    "assoc_T": "{:.4f}", "assoc_Y": "{:.4f}", "confounder_score": "{:.5f}"
                }),
                use_container_width=True, height=400,
            )
        # Insight
        if not confounder_df.empty:
            top_conf = confounder_df.iloc[0]["feature"]
            st.markdown(f"""
            <div style='background:rgba(66, 133, 244, 0.08);border:1px solid rgba(66, 133, 244, 0.2);
            border-radius:8px;padding:0.75rem 1rem;font-size:0.82rem;color:black;margin-top:0.5rem'>
            Biến <strong>{top_conf}</strong> có điểm nhiễu (confounder score) cao nhất —
            nó tương quan với cả việc nhận khuyến mãi và quyết định mua hàng. 
            Có thể làm sai lệch kết quả A/B test, cần kiểm soát biến này trong phân tích nhân quả.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Cần có cột True_T và True_Y để phân tích biến nhiễu.")

    # render_divider()

    # =========================================================
    # SHAP SUMMARY
    # =========================================================
    render_section_header("Mức Độ Quan Trọng Của Biến (SHAP - Giải Thích Tổng Thể)", "")

    shap_info = st.empty()
    run_shap = st.button("Tính Toán SHAP", key="diag_shap_btn")

    if run_shap:
        with st.spinner("Fitting surrogate + computing SHAP values…"):
            shap_df = compute_shap_surrogate(df, target_col=target_y)
        if shap_df.empty:
            shap_info.warning("Lỗi khi tính toán SHAP. Hãy chắc chắn lightgbm hoặc scikit-learn đã được cài đặt.")
        else:
            shap_info.success(f"Giá trị SHAP được tính trên mẫu tối đa 2,000 dòng. Hiển thị Top {min(12, len(shap_df))} biến quan trọng.")
            col_sh1, col_sh2 = st.columns([6, 4])
            with col_sh1:
                st.plotly_chart(
                    charts.chart_shap_summary(shap_df),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with col_sh2:
                st.dataframe(
                    shap_df.head(15).style.format({"mean_abs_shap": "{:.5f}"}),
                    use_container_width=True, height=430,
                )
    # else:
        # shap_info.info("Bấm nút bên trên để tính toán giá trị SHAP")

    # render_divider()

    # =========================================================
    # FULL DATA TABLE
    # =========================================================
    # render_section_header("Xem Trước Dữ Liệu Lọc", "")
    # display_cols = [c for c in [
    #     "customer_id", "customer_segment", "category", "region",
    #     target_y, "True_T", "True_Y",
    #     "price", "cogs", "Recency", "Frequency", "Monetary",
    #     "Uplift_Score", "P_Y_given_Promo", "P_Y_given_NoPromo",
    #     "promo_order_rate", "return_rate",
    # ] if c and c in df.columns]

    # st.dataframe(
    #     df[display_cols].head(500).style.format({
    #         c: "{:.4f}" for c in ["Uplift_Score", "P_Y_given_Promo", "P_Y_given_NoPromo",
    #                                "promo_order_rate", "return_rate"]
    #         if c in display_cols
    #     }),
    #     use_container_width=True, height=360,
    # )

    # csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    # st.download_button(
    #     "Tải xuống Dữ liệu (CSV)",
    #     csv_bytes, "diagnostics_filtered.csv", "text/csv",
    #     key="diag_download",
    # )

    # render_divider()

    # =========================================================
    # INSIGHTS
    # =========================================================
    # render_section_header("Điểm Nhấn Chính", "")
    # _render_diag_insights(metrics, confounder_df if treatment_col else pd.DataFrame())


def _render_diag_insights(metrics: dict, conf_df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        r2 = metrics.get("r2")
        if r2 is not None:
            if r2 > 0.7:
                render_insight(f"R² của mô hình = <strong>{r2:.3f}</strong> — mức độ phù hợp cao. "
                               "Dự báo Uplift đáng tin cậy.", "success")
            elif r2 > 0.4:
                render_insight(f"R² của mô hình = <strong>{r2:.3f}</strong> — mức độ phù hợp trung bình. "
                               "Chỉ nên dùng kết quả AI để tham khảo định hướng, không dùng như một dự báo chính xác tuyệt đối.", "warning")
            else:
                render_insight(f"R² của mô hình = <strong>{r2:.3f}</strong> — mức độ phù hợp yếu. "
                               "Cần xem xét lại việc kỹ thuật đặc trưng (Feature Engineering) hoặc huấn luyện lại mô hình.", "danger")
    with col2:
        if not conf_df.empty:
            render_insight(
                f"Top các biến nhiễu được phát hiện: "
                + ", ".join([f"<strong>{c}</strong>" for c in conf_df["feature"].head(3).tolist()])
                + ". Những biến này cần được kiểm soát chặt chẽ trong phân tích nhân quả.", "warning"
            )
        render_insight(
            "Giải thích AI (XAI) liên kết giữa Khuyến Mãi (Trang B) và Vận Hành (Trang C): "
            "Biểu đồ SHAP giải thích <em>tại sao</em> một số khách hàng có mức độ phản hồi cao với khuyến mãi (uplift cao), "
            "giúp củng cố niềm tin vào các quyết định điều hướng bởi AI.", "info"
        )
