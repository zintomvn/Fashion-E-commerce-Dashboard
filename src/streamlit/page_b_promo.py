"""
page_b_promo.py — Page B: Promo What-If (Uplift Targeting)
Covers: Uplift distribution, Gain curve, Expected profit by segment, Waterfall, Targeting table.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.streamlit.components import (
    render_page_header, render_section_header, render_metric_card,
    render_insight, render_assumption_box, render_divider,
)
from src.streamlit import charts, data as D
from src.streamlit.scenarios import PromoEngine


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _fmt(val, prefix="₫"):
    if val is None: return "N/A"
    if abs(val) >= 1e9: return f"{prefix}{val/1e9:.2f}B"
    if abs(val) >= 1e6: return f"{prefix}{val/1e6:.1f}M"
    if abs(val) >= 1e3: return f"{prefix}{val/1e3:.1f}K"
    return f"{prefix}{val:,.0f}"


# ---------------------------------------------------------------------------
# MAIN RENDER
# ---------------------------------------------------------------------------

def render(filters: dict) -> None:
    render_page_header(
        title="Khuyến Mãi",
        subtitle="Mô phỏng lợi nhuận kỳ vọng từ uplift model",
        # icon="",
        # timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

    # --- Load data ---
    with st.spinner("Loading uplift predictions…"):
        uplift_raw = D.load_uplift_enriched()

    if uplift_raw.empty:
        st.error("⚠️ Uplift data not found. Ensure `outputs/uplift_predictions_test_set.csv` exists.")
        return

    # =========================================================
    # SIDEBAR PROMO CONTROLS (inline in page, not global sidebar)
    # =========================================================
    # st.markdown("### Cấu Hình Khuyến Mãi")
    
    st.sidebar.markdown("### Cấu Hình Khuyến Mãi")
    
    # 1. Targeting
    st.sidebar.markdown("**1. Khách hàng mục tiêu**")
    avail_segments = sorted(uplift_raw["customer_segment"].dropna().unique().tolist()) if "customer_segment" in uplift_raw else []
    selected_segments = st.sidebar.multiselect("Chọn Segment(s)", options=avail_segments, default=[], key="promo_segments")
    
    avail_cats = sorted(uplift_raw["category"].dropna().unique().tolist()) if "category" in uplift_raw else []
    selected_cats = st.sidebar.multiselect("Chọn Category/Segment", options=avail_cats, default=[], key="promo_cats")
    
    target_mode = st.sidebar.radio("Cắt theo độ lớn Uplift", ["Top-N", "Ngưỡng Uplift"], horizontal=True, key="promo_target_mode")
    if target_mode == "Top-N":
        top_n = st.sidebar.number_input("Top-N khách hàng", min_value=10, max_value=100000, value=1000, step=100, key="promo_topn")
        uplift_thr = None
    else:
        top_n = None
        uplift_thr = st.sidebar.slider("Ngưỡng Uplift (Uplift threshold)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key="promo_thr")
            
    # 2. Promo Policy
    st.sidebar.markdown("**2. Chính sách Khuyến mãi**")
    promo_type_map = {"percentage": "Phần trăm", "fixed": "Giá trị cố định"}
    promo_type = st.sidebar.selectbox("Loại khuyến mãi", ["percentage", "fixed"], format_func=lambda x: promo_type_map[x], key="promo_type")
    
    max_fixed = float(uplift_raw["price"].max() * 0.3) if ("price" in uplift_raw and not uplift_raw.empty) else 100000.0
    max_fixed = max(max_fixed, 100000.0)
    max_disc = 50.0 if promo_type == "percentage" else max_fixed
    default_val = 10.0 if promo_type == "percentage" else 50000.0
    
    discount_value = st.sidebar.number_input(
        "Mức giảm giá" + (" (%)" if promo_type == "percentage" else " (₫)"),
        min_value=0.0, max_value=max_disc,
        value=default_val,
        step=1.0 if promo_type == "percentage" else 10000.0,
        key="promo_discount",
    )
    min_order_value = st.sidebar.number_input("Min Order Value", min_value=0, value=0, step=50000, key="promo_min_order")
    
    stackable_flag = st.sidebar.checkbox("Cho phép cộng dồn", value=False, key="promo_stackable")
    penalise_returns = st.sidebar.checkbox("Loại nhóm có tỷ lệ rời bỏ cao", value=True, key="promo_penalise")


    # render_assumption_box(
    #     "Xác suất mua hàng P(Y|Promo) và P(Y|NoPromo) được giữ nguyên từ mô hình T-learner. "
    #     "Mức giảm giá chỉ ảnh hưởng biên lợi nhuận, không thay đổi xác suất mua. "
    #     "Có tính hình phạt dựa trên tỷ lệ trả hàng (nếu được bật)."
    # )

    # render_divider()

    # =========================================================
    # APPLY FILTERS
    # =========================================================
    uplift = uplift_raw.copy()

    # Local segment filter
    if selected_segments and "customer_segment" in uplift.columns:
        uplift = uplift[uplift["customer_segment"].isin(selected_segments)]

    # Local category filter
    if selected_cats and "category" in uplift.columns:
        uplift = uplift[uplift["category"].isin(selected_cats)]

    # Global segment filter (fallback/intersection)
    if filters.get("segments") and "customer_segment" in uplift.columns:
        uplift = uplift[uplift["customer_segment"].isin(filters["segments"])]

    # Global category filter (fallback/intersection)
    if filters.get("categories") and "category" in uplift.columns:
        uplift = uplift[uplift["category"].isin(filters["categories"])]

    if uplift.empty:
        st.warning("No rows after applying filters.")
        return

    # =========================================================
    # RUN PROMO ENGINE
    # =========================================================
    with st.spinner("Computing expected uplift profit…"):
        engine  = PromoEngine(uplift)
        try:
            computed = engine.compute(
                promo_type=promo_type,
                discount_value=discount_value,
                penalise_returns=penalise_returns,
                top_n=top_n,
                uplift_threshold=uplift_thr,
            )
            stats = engine.summary_stats(computed)
        except Exception as e:
            st.error(f"Promo engine error: {e}")
            return

    # =========================================================
    # KPI CARDS — Promo Summary
    # =========================================================
    # render_section_header("Tổng Hợp Tác Động Khuyến Mãi", "")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Khách hàng mục tiêu", f"{stats.get('n_targeted', 0):,}",
                           icon="", accent="#4285F4")
    with k2:
        render_metric_card("Tổng lợi nhuận tăng thêm", _fmt(stats.get("total_uplift")),
                           delta_positive=(stats.get("total_uplift") or 0) > 0,
                           icon="", accent="#34A853")
    with k3:
        render_metric_card("Lợi nhuận tăng thêm / Khách", _fmt(stats.get("avg_uplift")),
                           icon="", accent="#FBBC05")
    with k4:
        render_metric_card("% Uplift dương", f"{stats.get('pct_positive', 0):.1f}%",
                           delta_positive=(stats.get("pct_positive") or 0) > 50,
                           icon="", accent="#4285F4")
    with k5:
        roi = (stats.get("total_uplift", 0) / stats.get("total_base", 1)) * 100 if stats.get("total_base") else 0
        render_metric_card("ROI (Tỷ suất sinh lời)", f"{roi:.1f}%",
                           delta_positive=roi > 0, icon="", accent="#EA4335")

    render_divider()

    # =========================================================
    # CHARTS ROW 1
    # =========================================================
    col_l, col_r = st.columns(2)

    with col_l:
        # render_section_header("Phân Phối Điểm Uplift", "")
        group_opt = st.selectbox(
            "Nhóm theo", ["customer_segment", "category", "acquisition_channel"],
            key="promo_violin_group"
        )
        st.plotly_chart(
            charts.chart_uplift_violin(computed, group_by=group_opt),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_r:
        # render_section_header("Lợi Nhuận Tăng Thêm Theo Phân Khúc", "")
        profit_group = st.selectbox(
            "Nhóm theo", ["customer_segment", "category", "acquisition_channel"],
            key="promo_profit_group"
        )
        st.plotly_chart(
            charts.chart_expected_profit_by_segment(computed, group_by=profit_group),
            use_container_width=True, config={"displayModeBar": False},
        )

    render_divider()

    # =========================================================
    # CHARTS ROW 2
    # =========================================================
    col_g, col_w = st.columns([6, 4])

    with col_g:
        # render_section_header("Đường Cong Lợi Ích — Lợi Nhuận Tăng Thêm Lũy Kế", "")
        st.plotly_chart(
            charts.chart_gain_curve(computed),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_w:
        # render_section_header("Biểu Đồ Thác P&L", "")
        total_base_profit  = stats.get("total_base", 0)
        promo_margin_effect = stats.get("total_promo", 0) - stats.get("total_base", 0)
        uplift_effect       = stats.get("total_uplift", 0) - promo_margin_effect
        net                 = stats.get("total_promo", 0) + stats.get("total_uplift", 0)
        st.plotly_chart(
            charts.chart_promo_waterfall(
                total_base_profit, promo_margin_effect, uplift_effect, net
            ),
            use_container_width=True, config={"displayModeBar": False},
        )

    # render_divider()

    # =========================================================
    # TARGETING TABLE
    # =========================================================
    render_section_header("Danh Sách Khách Hàng Mục Tiêu (500 người)", "")

    display_cols = [c for c in [
        "customer_id", "customer_segment", "category", "acquisition_channel",
        "price", "promo_price", "cogs",
        "P_Y_given_Promo", "P_Y_given_NoPromo", "Uplift_Score",
        "expected_base", "expected_promo", "expected_uplift_profit",
        "return_rate",
    ] if c in computed.columns]

    table_df = computed[display_cols].head(500).copy().sort_values(by="Uplift_Score", ascending=False)
    # Format numeric cols
    for col in ["price", "promo_price", "cogs", "expected_base", "expected_promo", "expected_uplift_profit"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(0)

    st.dataframe(table_df, use_container_width=True, height=380)

    # Download
    csv_bytes = computed[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Tải xuống Danh sách mục tiêu (CSV)",
        data=csv_bytes,
        file_name="promo_targeting_list.csv",
        mime="text/csv",
        key="promo_download",
    )

    # render_divider()

    # =========================================================
    # INSIGHTS
    # =========================================================
    # render_section_header("Thông Tin Tự Động", "")
    # _render_promo_insights(stats, computed)


def _render_promo_insights(stats: dict, computed: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if (stats.get("total_uplift") or 0) > 0:
            render_insight(
                f"Tác động khuyến mãi lên <strong>{stats.get('n_targeted', 0):,}</strong> khách hàng dự kiến tạo ra "
                f"lợi nhuận tăng thêm là <strong>{_fmt(stats.get('total_uplift'))}</strong>.", "success"
            )
        else:
            render_insight("Lợi nhuận tăng thêm kỳ vọng đang âm. Hãy giảm mức chiết khấu hoặc tăng ngưỡng uplift.", "danger")

        pct_pos = stats.get("pct_positive", 0)
        if pct_pos < 50:
            render_insight(
                f"Chỉ <strong>{pct_pos:.1f}%</strong> khách hàng được nhắm mục tiêu có mức tăng thêm dương. "
                "Cần điều chỉnh Ngưỡng Uplift > 0 để cải thiện hiệu quả.", "warning"
            )

    with col2:
        if not computed.empty and "customer_segment" in computed.columns:
            best_seg = computed.groupby("customer_segment", observed=True)["expected_uplift_profit"].sum().idxmax()
            render_insight(f"Phân khúc mang lại hiệu quả cao nhất: <strong>{best_seg}</strong>.", "info")

        if not computed.empty and "category" in computed.columns:
            best_cat = computed.groupby("category", observed=True)["expected_uplift_profit"].sum().idxmax()
            render_insight(f"Danh mục sản phẩm tạo uplift tốt nhất: <strong>{best_cat}</strong>.", "info")
