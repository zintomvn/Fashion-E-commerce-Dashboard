"""
page_a_overview.py — Page A: Business Overview & Forecast
Covers: Revenue/COGS time series, Gross Profit, Order status, Lead time, BCG portfolio.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.streamlit.components import (
    render_page_header, render_section_header, render_metric_card,
    render_insight, render_divider,
)
from src.streamlit import charts, data as D


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _fmt(val: float, prefix: str = "₫", suffix: str = "", billions: bool = True) -> str:
    if val is None:
        return "N/A"
    if billions and abs(val) >= 1e9:
        return f"{prefix}{val/1e9:.2f}B{suffix}"
    elif abs(val) >= 1e6:
        return f"{prefix}{val/1e6:.1f}M{suffix}"
    elif abs(val) >= 1e3:
        return f"{prefix}{val/1e3:.1f}K{suffix}"
    return f"{prefix}{val:,.0f}{suffix}"


@st.cache_data(show_spinner=False)
def _compute_segment_matrix(items_df: pd.DataFrame, orders_full: pd.DataFrame, rfm: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """Efficiently compute the unit share cross-tab matrix."""
    if items_df.empty or orders_full.empty or rfm.empty or portfolio.empty:
        return pd.DataFrame()
        
    items_cust = items_df[["order_id", "product_id", "quantity"]].merge(
        orders_full[["order_id", "customer_id"]], on="order_id", how="inner"
    )
    items_seg = items_cust.merge(
        rfm[["customer_id", "customer_segment"]], on="customer_id", how="inner"
    )
    items_full_df = items_seg.merge(
        portfolio[["product_id", "product_segment_pv"]], on="product_id", how="inner"
    )
    
    if items_full_df.empty:
        return pd.DataFrame()
        
    cross_tab = pd.crosstab(
        items_full_df["customer_segment"], 
        items_full_df["product_segment_pv"], 
        values=items_full_df["quantity"],
        aggfunc="sum",
        normalize="index"
    )
    return cross_tab


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


# ---------------------------------------------------------------------------
# MAIN RENDER
# ---------------------------------------------------------------------------

def render(filters: dict) -> None:
    render_page_header(
        title="Tổng Quan Lợi Nhuận",
        # subtitle="Doanh thu · Lợi nhuận · Đơn hàng · Dự báo XGBoost",
        subtitle="",
        # icon="",
        # timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

    # --- Load data ---
    with st.spinner("Loading data…"):
        sales_full  = D.load_sales()
        forecast    = D.load_forecast()
        orders_full = D.load_orders_enriched()
        portfolio   = D.load_product_portfolio()

    # --- Date filter ---
    sales = sales_full.copy()
    if filters.get("date_start") and filters.get("date_end"):
        mask = (
            (sales["Date"].dt.date >= filters["date_start"]) &
            (sales["Date"].dt.date <= filters["date_end"])
        )
        sales = sales[mask]

    kpis = D.compute_kpis(sales)

    # =========================================================
    # KPI CARDS
    # =========================================================
    # render_section_header("Chỉ Số Hiệu Quả Kinh Doanh (KPI)", "")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        render_metric_card(
            "Tổng doanh thu", _fmt(kpis.get("total_revenue")),
            delta=_fmt_pct(kpis.get("yoy_growth")),
            delta_positive=(kpis.get("yoy_growth") or 0) > 0,
            icon="", accent="#4285F4",
        )
    with c2:
        render_metric_card(
            "Tổng giá vốn", _fmt(kpis.get("total_cogs")),
            icon="", accent="#EA4335",
        )
    with c3:
        render_metric_card(
            "Lợi nhuận gộp", _fmt(kpis.get("total_gp")),
            delta=f"Biên lợi nhuận: {kpis.get('avg_gm_pct', 0):.1f}%",
            delta_positive=(kpis.get("avg_gm_pct") or 0) > 40,
            icon="", accent="#34A853",
        )
    with c4:
        render_metric_card(
            "Tăng trưởng doanh thu (YoY)", _fmt_pct(kpis.get("yoy_growth")),
            delta_positive=(kpis.get("yoy_growth") or 0) > 0,
            icon="", accent="#FBBC05",
        )
    with c5:
        render_metric_card(
            "Doanh thu trung bình ngày", _fmt(kpis.get("avg_daily")),
            delta=_fmt_pct(kpis.get("mom_pct")) + " MoM" if kpis.get("mom_pct") else None,
            delta_positive=(kpis.get("mom_pct") or 0) > 0,
            icon="", accent="#4285F4",
        )

    render_divider()

    # =========================================================
    # 1. REVENUE & COGS TREND
    # =========================================================
    # render_section_header("Doanh Thu & Giá Vốn — Thực Tế + Dự Báo", "")

    col_a, col_b = st.columns([7, 3])
    # with col_a:
    show_cogs     = st.checkbox("Hiển thị Giá Vốn", value=True, key="ov_show_cogs")
    show_forecast = st.checkbox("Hiển thị dự báo", value=True, key="ov_show_fc")
    fig_rev = charts.chart_revenue_trend(
        sales,
        forecast=forecast if show_forecast else None,
        show_cogs=show_cogs,
    )
    st.plotly_chart(fig_rev, use_container_width=True, config={"displayModeBar": False})

    # with col_b:
    #     render_section_header("Tổng hợp Dự báo", "")
    #     if forecast is not None and not forecast.empty:
    #         fc_rev = forecast["Revenue"].sum()
    #         fc_gp  = forecast["GrossProfit"].sum() if "GrossProfit" in forecast.columns else None
    #         render_metric_card("Dự báo Doanh thu", _fmt(fc_rev), icon="", accent="#FBBC05")
    #         if fc_gp:
    #             render_metric_card("Dự báo LNG", _fmt(fc_gp), icon="", accent="#34A853")
    #         st.markdown("<br>", unsafe_allow_html=True)
    #         st.dataframe(
    #             forecast[["Date", "Revenue", "COGS"]].head(10).style.format({
    #                 "Revenue": "₫{:,.0f}", "COGS": "₫{:,.0f}"
    #             }),
    #             use_container_width=True, height=260,
    #         )

    # =========================================================
    # 2. GROSS PROFIT AREA
    # =========================================================
    # render_section_header("Xu hướng Lợi Nhuận Gộp", "")
    st.plotly_chart(
        charts.chart_gross_profit(sales),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # render_divider()

    # =========================================================
    # ORDER STATUS + LEAD TIME
    # =========================================================
    col_l, col_r = st.columns(2)
    orders = orders_full.copy()

    # Filter orders by date if possible
    if not orders.empty and "order_date" in orders.columns and filters.get("date_start"):
        mask_o = (
            (orders["order_date"].dt.date >= filters["date_start"]) &
            (orders["order_date"].dt.date <= filters["date_end"])
        )
        orders = orders[mask_o]

    # Biểu đồ trạng thái đơn hàng theo tháng
    st.plotly_chart(
        charts.chart_order_status_monthly(orders),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # render_divider()

    # =========================================================
    # RFM & SEGMENT MATRIX
    # =========================================================
    # render_section_header("Phân Tích RFM & Hành Vi Khách Hàng", "")
    rfm = D.load_rfm()
    items_df = D.load_order_items()
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.plotly_chart(
            charts.chart_rfm_clustering(rfm),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        
    with col_r2:
        cross_tab = _compute_segment_matrix(items_df, orders_full, rfm, portfolio)
        st.plotly_chart(
            charts.chart_segment_matrix(cross_tab),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # render_divider()

    # =========================================================
    # BCG PORTFOLIO
    # =========================================================
    # render_section_header("Danh Mục Sản Phẩm — Ma Trận BCG", "")
    col_p, col_q = st.columns([7, 3])
    # with col_p:
    st.plotly_chart(
        charts.chart_bcg_portfolio(portfolio),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    # with col_q:
    #     render_section_header("Tỷ trọng danh mục", "")
    #     if not portfolio.empty and "product_segment_pv" in portfolio.columns:
    #         seg_counts = portfolio["product_segment_pv"].value_counts().reset_index()
    #         seg_counts.columns = ["Segment", "Count"]
    #         import plotly.express as px
    #         color_map = {
    #             "Stars": "#F5A623", "Cash Cows": "#2ECC71",
    #             "Dogs": "#E74C3C", "Question Marks": "#6C63FF",
    #         }
    #         fig_pie = px.pie(
    #             seg_counts, values="Count", names="Segment",
    #             color="Segment", color_discrete_map=color_map,
    #             hole=0.5,
    #         )
    #         fig_pie.update_layout(
    #             paper_bgcolor="rgba(0,0,0,0)",
    #             plot_bgcolor="rgba(0,0,0,0)",
    #             font=dict(family="Inter, sans-serif", color="#E8EAF0"),
    #             showlegend=True, height=300,
    #             margin=dict(l=0, r=0, t=20, b=0),
    #         )
    #         st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # render_divider()

    # =========================================================
    # AUTO INSIGHTS
    # =========================================================
    # render_section_header("Thông Tin Kinh Doanh Tự Động", "")
    # _render_insights(kpis, sales, orders, portfolio)


def _render_insights(kpis: dict, sales: pd.DataFrame, orders: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    """Generate and display automated business insights."""
    cols = st.columns(2)

    insights_left = []
    insights_right = []

    # Revenue trend
    yoy = kpis.get("yoy_growth")
    if yoy is not None:
        kind = "success" if yoy > 0 else "danger"
        insights_left.append((f"Tăng trưởng doanh thu (YoY) đạt <strong>{yoy:+.1f}%</strong>. "
                               + ("Hoạt động kinh doanh đang mở rộng." if yoy > 0 else "Doanh thu đang thu hẹp — cần tìm hiểu nguyên nhân."),
                               kind))

    # Gross margin
    gm = kpis.get("avg_gm_pct")
    if gm is not None:
        if gm > 35:
            insights_left.append((f"Biên lợi nhuận gộp ở mức <strong>{gm:.1f}%</strong> rất khả quan. "
                                   "Tiếp tục duy trì kỷ luật chi phí.", "success"))
        else:
            insights_left.append((f"Biên lợi nhuận gộp <strong>{gm:.1f}%</strong> đang chịu áp lực. "
                                   "Cân nhắc tối ưu hóa giá vốn hoặc điều chỉnh giá bán.", "warning"))

    # MoM
    mom = kpis.get("mom_pct")
    if mom is not None:
        kind = "success" if mom > 0 else "warning"
        insights_right.append((f"Tăng trưởng doanh thu 30 ngày (MoM): <strong>{mom:+.1f}%</strong>. "
                                + ("Động lực tăng trưởng ngắn hạn tích cực." if mom > 0 else "Theo dõi chặt chẽ dấu hiệu sụt giảm."),
                                kind))

    # Portfolio
    if not portfolio.empty and "product_segment_pv" in portfolio.columns:
        n_dogs = (portfolio["product_segment_pv"] == "Dogs").sum()
        n_stars = (portfolio["product_segment_pv"] == "Stars").sum()
        if n_dogs > 0:
            insights_right.append((f"Có <strong>{n_dogs}</strong> sản phẩm thuộc nhóm <em>Chó mực (Dogs)</em>. "
                                    "Cân nhắc ngừng kinh doanh hoặc tái định vị.", "warning"))
        insights_right.append((f"<strong>{n_stars}</strong> sản phẩm Ngôi sao (Stars) đang thúc đẩy tăng trưởng. "
                                "Tập trung đầu tư tồn kho và marketing cho nhóm này.", "info"))

    # Orders
    if not orders.empty and "order_status" in orders.columns:
        total = len(orders)
        returns = (orders["order_status"].str.lower() == "returned").sum() if total > 0 else 0
        ret_rate = returns / total * 100 if total > 0 else 0
        if ret_rate > 15:
            insights_left.append((f"Tỷ lệ hoàn hàng là <strong>{ret_rate:.1f}%</strong> — vượt ngưỡng 15%. "
                                   "Cần điều tra chất lượng sản phẩm hoặc khâu giao hàng.", "danger"))
        else:
            insights_left.append((f"Tỷ lệ hoàn hàng ở mức <strong>{ret_rate:.1f}%</strong> nằm trong biên độ an toàn.", "success"))

    with cols[0]:
        for text, kind in insights_left:
            render_insight(text, kind)
    with cols[1]:
        for text, kind in insights_right:
            render_insight(text, kind)
