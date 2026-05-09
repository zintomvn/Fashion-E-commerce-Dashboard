"""
page_c_ops.py — Page C: Ops What-If (Price / COGS / Inventory / Shipping / Returns)
Covers: Delta profit bar, Waterfall, Leaderboard, Supporting charts, Feature importance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.streamlit.components import (
    render_page_header, render_section_header, render_metric_card,
    render_insight, render_divider, render_assumption_box,
)
from src.streamlit import charts, data as D
from src.streamlit.scenarios import OpsEngine


def _fmt(val, prefix="₫"):
    if val is None: return "N/A"
    if abs(val) >= 1e6: return f"{prefix}{val/1e6:.1f}M"
    if abs(val) >= 1e3: return f"{prefix}{val/1e3:.1f}K"
    return f"{prefix}{val:,.0f}"


# ---------------------------------------------------------------------------
# MAIN RENDER
# ---------------------------------------------------------------------------

def render(filters: dict) -> None:
    render_page_header(
        title="Phân tích Vận Hành — Tối Ưu Hóa Lợi Nhuận",
        subtitle="Phân tích giả định theo các yếu tố: Giá, Giá vốn, Tồn kho, Vận chuyển, Hoàn hàng",
        # icon="",
        # timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

    # --- Load data ---
    with st.spinner("Loading counterfactual data…"):
        cf_raw      = D.load_counterfactual()
        best_raw    = D.load_best_actions()
        comp_models = D.load_component_models()
        returns_raw = D.load_returns()
        orders_raw  = D.load_orders_enriched()

    if cf_raw.empty:
        st.error("Counterfactual data not found. Ensure `outputs/profit_optimization_counterfactual_actions.csv` exists.")
        return

    engine = OpsEngine(cf_raw)

    # =========================================================
    # CONTROLS
    # =========================================================
    # render_section_header("Điều Khiển Kịch Bản", "")
    col1, col2, col3 = st.columns(3)

    with col1:
        all_segments = engine.segments
        selected_seg = st.selectbox("Phân khúc khách hàng", all_segments, key="ops_segment")

    with col2:
        def format_action_text(val):
            if pd.isna(val) or val == "(Tất cả)": return str(val)
            s = str(val)
            lower_s = s.lower()
            for sep in [" trong ", " cho ", " tại ", " của ", " ở "]:
                idx = lower_s.find(sep)
                if idx != -1:
                    main_part = s[:idx].strip()
                    sub_part = s[idx:].strip()
                    main_part = main_part[0].upper() + main_part[1:] if main_part else ""
                    return f"{main_part} ({sub_part})"
            return s[0].upper() + s[1:] if s else s

        all_actions = engine.actions
        selected_action = st.selectbox("Hành động", ["(Tất cả)"] + all_actions, format_func=format_action_text, key="ops_action")
        action_filter = None if selected_action == "(Tất cả)" else selected_action

    with col3:
        sort_map = {"Phần trăm thay đổi lợi nhuận": "delta_profit_pct", "Thay đổi lợi nhuận tuyệt đối": "delta_profit"}
        sort_label = st.radio("Sắp xếp", list(sort_map.keys()), key="ops_sort", horizontal=True)
        sort_by = sort_map[sort_label]

    # render_assumption_box(
    #     "Lợi nhuận kịch bản được tính toán từ phân tích đối ngẫu (counterfactual). "
    #     "Mỗi hành động đại diện cho một đòn bẩy vận hành (VD: giá +5%, chi phí -10%). "
    #     "delta_profit_pct = (Lợi nhuận kịch bản - Lợi nhuận cơ sở) / |Lợi nhuận cơ sở|."
    # )

    # render_divider()

    # Filter
    cf_filtered = engine.filter(segment=selected_seg, action=action_filter)
    cf_filtered = cf_filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # =========================================================
    # KPI CARDS
    # =========================================================
    best_row = engine.best_action(selected_seg)

    render_section_header(f"Kịch Bản Tốt Nhất — {selected_seg}", "")
    if best_row is not None:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card("Hành động tốt nhất", format_action_text(best_row.get("action", "N/A")),
                               icon="", accent="#FBBC05")
        with k2:
            render_metric_card("Lợi nhuận cơ sở", _fmt(best_row.get("baseline_profit")),
                               icon="", accent="#4285F4")
        with k3:
            render_metric_card("Lợi nhuận kịch bản", _fmt(best_row.get("scenario_profit")),
                               delta_positive=(best_row.get("delta_profit") or 0) > 0,
                               icon="", accent="#34A853")
        with k4:
            render_metric_card("Δ Lợi nhuận %", f"{best_row.get('delta_profit_pct', 0):+.2f}%",
                               delta_positive=(best_row.get("delta_profit_pct") or 0) > 0,
                               icon="", accent="#4285F4")
    else:
        st.info("Không có dữ liệu cho phân khúc này.")

    render_divider()

    # =========================================================
    # CHARTS ROW 1: Delta Bar + Waterfall
    # =========================================================
    col_l, col_r = st.columns([6, 4])

    with col_l:
        # render_section_header(f"% Thay Đổi Lợi Nhuận Theo Hành Động — {selected_seg}", "")
        st.plotly_chart(
            charts.chart_delta_profit_bar(cf_filtered, selected_segment=None),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_r:
        # render_section_header("Biểu Đồ Thác Kịch Bản", "")
        wf_row = engine.waterfall_inputs(selected_seg, action=action_filter)
        if wf_row is not None:
            st.plotly_chart(
                charts.chart_ops_waterfall(wf_row),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.info("Chọn một hành động cụ thể để xem biểu đồ thác.")

    # render_divider()

  
    # render_divider()

    # =========================================================
    # SUPPORTING CHARTS
    # =========================================================
    # render_section_header("Biểu Đồ Hỗ Trợ", "")
    # col_s1, col_s2 = st.columns(2)

    # with col_s1:
        # Return reason mix
    # if not returns_raw.empty and "return_reason" in returns_raw.columns:
    #     render_section_header("Phân Phối Lý Do Hoàn Hàng", "")
    #     reason_counts = returns_raw["return_reason"].value_counts().reset_index()
    #     reason_counts.columns = ["reason", "count"]
    #     import plotly.express as px
    #     fig_reason = px.bar(
    #         reason_counts.head(10),
    #         x="count", y="reason",
    #         orientation="h",
    #         color="count",
    #         color_continuous_scale=["#3498DB", "#6C63FF", "#E74C3C"],
    #     )
    #     fig_reason.update_layout(
    #         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    #         font=dict(family="Inter, sans-serif", color="#E8EAF0"),
    #         showlegend=False, height=320,
    #         coloraxis_showscale=False,
    #         margin=dict(l=20, r=20, t=30, b=20),
    #         yaxis=dict(gridcolor="#2D3250"),
    #         xaxis=dict(gridcolor="#2D3250"),
    #     )
    #     st.plotly_chart(fig_reason, use_container_width=True, config={"displayModeBar": False})
    # else:
    #     st.info("Dữ liệu hoàn hàng không khả dụng.")

    # with col_s2:
    #     # Lead time distribution (vs region)
    #     if not orders_raw.empty and "lead_time_days" in orders_raw.columns:
    #         render_section_header("Phân Phối Thời Gian Giao Hàng", "")
    #         st.plotly_chart(
    #             charts.chart_distribution(orders_raw, "lead_time_days"),
    #             use_container_width=True, config={"displayModeBar": False},
    #         )

    # render_divider()

    # =========================================================
    # FEATURE IMPORTANCE / EXPLAINABILITY
    # =========================================================
    render_section_header("Giải Thích Mô Hình — Mức Độ Quan Trọng Của Biến", "")
    if not comp_models.empty:
        col_e1, col_e2 = st.columns([5, 5])
        with col_e1:
            st.plotly_chart(
                charts.chart_feature_importance(comp_models, selected_seg),
                use_container_width=True, config={"displayModeBar": False},
            )
        with col_e2:
            # Show raw table
            seg_models = comp_models[comp_models["customer_segment"] == selected_seg]
            if not seg_models.empty:
                st.dataframe(
                    seg_models[["target", "best_model", "best_r2", "best_rmse", "top_mi", "top_importance"]].style.format({
                        "best_r2": "{:.3f}", "best_rmse": "{:,.1f}"
                    }),
                    use_container_width=True, height=280,
                )
    else:
        st.info("Dữ liệu mô hình thành phần không khả dụng.")


    # =========================================================
    # LEADERBOARD — All Segments
    # =========================================================
    render_section_header("Bảng Xếp Hạng — Hành Động Tốt Nhất Từng Phân Khúc", "")
    leaderboard = engine.leaderboard()
    if not leaderboard.empty:
        display_cols = [c for c in [
            "customer_segment", "action", "feature",
            "baseline_profit", "scenario_profit", "delta_profit", "delta_profit_pct",
            "delta_revenue", "delta_cost",
        ] if c in leaderboard.columns]
        
        display_df = leaderboard[display_cols].copy()
        if "action" in display_df.columns:
            display_df["action"] = display_df["action"].apply(format_action_text)

        st.dataframe(
            display_df.rename(columns={
                "baseline_profit": "Lợi nhuận cơ sở",
                "scenario_profit": "Lợi nhuận kịch bản",
                "delta_profit": "Thay đổi lợi nhuận",
                "delta_profit_pct": "% Thay đổi LN",
                "delta_revenue": "Thay đổi doanh thu",
                "delta_cost": "Thay đổi chi phí",
                "action": "Hành động",
                "feature": "Biến thay đổi",
                "customer_segment": "Phân khúc"
            }).style.format({
                "Lợi nhuận cơ sở": "₫{:,.0f}",
                "Lợi nhuận kịch bản": "₫{:,.0f}",
                "Thay đổi lợi nhuận":    "₫{:,.0f}",
                "% Thay đổi LN": "{:+.2f}%",
                "Thay đổi doanh thu":   "₫{:,.0f}",
                "Thay đổi chi phí":      "₫{:,.0f}",
            }),
            use_container_width=True, height=260,
        )


    # render_divider()

    # =========================================================
    # DETAILED TABLE
    # =========================================================
    # render_section_header("Bảng Chi Tiết Kịch Bản Vận Hành", "")
    # if not cf_filtered.empty:
    #     st.dataframe(
    #         cf_filtered.style.format({
    #             "baseline_profit": "₫{:,.0f}",
    #             "scenario_profit": "₫{:,.0f}",
    #             "delta_profit":    "₫{:,.0f}",
    #             "delta_profit_pct": "{:+.2f}%",
    #             "delta_revenue":   "₫{:,.0f}",
    #             "delta_cost":      "₫{:,.0f}",
    #         }),
    #         use_container_width=True, height=360,
    #     )
    #     csv_bytes = cf_filtered.to_csv(index=False).encode("utf-8")
    #     st.download_button(
    #         "Tải xuống kết quả kịch bản (CSV)",
    #         csv_bytes, "counterfactual_results.csv", "text/csv",
    #         key="ops_download",
    #     )

    # render_divider()

    # =========================================================
    # INSIGHTS
    # =========================================================
    render_section_header("Thông Tin Tự Động", "")
    _render_ops_insights(cf_filtered, selected_seg, best_row)


def _render_ops_insights(cf: pd.DataFrame, segment: str, best_row) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if best_row is not None:
            dp = best_row.get("delta_profit_pct", 0)
            kind = "success" if dp > 0 else "danger"
            render_insight(
                f"Hành động tối ưu cho <strong>{segment}</strong>: "
                f"<em>{best_row.get('action')}</em> → <em>{best_row.get('feature')}</em> "
                f"mang lại <strong>{dp:+.2f}%</strong> lợi nhuận tăng thêm.", kind
            )
        if not cf.empty:
            n_pos = (cf["delta_profit_pct"] > 0).sum()
            n_neg = (cf["delta_profit_pct"] <= 0).sum()
            render_insight(
                f"Có <strong>{n_pos}</strong> kịch bản cải thiện lợi nhuận, "
                f"<strong>{n_neg}</strong> kịch bản làm giảm lợi nhuận cho {segment}.",
                "info" if n_pos > n_neg else "warning"
            )
    with col2:
        if not cf.empty and "delta_revenue" in cf.columns:
            top_rev = cf.sort_values("delta_revenue", ascending=False).iloc[0]
            render_insight(
                f"Kịch bản tăng doanh thu mạnh nhất: <strong>{top_rev.get('action')} → {top_rev.get('feature')}</strong> "
                f"(Δ Doanh thu = {_fmt(top_rev.get('delta_revenue'))}).", "info"
            )
        if not cf.empty and "delta_cost" in cf.columns:
            top_cost = cf.sort_values("delta_cost").iloc[0]
            render_insight(
                f"Kịch bản giảm chi phí mạnh nhất: <strong>{top_cost.get('action')} → {top_cost.get('feature')}</strong> "
                f"(Δ Chi phí = {_fmt(top_cost.get('delta_cost'))}).", "success"
            )
