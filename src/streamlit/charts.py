"""
charts.py — Plotly chart factory for DATATHON-2026 Dashboard.
All functions return go.Figure objects with consistent enterprise styling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------
PALETTE = {
    "primary":   "#4285F4",
    "secondary": "#FBBC05",
    "success":   "#34A853",
    "danger":    "#EA4335",
    "info":      "#4285F4",
    "neutral":   "#9AA0A6",
    "bg":        "#0F1117",
    "card":      "#1E2130",
    "border":    "#2D3250",
    "text":      "#E8EAF0",
    "muted":     "#8B90A7",
}

SEGMENT_COLORS = {
    "Champions":                  "#4285F4",
    "Loyal Customers":            "#34A853",
    "Bargain Hunters":            "#FBBC05",
    "At Risk / Churn":            "#EA4335",
    "Hibernating / Low Engagement": "#9AA0A6",
}

BCG_COLORS = {
    "Stars":           "#FBBC05",
    "Cash Cows":       "#34A853",
    "Dogs":            "#EA4335",
    "Question Marks":  "#4285F4",
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Roboto, sans-serif", color="#333333", size=13),
    xaxis=dict(gridcolor="#E2E8F0", zeroline=False, showline=False),
    yaxis=dict(gridcolor="#E2E8F0", zeroline=False, showline=False),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    colorway=[
        "#4285F4", "#FBBC05", "#34A853", "#EA4335",
        "#A142F4", "#FA7B17", "#24C1E0", "#9AA0A6",
    ],
)

def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent enterprise theme to any figure."""
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color="#333333"), x=0.01, xanchor="left"),
    )
    return fig


def _fmt_vnd(val: float) -> str:
    """Format value as Vietnamese Dong abbreviated."""
    if abs(val) >= 1e9:
        return f"₫{val/1e9:.2f}B"
    elif abs(val) >= 1e6:
        return f"₫{val/1e6:.1f}M"
    elif abs(val) >= 1e3:
        return f"₫{val/1e3:.1f}K"
    return f"₫{val:,.0f}"


# ---------------------------------------------------------------------------
# PAGE A — OVERVIEW CHARTS
# ---------------------------------------------------------------------------

def chart_revenue_trend(
    sales: pd.DataFrame,
    forecast: pd.DataFrame | None = None,
    show_rolling: bool = True,
    show_cogs: bool = True
) -> go.Figure:
    """Dual-axis time series: Revenue & COGS with optional forecast overlay."""
    fig = go.Figure()

    # Actuals – Revenue
    fig.add_trace(go.Scatter(
        x=sales["Date"], y=sales["Revenue"],
        name="Doanh thu (Thực tế)",
        line=dict(color=PALETTE["primary"], width=1.5),
        opacity=0.8,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Doanh thu: %{customdata}<extra></extra>",
        customdata=[_fmt_vnd(v) for v in sales["Revenue"]],
    ))

    if show_rolling and "Rolling7_Revenue" in sales.columns:
        fig.add_trace(go.Scatter(
            x=sales["Date"], y=sales["Rolling7_Revenue"],
            name="Trung bình 7 ngày (Doanh thu)",
            line=dict(color=PALETTE["primary"], width=2.5, dash="dot"),
            opacity=1.0,
        ))

    if show_cogs:
        fig.add_trace(go.Scatter(
            x=sales["Date"], y=sales["COGS"],
            name="Giá vốn (Thực tế)",
            line=dict(color=PALETTE["danger"], width=1.5),
            opacity=0.6,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Giá vốn: %{customdata}<extra></extra>",
            customdata=[_fmt_vnd(v) for v in sales["COGS"]],
        ))

    # Forecast overlay
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["Date"], y=forecast["Revenue"],
            name="Doanh thu (Dự báo)",
            line=dict(color=PALETTE["secondary"], width=2),
        ))
        if show_cogs and "COGS" in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast["Date"], y=forecast["COGS"],
                name="Giá vốn (Dự báo)",
                line=dict(color="#FF6B6B", width=1.5),
                opacity=0.7,
            ))

        # Shade forecast region
        all_dates = pd.concat([sales["Date"], forecast["Date"]])
        split_date = forecast["Date"].min()
        fig.add_vrect(
            x0=split_date, x1=forecast["Date"].max(),
            fillcolor=PALETTE["secondary"], opacity=0.05,
            line_width=0, annotation_text="Dự báo",
            annotation_position="top left",
            annotation_font_color=PALETTE["secondary"],
        )

    _apply_theme(fig, "Doanh Thu & Giá Vốn — Thực Tế + Dự Báo")
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80),
        yaxis=dict(tickformat=",.0f"),
        height=420,
    )
    return fig


def chart_gross_profit(sales: pd.DataFrame) -> go.Figure:
    """Gross Profit area chart with rolling 7-day MA."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sales["Date"], y=sales["GrossProfit"],
        name="Lợi nhuận gộp",
        fill="tozeroy",
        fillcolor="rgba(66,133,244,0.15)",
        line=dict(color=PALETTE["primary"], width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Lợi nhuận gộp: %{customdata}<extra></extra>",
        customdata=[_fmt_vnd(v) for v in sales["GrossProfit"]],
    ))
    if "Rolling7_GP" in sales.columns:
        fig.add_trace(go.Scatter(
            x=sales["Date"], y=sales["Rolling7_GP"],
            name="Trung bình 7 ngày",
            line=dict(color=PALETTE["secondary"], width=2),
        ))
    _apply_theme(fig, "Lợi Nhuận Gộp (Doanh Thu − Giá Vốn)")
    fig.update_layout(
        height=300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80),
        yaxis=dict(tickformat=",.0f")
    )
    return fig


def chart_order_status_monthly(orders: pd.DataFrame) -> go.Figure:
    """Stacked bar: order_status by month."""
    if orders.empty or "order_date" not in orders.columns:
        return go.Figure()

    df = orders.copy()
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    grouped = df.groupby(["month", "order_status"], observed=True).size().reset_index(name="count")
    pivot = grouped.pivot(index="month", columns="order_status", values="count").fillna(0)

    fig = go.Figure()
    status_colors = {
        "delivered": PALETTE["success"],
        "shipped": PALETTE["info"],
        "cancelled": "#FA7B17",  # Orange
        "returned": PALETTE["danger"],
        "paid": PALETTE["neutral"], # Gray
        "created": "#FBBC05",    # Yellow
    }
    
    status_names = {
        "delivered": "Giao thành công",
        "shipped": "Đang giao",
        "cancelled": "Đã hủy",
        "returned": "Trả hàng",
        "paid": "Đã thanh toán",
        "created": "Đơn mới tạo",
    }
    
    for i, col in enumerate(pivot.columns):
        status_name = str(col).lower()
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[col],
            name=status_names.get(status_name, str(col).title()),
            marker_color=status_colors.get(status_name, PALETTE["neutral"]),
            hovertemplate="<b>%{y:,.0f}</b> đơn hàng<extra></extra>",
        ))
    _apply_theme(fig, "Phân Phối Trạng Thái Đơn Hàng (Theo Tháng)")
    fig.update_layout(
        barmode="stack", height=400, xaxis_tickangle=0,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80)
    )
    return fig


def chart_lead_time_box(orders: pd.DataFrame, group_by: str = "region") -> go.Figure:
    """Box plot of delivery lead time by region/city."""
    if orders.empty or "lead_time_days" not in orders.columns:
        return go.Figure()
    df = orders.dropna(subset=["lead_time_days", group_by])
    categories = df[group_by].value_counts().head(15).index.tolist()
    df = df[df[group_by].isin(categories)]

    fig = px.box(
        df, x=group_by, y="lead_time_days",
        color=group_by, color_discrete_sequence=list(PLOTLY_THEME["colorway"]),
        labels={"lead_time_days": "Số ngày", group_by: group_by.title()},
    )
    _apply_theme(fig, f"Thời Gian Giao Hàng Theo {group_by.title()}")
    fig.update_traces(marker=dict(outliercolor=PALETTE["danger"]))
    fig.update_layout(showlegend=False, height=340, xaxis_tickangle=-20)
    return fig


def chart_bcg_portfolio(portfolio: pd.DataFrame) -> go.Figure:
    """BCG scatter: gross_margin vs sales_volume with segment coloring."""
    if portfolio.empty:
        return go.Figure()
    df = portfolio.copy()
    color_map = {k: v for k, v in BCG_COLORS.items() if k in df["product_segment_pv"].values}

    fig = px.scatter(
        df,
        x="sales_volume", y="gross_margin",
        color="product_segment_pv",
        color_discrete_map=color_map,
        hover_name="product_name",
        hover_data={"category": True, "price": ":.0f", "cogs": ":.0f"},
        size_max=20,
        labels={"sales_volume": "Sản lượng bán (log scale)", "gross_margin": "Biên lợi nhuận gộp (%)"},
        log_x=True,
    )
    # Quadrant lines (medians)
    mx = df["sales_volume"].median()
    my = df["gross_margin"].median()
    fig.add_vline(x=mx, line_dash="dash", line_color=PALETTE["muted"], opacity=0.95)
    fig.add_hline(y=my, line_dash="dash", line_color=PALETTE["muted"], opacity=0.95)

    _apply_theme(fig, "Danh Mục Sản Phẩm — Ma Trận BCG")
    # Định dạng trục X log scale rõ ràng, không bị lỗi lặp minor ticks (2, 5)
    fig.update_xaxes(dtick=1, tickformat=",d")
    fig.update_layout(
        height=480, legend_title_text="Phân khúc",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80)
    )
    return fig


def chart_rfm_clustering(rfm: pd.DataFrame) -> go.Figure:
    """Scatter plot of RFM: Recency vs Monetary (log scale), colored by segment."""
    if rfm.empty or "recency" not in rfm.columns or "monetary" not in rfm.columns:
        return go.Figure()
    
    df = rfm.dropna(subset=["recency", "monetary", "customer_segment"])
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
        
    fig = px.scatter(
        df,
        x="recency", y="monetary",
        color="customer_segment",
        color_discrete_map=SEGMENT_COLORS,
        size="frequency" if "frequency" in df.columns else None,
        size_max=20,
        opacity=0.75,
        labels={
            "recency": "Số ngày từ lần mua gần nhất",
            "monetary": "Giá trị chi tiêu (log)",
            "customer_segment": "Phân khúc khách hàng",
            "frequency": "Tần suất mua"
        }
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color='rgba(0,0,0,0.5)')))
    fig.update_yaxes(type="log")
    _apply_theme(fig, "Phân Cụm RFM: Recency vs Monetary")
    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80)
    )
    return fig


def chart_segment_matrix(cross_tab: pd.DataFrame) -> go.Figure:
    """Heatmap of Unit Share: Customer Segment x Product Segment."""
    if cross_tab.empty:
        return go.Figure()
        
    # Order rows/cols if they exist
    row_order = ["Champions", "Loyal Customers", "At Risk / Churn", "Bargain Hunters", "Hibernating / Low Engagement"]
    col_order = ["Cash Cows", "Dogs", "Question Marks", "Stars"]
    
    rows = [r for r in row_order if r in cross_tab.index] + [r for r in cross_tab.index if r not in row_order]
    cols = [c for c in col_order if c in cross_tab.columns] + [c for c in cross_tab.columns if c not in col_order]
    
    cross_tab = cross_tab.loc[rows, cols]
    
    # Reverse rows so Champions is at the top
    cross_tab = cross_tab.iloc[::-1]
    
    fig = px.imshow(
        cross_tab.values,
        x=cross_tab.columns.tolist(),
        y=cross_tab.index.tolist(),
        color_continuous_scale=[
            [0.0, "#FFFFFF"],
            [1.0, PALETTE["primary"]]
        ],
        aspect="auto",
        text_auto=".1%",
        labels={"x": "Phân khúc sản phẩm", "y": "Phân khúc khách hàng", "color": "Tỷ trọng"}
    )
    _apply_theme(fig, "Tỷ trọng sản lượng: Khách hàng × Sản phẩm")
    fig.update_traces(xgap=1, ygap=1)
    fig.update_layout(height=500, coloraxis_showscale=True)
    return fig
    
# ---------------------------------------------------------------------------
# PAGE B — PROMO WHAT-IF CHARTS
# ---------------------------------------------------------------------------

def chart_uplift_violin(uplift: pd.DataFrame, group_by: str = "customer_segment") -> go.Figure:
    """Violin / box plot of Uplift_Score by segment."""
    if uplift.empty or "Uplift_Score" not in uplift.columns:
        return go.Figure()
    df = uplift.dropna(subset=[group_by, "Uplift_Score"])
    cats = df[group_by].value_counts().head(8).index.tolist()
    df = df[df[group_by].isin(cats)]
    colors = [SEGMENT_COLORS.get(c, PALETTE["info"]) for c in cats]

    fig = go.Figure()
    for i, cat in enumerate(cats):
        sub = df[df[group_by] == cat]["Uplift_Score"]
        fig.add_trace(go.Violin(
            y=sub, name=str(cat),
            box_visible=True, meanline_visible=True,
            fillcolor=colors[i], opacity=0.7,
            line_color=colors[i],
        ))
    _apply_theme(fig, f"Phân Phối Điểm Uplift Theo {group_by.replace('_', ' ').title()}")
    fig.update_layout(height=380, showlegend=False)
    return fig


def chart_gain_curve(uplift_sorted: pd.DataFrame) -> go.Figure:
    """Cumulative expected incremental profit gain curve."""
    if uplift_sorted.empty or "expected_uplift_profit" not in uplift_sorted.columns:
        return go.Figure()
    df = uplift_sorted.sort_values("expected_uplift_profit", ascending=False).reset_index(drop=True)
    df["cum_profit"] = df["expected_uplift_profit"].cumsum()
    df["pct_targeted"] = (df.index + 1) / len(df) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["pct_targeted"], y=df["cum_profit"],
        name="Đường cong lợi ích",
        line=dict(color=PALETTE["primary"], width=2.5),
        fill="tozeroy", fillcolor="rgba(66,133,244,0.1)",
        hovertemplate="Top %{x:.1f}% → Lợi nhuận lũy kế: %{customdata}<extra></extra>",
        customdata=[_fmt_vnd(v) for v in df["cum_profit"]],
    ))
    # Baseline random
    max_profit = df["cum_profit"].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, max_profit],
        name="Ngẫu nhiên (Cơ sở)",
        line=dict(color=PALETTE["muted"], dash="dash", width=1),
    ))
    _apply_theme(fig, "Đường Cong Lợi Ích (Gain Curve) — Lợi Nhuận Tăng Thêm Kỳ Vọng")
    fig.update_layout(
        height=360,
        xaxis_title="% Khách hàng mục tiêu",
        yaxis_title="Lợi Nhuận Tăng Thêm Lũy Kế",
        yaxis_tickformat=",.0f",
        legend=dict(orientation="h", yanchor="bottom", y=0.85, xanchor="right", x=1)
    )
    return fig


def chart_expected_profit_by_segment(uplift: pd.DataFrame, group_by: str = "customer_segment") -> go.Figure:
    """Horizontal bar: total expected uplift profit by segment."""
    if uplift.empty or "expected_uplift_profit" not in uplift.columns:
        return go.Figure()
    df = uplift.dropna(subset=[group_by])
    grouped = df.groupby(group_by, observed=True)["expected_uplift_profit"].sum().sort_values()
    colors = [SEGMENT_COLORS.get(c, PALETTE["info"]) for c in grouped.index]

    fig = go.Figure(go.Bar(
        x=grouped.values, y=grouped.index.astype(str),
        orientation="h",
        marker_color=colors,
        text=[_fmt_vnd(v) for v in grouped.values],
        # textposition="outside",
        textposition="auto",
        textangle=0,
        textfont=dict(size=12),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    _apply_theme(fig, f"Lợi Nhuận Tăng Thêm Kỳ Vọng Theo {group_by.replace('_', ' ').title()}")
    fig.update_layout(height=380, xaxis_title="Lợi Nhuận Tăng Thêm (₫)")
    return fig


def chart_promo_waterfall(
    base_profit: float,
    promo_margin_effect: float,
    uplift_effect: float,
    net_profit: float,
) -> go.Figure:
    """Waterfall: Baseline → Promo Margin → Uplift → Net."""
    categories = ["Cơ sở", "Tác động giảm biên lợi nhuận", "Tác động tăng số lượng", "Kỳ vọng ròng"]
    values     = [base_profit, promo_margin_effect, uplift_effect, net_profit]
    measures   = ["absolute", "relative", "relative", "total"]
    colors     = [PALETTE["info"], PALETTE["danger"], PALETTE["success"], PALETTE["primary"]]

    fig = go.Figure(go.Waterfall(
        x=categories, y=values, measure=measures,
        connector=dict(line=dict(color=PALETTE["border"])),
        textposition="outside",
        text=[_fmt_vnd(v) for v in values],
        increasing=dict(marker_color=PALETTE["success"]),
        decreasing=dict(marker_color=PALETTE["danger"]),
        totals=dict(marker_color=PALETTE["primary"]),
    ))
    _apply_theme(fig, "Biểu Đồ Thác Lợi Nhuận Khuyến Mãi")
    fig.update_layout(height=360)
    return fig


# ---------------------------------------------------------------------------
# PAGE C — OPS WHAT-IF CHARTS
# ---------------------------------------------------------------------------

def chart_delta_profit_bar(cf: pd.DataFrame, selected_segment: str | None = None) -> go.Figure:
    """Horizontal bar: delta_profit_pct per action for a segment."""
    if cf.empty:
        return go.Figure()
    df = cf.copy()
    if selected_segment:
        df = df[df["customer_segment"] == selected_segment]
    if df.empty:
        return go.Figure()

    df = df.sort_values("delta_profit_pct", ascending=True)
    colors = [PALETTE["success"] if v >= 0 else PALETTE["danger"] for v in df["delta_profit_pct"]]

    # Format action text for Y axis to be multiline
    def format_action(x):
        s = str(x)
        lower_s = s.lower()
        for sep in [" trong ", " cho ", " tại ", " của ", " ở "]:
            idx = lower_s.find(sep)
            if idx != -1:
                main_part = s[:idx].strip()
                sub_part = s[idx:].strip()
                main_part = main_part[0].upper() + main_part[1:] if main_part else ""
                return f"{main_part}<br><span style='font-size:10px;color:#8B90A7'>{sub_part}</span>"
        
        if len(s) > 24:
            space_idx = s.find(" ", 18)
            if space_idx != -1 and space_idx < 35:
                return f"{s[:space_idx]}<br><span style='font-size:10px;color:#8B90A7'>{s[space_idx+1:]}</span>"
            else:
                return f"{s[:22]}...<br><span style='font-size:10px;color:#8B90A7'>{s[22:]}</span>"
        
        return s[0].upper() + s[1:] if s else s
        
    y_labels = df["action"].apply(format_action)

    fig = go.Figure(go.Bar(
        x=df["delta_profit_pct"],
        y=y_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["delta_profit_pct"]],
        textposition="auto",
        insidetextanchor="end",
        customdata=df[["action", "feature"]].values,
        hovertemplate="<b>Hành động:</b> %{customdata[0]}<br><b>Biến:</b> %{customdata[1]}<br><b>Thay đổi Lơi nhuận:</b> %{x:+.2f}%<extra></extra>",
    ))
    _apply_theme(fig, f"% Thay Đổi Lợi nhuận Theo Hành Động — {selected_segment or 'Tất Cả'}")
    fig.update_layout(
        height=max(300, len(df) * 35 + 80), 
        xaxis_title="Δ Lợi nhuận (%)",
        margin=dict(l=150, r=60, t=40, b=40),
        yaxis=dict(automargin=True)
    )
    return fig


def chart_ops_waterfall(row: pd.Series, height: int = 380) -> go.Figure:
    """Waterfall chart for a single counterfactual row."""
    labels   = ["Lợi nhuận cơ sở", "Δ Doanh thu", "Δ Chi phí", "Lợi nhuận đề xuất"]
    values   = [
        row["baseline_profit"],
        row["delta_revenue"],
        -row["delta_cost"],
        row["scenario_profit"],
    ]
    measures = ["absolute", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        connector=dict(line=dict(color=PALETTE["border"])),
        textposition="outside",
        text=[_fmt_vnd(v) for v in values],
        increasing=dict(marker_color=PALETTE["success"]),
        decreasing=dict(marker_color=PALETTE["danger"]),
        totals=dict(marker_color=PALETTE["primary"]),
    ))
    _apply_theme(fig, "Kịch Bản Vận Hành: Hiện Tại → Đề Xuất")
    fig.update_layout(height=height)
    return fig


def chart_feature_importance(comp_models: pd.DataFrame, segment: str) -> go.Figure:
    """Parse top_mi string and render horizontal bar with Mutual Information scores."""
    if comp_models.empty:
        return go.Figure()
    df = comp_models[comp_models["customer_segment"] == segment]
    if df.empty:
        return go.Figure()

    # Parse comma-separated top_mi string
    features_raw = str(df["top_mi"].iloc[0])
    raw_items = features_raw.split(",")
    features = []
    scores = []
    
    for item in raw_items:
        if ":" in item:
            f, s = item.split(":", 1)
            features.append(f.strip().replace("num__", "").replace("cat__", ""))
            try:
                scores.append(float(s.strip()))
            except ValueError:
                scores.append(1.0)
        else:
            features.append(item.strip().replace("num__", "").replace("cat__", ""))
            scores.append(1.0)
            
    # Reverse so top is at the top of the horizontal bar chart
    features.reverse()
    scores.reverse()

    if len(scores) > 0 and scores[-1] != 1.0:
        # Use actual MI scores
        fig = go.Figure(go.Bar(
            x=scores, y=features,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, PALETTE["info"]], [1, PALETTE["primary"]]],
                showscale=False,
            ),
        ))
        _apply_theme(fig, f"Mức Độ Quan Trọng Của Biến (MI) — {segment}")
        fig.update_layout(height=max(300, len(features) * 28 + 80), xaxis_title="Điểm Mutual Information (MI)")
        return fig
    else:
        # Fallback to rank
        importances = list(range(1, len(features) + 1))
        fig = go.Figure(go.Bar(
            x=importances, y=features,
            orientation="h",
            marker=dict(
                color=importances,
                colorscale=[[0, PALETTE["info"]], [1, PALETTE["primary"]]],
                showscale=False,
            ),
        ))
        _apply_theme(fig, f"Mức Độ Quan Trọng Của Biến — {segment}")
        fig.update_layout(height=max(300, len(features) * 28 + 80), xaxis_title="Thứ hạng quan trọng (Mức độ tăng dần)")
        return fig


def chart_feature_importance_for_target(comp_models: pd.DataFrame, segment: str, target: str) -> go.Figure:
    """Render horizontal bar of feature importances for a specific target in a segment.
    
    Parses top_importance (comma-separated feature names) and normalises scores so
    the chart is always meaningful even without explicit numeric scores.
    """
    if comp_models.empty:
        return go.Figure()
    df = comp_models[(comp_models["customer_segment"] == segment) & (comp_models["target"] == target)]
    if df.empty:
        return go.Figure()

    row = df.iloc[0]

    # --- Try to get numeric scores from top_importance ---
    features_raw = str(row.get("top_importance", ""))
    raw_items = [x.strip() for x in features_raw.split(",") if x.strip()]

    features, scores = [], []
    for item in raw_items:
        if ":" in item:
            f, s = item.split(":", 1)
            features.append(f.strip().replace("num__", "").replace("cat__", ""))
            try:
                scores.append(float(s.strip()))
            except ValueError:
                scores.append(0.0)
        else:
            features.append(item.replace("num__", "").replace("cat__", ""))
            scores.append(0.0)

    if not features:
        return go.Figure()

    # If no numeric scores, assign rank-based scores (top = highest rank)
    if all(s == 0.0 for s in scores):
        scores = list(range(len(features), 0, -1))

    # Reverse so highest-importance feature appears at top of horizontal bar
    features = features[::-1]
    scores = scores[::-1]

    TARGET_LABELS = {
        "unit_price": "Giá bán (unit_price)",
        "quantity": "Số lượng (quantity)",
        "cost": "Chi phí (cost)",
    }
    title = f"{TARGET_LABELS.get(target, target)}"

    bar_colors = [PALETTE["primary"]] * len(features)
    bar_colors[-1] = PALETTE["success"]  # Highlight the most important feature

    fig = go.Figure(go.Bar(
        x=scores, y=features,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, PALETTE["info"]], [1, PALETTE["primary"]]],
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    _apply_theme(fig, title)
    fig.update_layout(
        height=max(280, len(features) * 30 + 80),
        xaxis_title="Mức độ quan trọng",
        margin=dict(l=10, r=10, t=50, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# PAGE D — DIAGNOSTICS / XAI CHARTS
# ---------------------------------------------------------------------------

def chart_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Annotated correlation heatmap."""
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return go.Figure()
    corr = df[available].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, PALETTE["danger"]],
            [0.5, "#FFFFFF"],
            [1.0, PALETTE["primary"]]
        ],
        zmid=0,
        zmin=-1, zmax=1,
        texttemplate="%{z:.2f}",
        hovertemplate="(%{x}, %{y}): %{z:.2f}<extra></extra>",
        colorbar=dict(title="r"),
        xgap=1, ygap=1,
    ))
    _apply_theme(fig, "Heatmap Tương Quan Các Biến")
    fig.update_layout(height=500)
    return fig


def chart_scatter_regression(
    df: pd.DataFrame, x_col: str, y_col: str
) -> go.Figure:
    """Scatter with OLS regression line."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    df_plot = df[[x_col, y_col]].dropna().copy()
    if len(df_plot) < 5:
        return go.Figure()

    x_vals = df_plot[x_col].values
    y_vals = df_plot[y_col].values
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    
    is_binary = len(np.unique(y_vals)) == 2 or y_col == "True_Y"
    
    if is_binary:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(x_vals.reshape(-1, 1), y_vals)
        y_line = model.predict_proba(x_line.reshape(-1, 1))[:, 1]
        line_name = "Hồi quy Logistic"
    else:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        y_line = np.polyval(coeffs, x_line)
        line_name = f"OLS (slope={coeffs[0]:.3f})"

    fig = go.Figure()
    
    if is_binary:
        jitter = np.random.uniform(-0.05, 0.05, size=len(y_vals))
        y_scatter = y_vals + jitter
    else:
        y_scatter = y_vals

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_scatter,
        mode="markers",
        marker=dict(color=PALETTE["primary"], opacity=0.4, size=4),
        name="Quan sát",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=PALETTE["secondary"], width=2),
        name=line_name,
    ))
    _apply_theme(fig, f"Biểu Đồ Phân Tán: {x_col} vs {y_col}")
    fig.update_layout(height=420, xaxis_title=x_col, yaxis_title=y_col, 
        # legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def chart_confounder_bar(confounder_df: pd.DataFrame) -> go.Figure:
    """Bar chart of confounder scores."""
    if confounder_df.empty:
        return go.Figure()
    df = confounder_df.sort_values("confounder_score", ascending=True).tail(15)
    colors_vals = df["confounder_score"].values
    max_c = max(colors_vals) if max(colors_vals) > 0 else 1

    fig = go.Figure(go.Bar(
        x=df["confounder_score"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=colors_vals,
            colorscale=[[0, "#E8F0FE"], [1, PALETTE["primary"]]],
            cmax=max_c, cmin=0, colorbar=dict(title="Điểm"),
        ),
        hovertemplate="%{y}: điểm=%{x:.4f}<br>assoc_T=%{customdata[0]:.3f}, assoc_Y=%{customdata[1]:.3f}<extra></extra>",
        customdata=df[["assoc_T", "assoc_Y"]].values,
    ))
    _apply_theme(fig, "Top Biến Gây Nhiễu")
    fig.update_layout(height=420)
    return fig


def chart_shap_summary(shap_df: pd.DataFrame) -> go.Figure:
    """Beeswarm-style SHAP summary as diverging bar (simplified)."""
    if shap_df.empty:
        return go.Figure()
    # shap_df: columns = [feature, mean_abs_shap]
    df = shap_df.sort_values("mean_abs_shap", ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["mean_abs_shap"],
            colorscale=[[0, PALETTE["info"]], [1, PALETTE["primary"]]],
            showscale=False,
        ),
    ))
    _apply_theme(fig, "Mức Độ Quan Trọng Theo SHAP (Trung bình |SHAP|)")
    fig.update_layout(height=420, xaxis_title="Trung bình |Giá trị SHAP|")
    return fig


def chart_distribution(df: pd.DataFrame, col: str, nbins: int = 40) -> go.Figure:
    """Histogram + KDE (approximated with box-plot overlay)."""
    if df.empty or col not in df.columns:
        return go.Figure()
    vals = df[col].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=nbins, name=col,
        marker_color=PALETTE["primary"],
        opacity=0.75,
    ))
    _apply_theme(fig, f"Phân Phối Của {col}")
    fig.update_layout(height=420, xaxis_title=col, yaxis_title="Số lượng")
    return fig


def chart_box_by_class(df: pd.DataFrame, num_col: str, class_col: str) -> go.Figure:
    """Boxplot of a numeric feature grouped by binary class."""
    if df.empty or num_col not in df.columns or class_col not in df.columns:
        return go.Figure()
    
    df_plot = df[[num_col, class_col]].dropna().copy()
    # Convert class to string for categorical axis
    df_plot[class_col] = df_plot[class_col].astype(int).astype(str)
    
    fig = px.box(
        df_plot, x=class_col, y=num_col,
        color=class_col, 
        color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"]],
        labels={class_col: "Lớp (Class)", num_col: num_col}
    )
    _apply_theme(fig, f"Phân Bố {num_col} Theo {class_col}")
    fig.update_layout(height=420, showlegend=False)
    return fig

def chart_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Heatmap for Confusion Matrix."""
    if cm is None or len(cm) != 2:
        return go.Figure()
    
    # Calculate percentages for annotations
    cm_pct = cm / cm.sum() * 100
    
    labels = ["0 (Negative)", "1 (Positive)"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#FFFFFF"], [1, PALETTE["primary"]]],
        text=[[f"{cm[i][j]:,} ({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        hovertemplate="Thực tế: %{y}<br>Dự đoán: %{x}<br>Số lượng: %{z}<extra></extra>",
        showscale=False,
        xgap=1, ygap=1,
    ))
    
    # Update layout
    _apply_theme(fig, "Ma Trận Nhầm Lẫn (Confusion Matrix)")
    fig.update_layout(
        height=420,
        xaxis_title="Dự Đoán",
        yaxis_title="Thực Tế",
        yaxis=dict(autorange="reversed")
    )
    return fig


def chart_roc_curve(fpr, tpr, auc: float) -> go.Figure:
    """ROC Curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f"ROC (AUC = {auc:.3f})",
        line=dict(color=PALETTE["primary"], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Ngẫu nhiên",
        line=dict(color=PALETTE["muted"], width=2, dash="dash")
    ))
    _apply_theme(fig, "Đường Cong ROC")
    fig.update_layout(
        height=420,
        xaxis_title="Tỷ lệ Dương Tính Giả (FPR)",
        yaxis_title="Tỷ lệ Dương Tính Thật (TPR)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80)
    )
    return fig

