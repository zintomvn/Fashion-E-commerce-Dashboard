"""
components.py — Reusable UI components for DATATHON-2026 Dashboard.
Metric cards, sidebar panels, insight boxes, etc.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
/* === Google Font === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* === Root === */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* === Hide default Streamlit chrome === */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* === Metric Cards === */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.18);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent, #6C63FF);
    border-radius: 3px 0 0 3px;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5F6368;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #202124;
    line-height: 1.1;
}
.metric-delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}
.metric-delta.positive { color: #2ECC71; }
.metric-delta.negative { color: #E74C3C; }
.metric-delta.neutral  { color: #8B90A7; }
.metric-icon {
    font-size: 1.3rem;
    position: absolute;
    right: 1rem; top: 1rem;
    opacity: 0.25;
}

/* === Section Headers === */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2D3250;
}
.section-header h3 {
    font-size: 1.00rem;
    font-weight: 700;
    color: #202124;
    margin: 0;
    letter-spacing: 0.02em;
}

/* === Page Title === */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4285F4, #24C1E0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.1rem;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #8B90A7;
    margin-bottom: 1.2rem;
}

/* === Insight Cards === */
.insight-card {
    background: #1A1D2E;
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: #C8CADB;
    line-height: 1.5;
}
.insight-card.success  { border-color: #2ECC71; }
.insight-card.warning  { border-color: #F5A623; }
.insight-card.danger   { border-color: #E74C3C; }
.insight-card.info     { border-color: #6C63FF; }

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: #F8F9FA !important;
    border-right: 1px solid #E0E0E0 !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stMarkdown p {
    color: #202124 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    color: #202124 !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
    max-height: none !important;
    min-height: 80px !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-weight: 800 !important;
    color: #202124 !important;
    margin-bottom: 0.2rem;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiselect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stDateInput label {
    font-size: 0.7rem !important;
    font-weight: 800 !important;
    color: #202124 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] .element-container {
    margin-bottom: -0.8rem;
}
button[kind="header"] {
    color: #202124 !important;
}
.sidebar-brand {
    text-align: center;
    padding: 1rem 0.5rem 1.5rem;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 1.2rem;
}
.sidebar-brand h2 {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4285F4, #24C1E0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.25rem 0 0.1rem;
}
.sidebar-brand p {
    font-size: 0.7rem;
    color: #8B90A7;
    margin: 0;
}

/* === Tab styling === */
[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
.stTabs [role="tab"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #8B90A7 !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    color: #6C63FF !important;
}

/* === Dataframe === */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* === Divider === */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2D3250, transparent);
    margin: 1.5rem 0;
}

/* === Assumption box === */
.assumption-box {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.78rem;
    color: #A0A4BC;
    line-height: 1.6;
}
</style>
"""


# ---------------------------------------------------------------------------
# HELPER RENDERERS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: str,
    icon: str = "",
    timestamp: str | None = None,
) -> None:
    # cols = st.columns([1, 10])
    # with cols[0]:
    #     st.markdown(f"<div style='font-size:2.4rem;margin-top:0.2rem'>{icon}</div>",
    #                 unsafe_allow_html=True)
    # with cols[1]:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    ts_str = f" &nbsp;·&nbsp; <span style='color:#8B90A7;font-size:0.72rem'>Cập nhật: {timestamp}</span>" if timestamp else ""
    st.markdown(f"<div class='page-subtitle'>{subtitle}{ts_str}</div>", unsafe_allow_html=True)
    # st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_positive: bool | None = None,
    icon: str = "",
    accent: str = "#4285F4",
) -> None:
    delta_class = ""
    delta_html  = ""
    if delta is not None:
        if delta_positive is True:
            delta_class = "positive"; arrow = "▲"
        elif delta_positive is False:
            delta_class = "negative"; arrow = "▼"
        else:
            delta_class = "neutral"; arrow = "–"
        delta_html = f"<div class='metric-delta {delta_class}'>{arrow} {delta}</div>"

    html = f"""
    <div class='metric-card' style='--accent:{accent}'>
        <div class='metric-icon'>{icon}</div>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "") -> None:
    st.markdown(
        f"<div class='section-header'><span>{icon}</span><h3>{title}</h3></div>" if icon else f"<div class='section-header'><h3>{title}</h3></div>",
        unsafe_allow_html=True,
    )


def render_insight(text: str, kind: str = "info") -> None:
    """kind: 'success', 'warning', 'danger', 'info'."""
    icons = {"success": "Thành công:", "warning": "Cảnh báo:", "danger": "Rủi ro:", "info": "Chi tiết:"}
    icon = icons.get(kind, "Chi tiết:")
    st.markdown(
        f"<div class='insight-card {kind}'><strong>{icon}</strong> {text}</div>",
        unsafe_allow_html=True,
    )


def render_assumption_box(text: str) -> None:
    st.markdown(
        f"<div class='assumption-box'><strong>Giả định:</strong> {text}</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class='sidebar-brand'>
            <h2>VinCommerce AI</h2>
            <p>Phân Tích & Ra Quyết Định</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR FILTER PANEL
# ---------------------------------------------------------------------------

def sidebar_global_filters(
    orders_df=None,
    uplift_df=None,
    sales_df=None,
) -> dict:
    """
    Render global sidebar filters and return selected values as dict.
    """
    # render_sidebar_brand()

    st.sidebar.markdown("### Bộ Lọc Chung")

    filters = {}

    # --- Date range (from sales) ---
    if sales_df is not None and not sales_df.empty:
        min_dt = sales_df["Date"].min().date()
        max_dt = sales_df["Date"].max().date()
        date_range = st.sidebar.date_input(
            "Khoảng thời gian",
            value=(min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
            key="global_date_range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            filters["date_start"], filters["date_end"] = date_range
        else:
            filters["date_start"] = min_dt
            filters["date_end"]   = max_dt
    else:
        filters["date_start"] = None
        filters["date_end"]   = None

    st.sidebar.markdown("---")
    # st.sidebar.markdown(
    #     "<div style='font-size:0.68rem;color:#555;text-align:center'>"
    #     "DATATHON-2026 VinUni · Analytics Dashboard<br>"
    #     "Data © VinCommerce Vietnam"
    #     "</div>",
    #     unsafe_allow_html=True,
    # )

    return filters
