"""
streamlit_whatif_dashboard.py
─────────────────────────────
DATATHON-2026 VinUni · AI-Powered What-If Analytics Dashboard

Entrypoint:
    streamlit run streamlit_whatif_dashboard.py

Pages:
    A — Business Overview & Forecast
    B — Promo What-If (Uplift Targeting)
    C — Ops What-If (Counterfactual Profit)
    D — Regression Diagnostics & XAI

Architecture:
    src/streamlit/
        data.py        — Cached data loaders
        charts.py      — Plotly chart factory
        scenarios.py   — Promo & Ops engines + diagnostics helpers
        components.py  — Reusable UI components + CSS
        page_a_overview.py
        page_b_promo.py
        page_c_ops.py
        page_d_diagnostics.py
"""

import sys
import pathlib

# ---------------------------------------------------------------------------
# Path setup (allow running from project root without installing package)
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VinCommerce AI Dashboard · DATATHON-2026",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "DATATHON-2026 VinUni · Analytics & Decision Intelligence Dashboard",
    },
)

# ---------------------------------------------------------------------------
# Lazy imports (after path setup)
# ---------------------------------------------------------------------------
from src.streamlit.components import inject_css, sidebar_global_filters
from src.streamlit import data as D
import src.streamlit.page_a_overview   as page_a
import src.streamlit.page_b_promo      as page_b
import src.streamlit.page_c_ops        as page_c
import src.streamlit.page_d_diagnostics as page_d

# ---------------------------------------------------------------------------
# INJECT GLOBAL CSS
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# SIDEBAR — page navigation + global filters
# ---------------------------------------------------------------------------
PAGES = {
    "Tổng quan Lợi Nhuận":        "A",
    "Khuyến mãi":     "B",
    "Vận hành": "C",
    "Phân tích và Diễn giải":          "D",
}

# Pre-load lightweight data for sidebar filter options
with st.spinner("Initialising…"):
    _sales_df   = D.load_sales()
    _orders_df  = D.load_orders_enriched()
    _uplift_df  = D.load_uplift_enriched()

st.sidebar.markdown("## Trang")
selected_page_label = st.sidebar.radio(
    "Go to page",
    list(PAGES.keys()),
    key="nav_page",
    label_visibility="collapsed",
)
selected_page = PAGES[selected_page_label]


# Global filters
filters = sidebar_global_filters(
    orders_df=_orders_df,
    uplift_df=_uplift_df,
    sales_df=_sales_df,
)

# ---------------------------------------------------------------------------
# ROUTING
# ---------------------------------------------------------------------------
try:
    if selected_page == "A":
        page_a.render(filters)
    elif selected_page == "B":
        page_b.render(filters)
    elif selected_page == "C":
        page_c.render(filters)
    elif selected_page == "D":
        page_d.render(filters)
except Exception as exc:
    st.error(f"❌ Page error: {exc}")
    st.exception(exc)
