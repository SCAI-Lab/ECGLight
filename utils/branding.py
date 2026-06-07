# utils/branding.py
"""
Logo and branding helpers for the ECG Dashboard.
Renders the SCAI Lab sidebar logo and the institutional footer logos.
All logo file reads are cached so they only happen once.
"""

import os
import base64
import streamlit as st

_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


@st.cache_data(show_spinner=False)
def _load_logo_b64(filename: str) -> str | None:
    """Load a logo file as base64 string. Cached per filename."""
    filepath = os.path.join(_ASSET_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def render_sidebar_logo():
    """Render the SCAI Lab logo above the dashboard title in the sidebar."""
    b64 = _load_logo_b64("scai_lab_logo.svg")
    if b64:
        st.sidebar.markdown(
            f'<div style="text-align: center; padding: 8px 0 4px 0;">'
            f'<img src="data:image/svg+xml;base64,{b64}" alt="SCAI Lab" '
            f'style="height: 65px;" />'
            f'</div>',
            unsafe_allow_html=True
        )


def render_logo_footer():
    """
    Render institutional logos as a footer bar at the bottom of each page.
    Order: ETH Zürich → SUPSI → Vanvitelli (per user preference).
    """
    # Ordered list: (filename, alt-text, height)
    logos = [
        ("ETH_Zürich_Logo_black.svg.png", "ETH Zürich", "32px"),
        ("Logo_of_SUPSI_Scuola_universitaria_professionale_della_Svizzera_italiana_01.png", "SUPSI", "38px"),
        ("Logo_Vanvitelli_university.svg.png", "Università della Campania Luigi Vanvitelli", "38px"),
    ]

    logo_html_items = []
    for filename, alt, height in logos:
        b64 = _load_logo_b64(filename)
        if b64:
            logo_html_items.append(
                f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
                f'style="height: {height}; opacity: 0.7; transition: opacity 0.2s ease;" '
                f'onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=0.7" />'
            )

    if logo_html_items:
        logos_row = "".join(logo_html_items)
        st.markdown("---")
        st.markdown(
            f'<div style="display: flex; align-items: center; justify-content: center; '
            f'gap: 36px; padding: 12px 0 8px 0; flex-wrap: wrap;">'
            f'{logos_row}</div>',
            unsafe_allow_html=True
        )
