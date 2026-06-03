# utils/css.py
"""
Centralized CSS theming for the ECG Dashboard.
Injects the complete light clinical theme stylesheet.
"""

import streamlit as st


def inject_custom_css():
    """Inject the full custom CSS theme into the Streamlit page."""
    st.markdown("""
<style>
    /* ===== Light clinical theme variables ===== */
    :root {
        --accent: #E63946;
        --accent-light: #FFF1F2;
        --accent-dark: #BE1622;
        --teal: #0D9488;
        --teal-light: #CCFBF1;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --card-shadow: rgba(15, 23, 42, 0.06);
        --text-primary: #1E293B;
        --text-secondary: #64748B;
        --text-muted: #94A3B8;
        --bg-main: #F8FAFC;
        --bg-grid-major: rgba(226, 232, 240, 0.5);
        --bg-grid-minor: rgba(226, 232, 240, 0.25);
    }

    /* ===== Global: clean white with subtle ECG paper grid ===== */
    .stApp {
        background-color: var(--bg-main) !important;
        background-image:
            linear-gradient(var(--bg-grid-major) 1px, transparent 1px),
            linear-gradient(90deg, var(--bg-grid-major) 1px, transparent 1px),
            linear-gradient(var(--bg-grid-minor) 0.5px, transparent 0.5px),
            linear-gradient(90deg, var(--bg-grid-minor) 0.5px, transparent 0.5px);
        background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px;
        background-attachment: fixed;
    }

    /* ===== Typography ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ===== Header gradient text ===== */
    .glow-text {
        background: linear-gradient(135deg, #E63946 0%, #D62828 40%, #0D9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: inline-block;
    }

    /* ===== Elevated white cards ===== */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px var(--card-shadow), 0 4px 16px var(--card-shadow);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: #CBD5E1;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.1);
    }

    /* ===== Subtitle ===== */
    .section-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    /* ===== Metric cards ===== */
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
        flex-wrap: wrap;
        gap: 15px;
    }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: linear-gradient(135deg, #FFF1F2 0%, #FFFFFF 100%);
        border: 1px solid #FECDD3;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--accent);
        box-shadow: 0 4px 16px rgba(230, 57, 70, 0.12);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E63946, #0D9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-primary) !important;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, #E63946 0%, #D62828 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(230, 57, 70, 0.3) !important;
    }

    /* ===== Notifications ===== */
    div[data-testid="stNotification"] {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)
