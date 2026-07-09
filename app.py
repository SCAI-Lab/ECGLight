# app.py
"""
Main entry point for the ECGLight Dashboard.
Slim router that delegates to page modules in utils/.
"""

import os
import sys
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import configuration
import config

# Import utility modules (CSS, branding, hardware, pages)
from utils.css import inject_custom_css
from utils.branding import render_sidebar_logo, render_logo_footer
from utils.hardware import render_sidebar_hardware
from utils import page_digitizer, page_csv_viewer, page_classifier

# Backend runners are imported lazily inside page routing blocks below
# to avoid loading torch/sklearn/scipy when the user only needs the CSV viewer.

# ==============================================================================
# PAGE CONFIG (must be the first Streamlit command)
# ==============================================================================
st.set_page_config(
    page_title="ECGLight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# INJECT CSS THEME
# ==============================================================================
inject_custom_css()

# ==============================================================================
# SIDEBAR
# ==============================================================================
# SCAI Lab logo at the top of the sidebar
render_sidebar_logo()

st.sidebar.markdown(
    '<h1 style="font-size: 1.8rem; margin-bottom: 5px; font-weight: 800; letter-spacing: -0.5px; text-align: center; padding-right: 1.5rem;">'
    '⚡ <span class="glow-text">ECGLight</span></h1>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<p class="section-subtitle" style="text-align: center;">Lightweight Digitization & Classification Suite</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Navigation (CSV Signal Viewer above Classification)
page = st.sidebar.radio(
    "🧭 Navigation Pages",
    ["📷 ECG Image Digitizer", "📈 ECG Signal Viewer", "❤️ ECG Classification"],
    index=0
)

# Keep track of active page to reset session state on page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = page

if st.session_state["current_page"] != page:
    # Clear session state keys to reset state on page change to prevent slowing things down,
    # but we DO NOT clear "latest_digitized_path" to share the default digitized CSV.
    keys_to_clear = [
        "digitized_df", "digitized_name", "last_uploaded_name",
        "clf_results", "clf_logs", "selected_task",
        "csv_viewer_upload", "classifier_csv_upload"
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    
    st.session_state["current_page"] = page
    st.rerun()

st.sidebar.markdown("---")

# Hardware info (cached — only runs once per session)
render_sidebar_hardware()

# ==============================================================================
# YOLO MODEL CHECK (used by Page 1)
# ==============================================================================
yolo_configured = all([
    config.YOLO_BOX_MODEL_PATH,
    config.YOLO_LEAD_NAME_MODEL_PATH,
    config.YOLO_PULSE_MODEL_PATH,
    config.YOLO_SEGMENTATION_MODEL_PATH
])

# ==============================================================================
# PAGE ROUTING (lazy backend imports — only load heavy modules when needed)
# ==============================================================================
if page == "📷 ECG Image Digitizer":
    from backend import digitization_runner
    page_digitizer.render(config, digitization_runner, yolo_configured)
    render_logo_footer()

elif page == "📈 ECG Signal Viewer":
    page_csv_viewer.render()
    render_logo_footer()

elif page == "❤️ ECG Classification":
    from backend import classification_runner
    page_classifier.render(config, classification_runner)
    render_logo_footer()
