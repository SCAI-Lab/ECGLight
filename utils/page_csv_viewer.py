# utils/page_csv_viewer.py
"""
Page 2: ECG Signal Viewer.
Upload any CSV file or view the latest digitized signals natively with interactive Streamlit charts.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st


def render():
    """Render the ECG Signal Viewer page."""

    # Page heading
    st.markdown(
        '<h1 style="font-weight: 800; letter-spacing: -0.5px;">📈 <span class="glow-text">ECG Signal Viewer</span></h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="section-subtitle">'
        'Upload any CSV file containing ECG signals or view the latest digitized signals using fully interactive charts.'
        '</p>',
        unsafe_allow_html=True
    )

    # --- Load Data Section ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Load Signal Data")

    col_upload, col_settings = st.columns([2, 1])

    with col_upload:
        csv_file = st.file_uploader(
            "Upload a CSV file with ECG signals",
            type=["csv"],
            key="csv_viewer_upload",
            help="Each column should represent a lead or signal channel. Rows represent sequential samples."
        )

    with col_settings:
        st.markdown("**Display Settings**")
        sampling_rate = st.number_input(
            "Sampling Rate (Hz)",
            min_value=1,
            max_value=10000,
            value=500,
            step=1,
            help="Used to compute a time axis in seconds for the x-axis."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Load from uploaded file
    df_signals = None
    if csv_file is not None:
        try:
            df_signals = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"❌ Error reading uploaded CSV: {e}")

    # --- Visualization & Summary ---
    if df_signals is not None and not df_signals.empty:
        numeric_cols = df_signals.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in df_signals.columns if c not in numeric_cols]

        if not numeric_cols:
            st.error("No numeric columns found in the CSV. ECG signals must be numeric data.")
        else:
            _render_summary(df_signals, numeric_cols, non_numeric, sampling_rate)
            _render_plots(df_signals, numeric_cols, sampling_rate)
            _render_export(df_signals, numeric_cols)
    else:
        _show_guide()


def _render_summary(df, numeric_cols, non_numeric, sampling_rate):
    """Render dataset summary card."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Summary")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Total Rows", f"{len(df):,}")
    with summary_cols[1]:
        st.metric("Signal Channels", f"{len(numeric_cols)}")
    with summary_cols[2]:
        duration_s = len(df) / sampling_rate
        st.metric("Duration", f"{duration_s:.2f} s")
    with summary_cols[3]:
        st.metric("Sampling Rate", f"{sampling_rate} Hz")

    if non_numeric:
        st.markdown(f"**Non-numeric columns** (excluded from visualization): `{non_numeric}`")

    with st.expander("🔍 Preview raw data (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _render_plots(df, numeric_cols, sampling_rate):
    """Render the signal visualization section using native Streamlit line charts."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Interactive Signal Plots")
    st.markdown(
        "<p style='font-size: 0.88rem; color: #64748B; margin-top: -10px; margin-bottom: 20px;'>"
        "Zoom, pan, and hover over the interactive charts to inspect exact signal values."
        "</p>",
        unsafe_allow_html=True
    )

    # Smart defaults: pick standard 12-lead names if present, else first 4
    standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    default_selection = [c for c in standard_leads if c in numeric_cols]
    if not default_selection:
        default_selection = numeric_cols[:min(4, len(numeric_cols))]

    selected_signals = st.multiselect(
        "Select signals/leads to visualize",
        options=numeric_cols,
        default=default_selection
    )

    plot_mode = st.radio(
        "Plot layout",
        ["Overlaid (all on one chart)", "Stacked (separate subplots per lead)"],
        horizontal=True
    )

    if selected_signals:
        n_samples = len(df)
        time_axis = np.arange(n_samples) / sampling_rate

        # Create copy of dataframe with selected signals and set time as index
        plot_df = df[selected_signals].copy()
        plot_df.index = time_axis
        plot_df.index.name = "Time (seconds)"

        # Define 12-lead premium clinical color palette
        lead_colors_map = {
            'I': '#E63946',    # Clinical Red
            'II': '#0D9488',   # Teal
            'III': '#1E3A8A',  # Deep Blue
            'aVR': '#F59E0B',  # Amber
            'aVL': '#8B5CF6',  # Violet
            'aVF': '#EC4899',  # Pink
            'V1': '#10B981',   # Emerald
            'V2': '#3B82F6',   # Blue
            'V3': '#6366F1',   # Indigo
            'V4': '#F43F5E',   # Rose
            'V5': '#84CC16',   # Lime
            'V6': '#06B6D4'    # Cyan
        }
        color_palette = [
            '#E63946', '#0D9488', '#1E3A8A', '#F59E0B', '#8B5CF6', '#EC4899',
            '#10B981', '#3B82F6', '#6366F1', '#F43F5E', '#84CC16', '#06B6D4'
        ]

        if plot_mode.startswith("Overlaid"):
            colors_list = [lead_colors_map.get(col, color_palette[i % len(color_palette)]) 
                           for i, col in enumerate(selected_signals)]
            st.line_chart(plot_df, color=colors_list, height=450, use_container_width=True)
        else:
            for i, lead in enumerate(selected_signals):
                st.markdown(f"**Lead {lead}**")
                lead_color = lead_colors_map.get(lead, color_palette[i % len(color_palette)])
                st.line_chart(plot_df[[lead]], color=lead_color, height=220, use_container_width=True)

        # Signal statistics
        with st.expander("📊 Signal Statistics"):
            stats_df = df[selected_signals].describe().T
            stats_df["range"] = stats_df["max"] - stats_df["min"]
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("Select one or more signal channels from the dropdown above to visualize.")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_export(df, numeric_cols):
    """Render the export/download section."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Export Options")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_export = df[numeric_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Numeric Signals (CSV)",
            data=csv_export,
            file_name="ecg_signals.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp_col2:
        full_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=full_csv,
            file_name="ecg_full_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _show_guide():
    """Display the placeholder guide when no CSV is uploaded."""
    st.markdown("""
    <div class="glass-card">
        <h3 style="background: linear-gradient(135deg, #0D9488, #E63946); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 12px;">📈 How to Use the ECG Signal Viewer</h3>
        <p style="font-size: 0.92rem; line-height: 1.7; color: #475569;">
            Upload any CSV file containing ECG signal data. The viewer supports:
        </p>
        <ul style="font-size: 0.9rem; color: #64748B; margin-left: 20px; margin-top: 10px; line-height: 1.8;">
            <li><strong>Multi-lead signals</strong> — each column is treated as a separate signal channel</li>
            <li><strong>Interactive plotting</strong> — zoom, pan, hover, and overlay features via native canvas-based Streamlit plots</li>
            <li><strong>Stacked or Overlaid modes</strong> for comparing waveforms</li>
            <li><strong>Adjustable sampling rate</strong> to compute the time axis in seconds</li>
            <li><strong>Signal statistics</strong> including mean, std, min/max, and dynamic range</li>
        </ul>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 15px; font-style: italic;">
            Supported format: CSV with numeric columns. Non-numeric columns (e.g., class labels, timestamps)
            are automatically excluded from visualization.
        </p>
    </div>
    """, unsafe_allow_html=True)
