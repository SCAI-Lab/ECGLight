# utils/page_csv_viewer.py
"""
Page 2: ECG Signal Viewer.
Upload any CSV file or view the latest digitized signals natively with interactive Streamlit charts.
Includes signal trimming tool for individual leads.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# Cached helpers — avoid re-parsing / re-computing on every Streamlit rerun
# ==============================================================================

@st.cache_data(show_spinner="Parsing CSV file...")
def _parse_csv(file_content: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame. Cached on raw file content."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_content))


@st.cache_data(show_spinner=False)
def _make_csv_bytes(df_values, df_columns) -> bytes:
    """Generate downloadable CSV bytes. Cached to avoid repeated .to_csv() calls."""
    df = pd.DataFrame(df_values, columns=df_columns)
    return df.to_csv(index=False).encode('utf-8')


def _downsample_for_plot(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Downsample a DataFrame for plotting by taking every nth row.
    
    Keeps every nth row so the total does not exceed ``max_points``.
    This prevents Streamlit's Vega-Lite renderer from choking on very
    large datasets while preserving the overall waveform shape.
    """
    n = len(df)
    if n <= max_points:
        return df
    step = max(1, n // max_points)
    return df.iloc[::step]


# ==============================================================================
# Main render entry point
# ==============================================================================

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

    # Load from uploaded file — cached to avoid re-parsing on every rerun
    df_signals = None
    if csv_file is not None:
        try:
            raw_bytes = csv_file.getvalue()
            df_signals = _parse_csv(raw_bytes)
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
            _render_trim_tool(df_signals, numeric_cols, sampling_rate)
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
    """Render the signal visualization section — stacked subplots only (one chart per lead)."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Interactive Signal Plots")
    st.markdown(
        "<p style='font-size: 0.88rem; color: #64748B; margin-top: -10px; margin-bottom: 20px;'>"
        "Zoom, pan, and hover over the interactive charts to inspect exact signal values. "
        "Each lead is plotted in its own chart for clarity."
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

    if selected_signals:
        n_samples = len(df)
        time_axis = np.arange(n_samples) / sampling_rate

        # Build plot dataframe and downsample for large datasets
        plot_df = df[selected_signals].copy()
        plot_df.index = time_axis
        plot_df.index.name = "Time (seconds)"

        plot_df = _downsample_for_plot(plot_df, max_points=5000)

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

        # Stacked layout: separate subplot per lead
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


# ==============================================================================
# Signal Trimming Tool
# ==============================================================================

def _render_trim_tool(df, numeric_cols, sampling_rate):
    """Render the signal trimming section.
    
    Allows the user to:
    - Select a specific lead to trim
    - Set start and end sample indices (or times)
    - Preview the trimmed segment
    - Apply the trim across all leads and download the updated CSV
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✂️ Signal Trimming Tool")
    st.markdown(
        "<p style='font-size: 0.88rem; color: #64748B; margin-top: -10px; margin-bottom: 20px;'>"
        "Select a lead to preview, choose start and end points, then apply the trim to all leads and download the result."
        "</p>",
        unsafe_allow_html=True
    )

    total_samples = len(df)
    max_time_s = total_samples / sampling_rate
    # Floor-round to 2 decimals so the default value never exceeds max_value
    max_time_s_rounded = float(np.floor(max_time_s * 100) / 100)

    col_lead, col_start, col_end = st.columns([1, 1, 1])

    with col_lead:
        trim_lead = st.selectbox(
            "Preview Lead",
            options=numeric_cols,
            index=0,
            help="The lead shown in the preview chart below. The trim will be applied to ALL leads."
        )

    with col_start:
        start_time = st.number_input(
            "Start Time (s)",
            min_value=0.0,
            max_value=max_time_s_rounded,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Start of the region to keep."
        )

    with col_end:
        end_time = st.number_input(
            "End Time (s)",
            min_value=0.0,
            max_value=max_time_s_rounded,
            value=max_time_s_rounded,
            step=0.01,
            format="%.2f",
            help="End of the region to keep."
        )

    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    # Clamp indices
    start_idx = max(0, min(start_idx, total_samples - 1))
    end_idx = max(start_idx + 1, min(end_idx, total_samples))

    trimmed_samples = end_idx - start_idx
    trimmed_duration = trimmed_samples / sampling_rate

    st.markdown(
        f"**Trim range**: sample `{start_idx:,}` → `{end_idx:,}` "
        f"({trimmed_samples:,} samples, {trimmed_duration:.2f} s)"
    )

    # Preview the trimmed region for the selected lead
    if trim_lead:
        preview_data = df[trim_lead].iloc[start_idx:end_idx].values
        preview_time = np.arange(len(preview_data)) / sampling_rate + start_time

        preview_df = pd.DataFrame({trim_lead: preview_data}, index=preview_time)
        preview_df.index.name = "Time (seconds)"

        preview_df = _downsample_for_plot(preview_df, max_points=5000)

        st.markdown(f"**Preview — Lead {trim_lead} (trimmed)**")

        lead_colors_map = {
            'I': '#E63946', 'II': '#0D9488', 'III': '#1E3A8A',
            'aVR': '#F59E0B', 'aVL': '#8B5CF6', 'aVF': '#EC4899',
            'V1': '#10B981', 'V2': '#3B82F6', 'V3': '#6366F1',
            'V4': '#F43F5E', 'V5': '#84CC16', 'V6': '#06B6D4'
        }
        preview_color = lead_colors_map.get(trim_lead, '#E63946')
        st.line_chart(preview_df, color=preview_color, height=220, use_container_width=True)

    # Action buttons
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        # Download trimmed CSV
        trimmed_df = df[numeric_cols].iloc[start_idx:end_idx]
        trimmed_csv = _make_csv_bytes(trimmed_df.values, trimmed_df.columns.tolist())
        st.download_button(
            label="📥 Download Trimmed CSV",
            data=trimmed_csv,
            file_name="ecg_trimmed.csv",
            mime="text/csv",
            use_container_width=True
        )

    with btn_col2:
        # Download the full dataset (all columns, trimmed rows)
        trimmed_full = df.iloc[start_idx:end_idx]
        trimmed_full_csv = _make_csv_bytes(trimmed_full.values, trimmed_full.columns.tolist())
        st.download_button(
            label="📥 Download Full Trimmed Dataset",
            data=trimmed_full_csv,
            file_name="ecg_trimmed_full.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# Export
# ==============================================================================

def _render_export(df, numeric_cols):
    """Render the export/download section."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Export Options")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_export = _make_csv_bytes(df[numeric_cols].values, list(numeric_cols))
        st.download_button(
            label="📥 Download Numeric Signals (CSV)",
            data=csv_export,
            file_name="ecg_signals.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp_col2:
        full_csv = _make_csv_bytes(df.values, list(df.columns))
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
            <li><strong>Stacked subplots</strong> — one chart per lead for clear side-by-side comparison</li>
            <li><strong>Signal trimming</strong> — select start/end times, preview the trimmed waveform, and download the result</li>
            <li><strong>Adjustable sampling rate</strong> to compute the time axis in seconds</li>
            <li><strong>Signal statistics</strong> including mean, std, min/max, and dynamic range</li>
        </ul>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 15px; font-style: italic;">
            Supported format: CSV with numeric columns. Non-numeric columns (e.g., class labels, timestamps)
            are automatically excluded from visualization.
        </p>
    </div>
    """, unsafe_allow_html=True)
