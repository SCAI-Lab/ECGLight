# utils/page_digitizer.py
"""
Page 1: ECG Image Digitizer.
Handles image upload, YOLO pipeline execution, and digitized signal export.
"""

import os
import tempfile
import streamlit as st


def render(config, digitization_runner, yolo_configured):
    """Render the ECG Image Digitizer page."""

    # Page heading
    st.markdown(
        '<h1 style="font-weight: 800; letter-spacing: -0.5px;">📷 <span class="glow-text">ECG Image Digitizer</span></h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="section-subtitle">'
        'Convert printed/scanned 12-lead ECG paper reports into high-resolution digitized CSV voltage signals.'
        '</p>',
        unsafe_allow_html=True
    )

    # Check YOLO models at UI rendering
    if not yolo_configured:
        st.warning(
            "⚠️ **YOLO Detection & Segmentation Models are not configured!**\n\n"
            "The model paths in `config.py` are currently empty string placeholders. "
            "You can still preview the image and explore the dashboard layout. "
            "To execute the digitization pipeline, please populate the model paths "
            "in your local `config.py` file with the correct YOLO weights paths."
        )

    # Use a clean 2-column layout (Left: Controls & Outputs, Right: Preview or Guide)
    col1, col2 = st.columns([2, 3])

    image_to_use = None
    image_name = ""

    # Initialize last uploaded file tracking to maintain consistent state
    if "last_uploaded_name" not in st.session_state:
        st.session_state["last_uploaded_name"] = None

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload ECG Scan")
        uploaded_file = st.file_uploader(
            "Select an ECG report image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # If a new file is uploaded, clear any stale state from previous runs
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                st.session_state.pop("digitized_df", None)
                st.session_state.pop("digitized_name", None)
                st.session_state["last_uploaded_name"] = uploaded_file.name

            # Save uploaded file to temp file to be processed by OpenCV/ultralytics
            tfile = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            )
            tfile.write(uploaded_file.read())
            tfile.close()
            image_to_use = tfile.name
            image_name = os.path.splitext(uploaded_file.name)[0]
        else:
            # Clear state if file is removed
            st.session_state.pop("digitized_df", None)
            st.session_state.pop("digitized_name", None)
            st.session_state["last_uploaded_name"] = None

        # Display results or trigger button
        if "digitized_df" in st.session_state:
            _show_results()
        elif image_to_use:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Digitization Control")
            st.markdown(
                "<p style='font-size: 0.88rem; color: #475569; margin-bottom: 15px;'>"
                "The image is ready. Click the button below to execute the digitization pipeline."
                "</p>",
                unsafe_allow_html=True
            )
            run_btn = st.button("⚡ Start Digitization Pipeline", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if run_btn:
                _run_pipeline(config, digitization_runner, image_to_use, image_name)

    with col2:
        if image_to_use:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ ECG Report Preview")
            st.image(image_to_use, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            _show_guide()


def _run_pipeline(config, digitization_runner, image_path, image_name):
    """Execute the digitization pipeline with progress tracking."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Executing Digitization...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.markdown("🔄 **Stage 1/5**: Loading models & preprocessing image...")
        progress_bar.progress(15)

        digitization_runner.check_models_configured(config)
        digitization_runner.check_model_files_exist(config)

        status_text.markdown("🔄 **Stage 2/5**: Executing lead segmentation & masking...")
        progress_bar.progress(35)

        status_text.markdown("🔄 **Stage 3/5**: Running sequential YOLO detectors (Leads, names, pulses)...")
        progress_bar.progress(55)

        df_digitized, ecg_inst = digitization_runner.run_digitization_pipeline(image_path, config)

        status_text.markdown("🔄 **Stage 4/5**: Calibrating scale from reference pulses...")
        progress_bar.progress(80)

        status_text.markdown("🔄 **Stage 5/5**: Building grid and extracting signals...")
        progress_bar.progress(95)

        progress_bar.progress(100)
        status_text.success("✅ **Digitization completed successfully!**")

        # Automatically save the digitized CSV to output directory
        os.makedirs(config.DIGITIZATION_OUTPUT_DIR, exist_ok=True)
        
        # Save both a named file and the shared "latest_digitized.csv" default file
        named_path = os.path.join(config.DIGITIZATION_OUTPUT_DIR, f"{image_name}_digitized.csv")
        default_path = os.path.join(config.DIGITIZATION_OUTPUT_DIR, "latest_digitized.csv")
        df_digitized.to_csv(named_path, index=False)
        df_digitized.to_csv(default_path, index=False)

        st.session_state["digitized_df"] = df_digitized
        st.session_state["digitized_name"] = image_name
        st.rerun()

    except digitization_runner.ModelPathNotSetError as e:
        st.error(str(e))
    except digitization_runner.ModelFileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"❌ Pipeline Failed: {str(e)}")
        st.exception(e)
    st.markdown('</div>', unsafe_allow_html=True)


def _show_results():
    """Display digitized signal results and export controls."""
    df_res = st.session_state["digitized_df"]
    name_res = st.session_state["digitized_name"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Digitization Complete")
    st.success("✅ **ECG signals extracted successfully!**")

    # Summary details
    st.markdown(f"**File Name:** `{name_res}`")
    st.markdown(f"**Total Samples:** `{len(df_res):,}`")
    st.markdown(f"**Detected Lead Channels ({len(df_res.columns)}):**")

    # Format leads as inline badges/code elements
    leads_str = " ".join([f"`{c}`" for c in df_res.columns])
    st.markdown(leads_str)

    st.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # Export controls
    csv_data = df_res.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Digitized Signals (CSV)",
        data=csv_data,
        file_name=f"{name_res}_digitized.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown(
        "<div style='margin-top: 15px; padding: 12px; background-color: #F0FDFA; border-left: 4px solid #0D9488; border-radius: 4px; font-size: 0.88rem; color: #0F766E; line-height: 1.5;'>"
        "💡 **Pro-Tip:** Navigate to the **📈 CSV Signal Viewer** page using the sidebar menu and upload this downloaded CSV file to visualize, customize plots, overlay, and inspect the waves!"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def _show_guide():
    """Display the guide/welcome panel when idle."""
    st.markdown("""
    <div class="glass-card">
        <h3 style="background: linear-gradient(135deg, #E63946, #0D9488); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 12px;">🩺 Digital Electrocardiography Conversion</h3>
        <p style="font-size: 0.92rem; line-height: 1.7; color: #475569;">
            Load an ECG paper report scan in the panel on the left to start the digitization pipeline.
            The deep learning models will automatically analyze the image to:
        </p>
        <ul style="font-size: 0.9rem; color: #64748B; margin-left: 20px; margin-top: 10px; line-height: 1.8;">
            <li>Detect individual 12-lead regions using YOLO object detection.</li>
            <li>Perform precise signal waveform segmentation with YOLO patch models.</li>
            <li>Identify lead labels (I, II, aVR, V1-V6) with a text classifier.</li>
            <li>Locate reference calibration pulses to calculate scale constants.</li>
            <li>Reconstruct physical digitized voltage values calibrated in millivolts (mV).</li>
        </ul>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 15px; font-style: italic;">
            Upload an ECG scan image to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)
