# utils/page_classifier.py
"""
Page 3: ECG Classification.
Handles CSV upload and pre-trained model inference.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render(config, classification_runner):
    """Render the Classification page."""

    # Page heading
    st.markdown(
        '<h1 style="font-weight: 800; letter-spacing: -0.5px;">❤️ <span class="glow-text">ECG Classification</span></h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="section-subtitle">'
        'Run diagnostic classification tasks (MI vs Normal, OMI vs non-OMI, Pre vs Post-Surgery) '
        'using advanced pre-trained models.'
        '</p>',
        unsafe_allow_html=True
    )

    # Check dependencies before proceeding
    dep_ok = True
    try:
        classification_runner.check_classification_dependencies()
    except classification_runner.DependencyMissingError as e:
        st.warning(str(e))
        st.info(
            "The application will show layout placeholders. "
            "Install the packages in your terminal and restart/refresh to activate."
        )
        dep_ok = False

    # 2-Column layout: Left panel (Controls), Right panel (Execution Steps & Results)
    col1, col2 = st.columns([2, 3])

    with col1:
        # 1. Upload section
        csv_to_use = _render_upload(config)
        
        # 2. Configuration & run button
        run_class_btn, selected_task, task_meta = _render_controls(config, classification_runner, csv_to_use, dep_ok)

    with col2:
        # Placeholder for real-time progress of running phases
        status_placeholder = st.empty()

        if run_class_btn:
            if not csv_to_use:
                st.error("Please upload a CSV dataset first.")
            else:
                _run_classification(config, classification_runner, csv_to_use, task_meta, selected_task, status_placeholder)

        # 3. Show results in the right panel if classification is completed and matches selected task
        if ("clf_results" in st.session_state and
                st.session_state.get("selected_task") == selected_task):
            _show_results(task_meta, selected_task)
        elif not run_class_btn:
            # Show a helpful guide card when idle
            _show_idle_guide()


def _render_upload(config):
    """Render the CSV upload section. Returns the path to the CSV file or None."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload ECG Signals Dataset")

    uploaded_csv = st.file_uploader(
        "Select digitized ECG CSV dataset (multi-subject format with lead names as columns)",
        type=["csv"],
        key="classifier_csv_upload"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    csv_to_use = None

    if uploaded_csv is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tfile.write(uploaded_csv.read())
        tfile.close()
        csv_to_use = tfile.name

    if csv_to_use:
        _render_dataset_preview(csv_to_use)

    return csv_to_use


def _render_dataset_preview(csv_path):
    """Show a preview of the loaded dataset."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Preview")
    try:
        df_preview = pd.read_csv(csv_path, nrows=100)
        st.dataframe(df_preview.head(10), use_container_width=True)

        row_count = len(pd.read_csv(csv_path, usecols=[0]))
        leads_found = [col for col in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                                        'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                       if col in df_preview.columns]
        st.markdown(
            f"**Dataset Summary:**\n"
            f"- **Total Rows**: `{row_count:,}` samples\n"
            f"- **Available Leads**: `{leads_found}`"
        )

        if 'class' in df_preview.columns:
            classes_found = pd.read_csv(csv_path, usecols=['class'])['class'].value_counts().to_dict()
            st.markdown(f"- **Class Distribution**: `{classes_found}`")
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def _render_controls(config, classification_runner, csv_to_use, dep_ok):
    """Render task selection and run button in left panel."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Task & Model Configuration")

    task_options = list(config.CLASSIFICATION_TASKS.keys())
    selected_task = st.selectbox(
        "Select Diagnostic Classification Task",
        options=task_options,
        index=0
    )

    task_meta = config.CLASSIFICATION_TASKS[selected_task]
    st.markdown(
        f"<p style='color: #64748B; font-size: 0.85rem; font-style: italic; "
        f"margin-top: -8px;'>{task_meta['description']}</p>",
        unsafe_allow_html=True
    )

    # Load pre-trained model metadata to display stats
    try:
        _, metadata = classification_runner.load_pretrained_model_and_metadata(task_meta['model_dir'])
        st.success(f"🎯 **Pre-trained model found!**")
        st.markdown(
            f"- **Model Class**: `{metadata['model_name']}`\n"
            f"- **Expected Input**: `{metadata['n_features']} leads x {metadata['n_timesteps']} timesteps`\n"
            f"- **Held-out Test Accuracy**: `{metadata['test_metrics']['accuracy']:.2%}`\n"
            f"- **Training Date**: `{metadata['training_date']}`"
        )
    except Exception as e:
        st.error(f"Could not load pre-trained model metadata: {e}")

    run_class_btn = st.button(
        "🎯 Run Diagnostic Classification",
        use_container_width=True, disabled=not dep_ok
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    return run_class_btn, selected_task, task_meta


def _run_classification(config, classification_runner, csv_path, task_meta, selected_task, status_placeholder):
    """Execute the classification pipeline, rendering logs inside status_placeholder."""
    with status_placeholder.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Running Classification Pipeline...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    logs = []
    try:
        status_text.markdown("🔄 **Phase 1/3**: Loading pre-trained model weights...")
        progress_bar.progress(20)
        model, metadata = classification_runner.load_pretrained_model_and_metadata(task_meta['model_dir'])
        logs.append(f"🎯 **Model Weights Loaded**: Loaded pre-trained `{metadata['model_name']}` for use-case `{metadata['use_case']}`.")

        status_text.markdown("🔄 **Phase 2/3**: Segmenting signals into heartbeats around R-peaks (Pan-Tompkins)...")
        progress_bar.progress(50)
        temp_segmented_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        segmented_df = classification_runner.segment_uploaded_csv(
            csv_path, temp_segmented_csv, target_fs=config.TARGET_FS
        )
        n_beats = segmented_df.index.get_level_values('subject_id').nunique() if not segmented_df.empty else 0
        logs.append(f"⚡ **Heartbeat Segmentation**: Segmented raw continuous signal into `{n_beats}` individual heartbeats around detected R-peaks (Pan-Tompkins).")

        status_text.markdown("🔄 **Phase 3/3**: Normalizing lead voltages & executing model inference...")
        progress_bar.progress(80)
        X, valid_ids, y_true = classification_runner.preprocess_dataframe_for_inference(segmented_df, metadata)
        logs.append(f"📊 **Data Preprocessing**: Formatted data into Numpy3D tensor shape `{X.shape}` and applied max-absolute voltage normalization.")

        results = classification_runner.run_pretrained_inference(model, X, metadata, valid_ids)
        logs.append(f"🔮 **Inference Engine**: Evaluated model predictions and computed confidence levels for `{len(valid_ids)}` inputs.")

        # Check if valid ground-truth labels are present
        model_labels = [str(x).strip().lower() for x in metadata['class_labels']]
        has_ground_truth = len(y_true) > 0 and any(str(lbl).strip().lower() in model_labels for lbl in y_true)
        
        if has_ground_truth:
            results['metrics'] = classification_runner.calculate_evaluation_metrics(
                y_true, results['predicted_class'], metadata['class_labels'], metadata['positive_class']
            )
            logs.append("📈 **Evaluation Metrics**: Calculated performance metrics (Accuracy, F1, Sensitivity, Specificity) using available ground-truth labels.")
        else:
            results['metrics'] = None

        progress_bar.progress(100)
        status_text.success("✅ **Classification Pipeline Completed!**")

        st.session_state["clf_results"] = results
        st.session_state["clf_logs"] = logs
        st.session_state["selected_task"] = selected_task

        # Clear progress bar card since finished results are rendering
        status_placeholder.empty()

    except classification_runner.DependencyMissingError as e:
        status_text.error(str(e))
    except Exception as e:
        status_text.error(f"❌ Classification Pipeline Failed: {str(e)}")
        st.exception(e)


def _show_results(task_meta, selected_task):
    """Display execution logs, performance metrics, confusion matrix, and predictions table."""
    results = st.session_state["clf_results"]
    logs = st.session_state.get("clf_logs", [])
    metrics = results.get("metrics")

    # 1. Pipeline Execution Status Logs
    if logs:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Pipeline Execution Logs")
        for log in logs:
            st.markdown(f"- {log}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Performance Metrics
    if metrics is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Diagnostic Classification Performance Metrics")

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{metrics['Accuracy']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{metrics['F1']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sensitivity</div>
                <div class="metric-value">{metrics['Sensitivity']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Specificity</div>
                <div class="metric-value">{metrics['Specificity']:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_cm1, col_cm2 = st.columns([1, 1])
        with col_cm1:
            st.markdown("**Confusion Matrix Counts:**")
            st.markdown(
                f"- **True Negatives (TN)**: `{metrics['TN']}`\n"
                f"- **False Positives (FP)**: `{metrics['FP']}`\n"
                f"- **False Negatives (FN)**: `{metrics['FN']}`\n"
                f"- **True Positives (TP)**: `{metrics['TP']}`"
            )
        with col_cm2:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            cm_arr = np.array(metrics["Confusion Matrix"])
            ax_cm.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Reds)

            class_labels_mapped = list(task_meta["labels"].values())
            ax_cm.set_xticks(range(len(class_labels_mapped)))
            ax_cm.set_yticks(range(len(class_labels_mapped)))
            ax_cm.set_xticklabels(class_labels_mapped, fontsize=8)
            ax_cm.set_yticklabels(class_labels_mapped, fontsize=8)
            ax_cm.set_xlabel("Predicted Label", fontsize=9)
            ax_cm.set_ylabel("True Label", fontsize=9)

            for r in range(cm_arr.shape[0]):
                for c in range(cm_arr.shape[1]):
                    ax_cm.text(c, r, f"{cm_arr[r, c]}",
                               ha="center", va="center",
                               color="white" if cm_arr[r, c] > (cm_arr.max() / 2) else "black",
                               fontweight="bold")

            fig_cm.patch.set_facecolor('white')
            ax_cm.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig_cm)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Diagnostic Classification Predictions")
        st.info("💡 **Inference Mode:** The uploaded dataset does not contain ground-truth class labels. Diagnosis outputs for each subject/heartbeat are listed below.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Predictions Table
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Heartbeat Classification Predictions")
    
    df_preds = pd.DataFrame({
        "Subject/Heartbeat ID": results['subject_id'],
        "Predicted Diagnosis": results['predicted_class']
    })
    if 'confidence' in results:
        df_preds["Model Confidence"] = [f"{c:.1%}" for c in results['confidence']]
        
    st.dataframe(df_preds, use_container_width=True)
    
    # Download predictions CSV
    preds_csv = df_preds.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions Table (CSV)",
        data=preds_csv,
        file_name=f"{selected_task.replace(' ', '_').lower()}_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def _show_idle_guide():
    """Display the instructions/guide in the right panel when idle."""
    st.markdown("""
    <div class="glass-card">
        <h3 style="background: linear-gradient(135deg, #E63946, #0D9488); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 12px;">❤️ Cardiac Diagnosis Workstation</h3>
        <p style="font-size: 0.92rem; line-height: 1.7; color: #475569;">
            This module applies pre-trained machine learning and deep learning models to classify cardiac conditions from digitized ECG voltage waveforms.
        </p>
        <p style="font-size: 0.92rem; line-height: 1.7; color: #475569;">
            <strong>Standard Workstation Steps:</strong>
        </p>
        <ul style="font-size: 0.9rem; color: #64748B; margin-left: 20px; margin-top: 10px; line-height: 1.8;">
            <li>Select a diagnostic task (e.g. OMI vs non-OMI) in the configuration panel on the left.</li>
            <li>Click <strong>🎯 Run Diagnostic Classification</strong> to initiate the pipeline.</li>
            <li>The system will segment continuous signals into separate heartbeats around R-peaks.</li>
            <li>Inference will run instantly using pre-trained weights.</li>
        </ul>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 15px; font-style: italic;">
            Upload a CSV in the configuration panel on the left to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)
