# config.py
"""
Centralized configuration settings for the ECG Digitization and Classification Dashboard.
This holds paths, hyperparameter defaults, and metadata for classification tasks.
"""

import os

# ==============================================================================
# YOLO Models Settings (For Digitization)
# ==============================================================================
# These paths are initially left empty as requested. The user will set them in their environment 
# or directly in this file. The Streamlit app will warn if they are empty.
YOLO_BOX_MODEL_PATH = "models/digitization_models/yolo11_full/weights/best.pt"          # Bounding boxes for full lead regions
YOLO_LEAD_NAME_MODEL_PATH = "models/digitization_models/yolo11_lead/weights/best.pt"    # Classification for lead labels (I, II, etc.)
YOLO_PULSE_MODEL_PATH = "models/digitization_models/yolo11_pulse/weights/best.pt"        # Detection for calibration reference pulses
YOLO_SEGMENTATION_MODEL_PATH = "models/digitization_models/yolo11_patch/weights/best.pt" # Path to the YOLO patch segmentation model checkpoint

# ==============================================================================
# ECG Signal Metadata
# ==============================================================================
# Target sampling frequency for digitization and classification (in Hz)
TARGET_FS = 500

# Ordered standard list of 12 ECG Leads
LEAD_NAMES = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']

# ==============================================================================
# Classification Task Mapping
# ==============================================================================
# Maps the task selected in the Streamlit UI to its associated model and target class labels.
CLASSIFICATION_TASKS = {
    "Normal vs Myocardial Infarction (MI) - Segmented": {
        "model_dir": "mi_vs_normal_segmented",
        "model": "Arsenal",
        "labels": {0: "NORMAL", 1: "MYOCARDIAL_INFARCTION"},
        "description": "Differentiates normal healthy heartbeats from Myocardial Infarction using the pre-trained Arsenal ensemble classifier (Segmented Heartbeats)."
    },
    "Occlusive Myocardial Infarction (OMI) vs non-OMI": {
        "model_dir": "omi_vs_nonomi",
        "model": "Rocket",
        "labels": {0: "non-OMI", 1: "OMI"},
        "description": "Identifies Occlusive Myocardial Infarction (OMI) vs Non-Occlusive Myocardial Infarction/controls using the pre-trained Rocket classifier."
    },
    "Pre-Procedural vs Post-Procedural MI": {
        "model_dir": "ecg_surgery",
        "model": "InceptionTime",
        "labels": {0: "post-procedural MI", 1: "pre-procedural MI"},
        "description": "Classifies pre-procedural MI vs post-procedural MI using the pre-trained InceptionTime deep learning time-series classifier."
    }
}


# ==============================================================================
# File Management
# ==============================================================================
# Base output directories for storing run results
OUTPUT_BASE_DIR = "output"
DIGITIZATION_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "digitization")
CLASSIFICATION_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "classification")

# Ensure base directories exist
os.makedirs(DIGITIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_OUTPUT_DIR, exist_ok=True)
