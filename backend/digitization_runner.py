# backend/digitization_runner.py
"""
Backend adapter for ECG Digitization.
Wraps the ECGImage pipeline from digitization.py and manages model caching.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import torch

# Add parent directory to path to ensure digitization and other local modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from digitization import ECGImage
    from ultralytics import YOLO
except ImportError as e:
    # We will log the error but keep running so the dashboard can load and display elegant warnings
    pass

class ModelPathNotSetError(Exception):
    """Raised when one or more YOLO model paths are not set in the configuration."""
    pass

class ModelFileNotFoundError(Exception):
    """Raised when a specified YOLO model file does not exist on disk."""
    pass

def check_models_configured(config):
    """
    Verify that all YOLO model paths are set in the configuration.
    """
    missing_configs = []
    if not config.YOLO_BOX_MODEL_PATH:
        missing_configs.append("YOLO Box Model (YOLO_BOX_MODEL_PATH)")
    if not config.YOLO_LEAD_NAME_MODEL_PATH:
        missing_configs.append("YOLO Lead Name Model (YOLO_LEAD_NAME_MODEL_PATH)")
    if not config.YOLO_PULSE_MODEL_PATH:
        missing_configs.append("YOLO Pulse Model (YOLO_PULSE_MODEL_PATH)")
    if not config.YOLO_SEGMENTATION_MODEL_PATH:
        missing_configs.append("YOLO Segmentation Model (YOLO_SEGMENTATION_MODEL_PATH)")
        
    if missing_configs:
        raise ModelPathNotSetError(
            f"The following YOLO model paths are not set in config.py:\n" + 
            "\n".join([f"- {m}" for m in missing_configs]) + 
            "\n\nPlease configure these paths in config.py before running digitization."
        )

def check_model_files_exist(config):
    """
    Verify that the configured model files actually exist on disk.
    """
    missing_files = []
    for name, path in [
        ("Box Model", config.YOLO_BOX_MODEL_PATH),
        ("Lead Name Model", config.YOLO_LEAD_NAME_MODEL_PATH),
        ("Pulse Model", config.YOLO_PULSE_MODEL_PATH),
        ("Segmentation Model", config.YOLO_SEGMENTATION_MODEL_PATH)
    ]:
        if path and not os.path.exists(path):
            missing_files.append(f"{name} at path: '{path}'")
            
    if missing_files:
        raise ModelFileNotFoundError(
            f"The following model files were not found on disk:\n" + 
            "\n".join([f"- {f}" for f in missing_files]) + 
            "\n\nPlease verify that the files are downloaded and located at these paths."
        )

@st.cache_resource
def load_yolo_models(box_path, lead_name_path, pulse_path):
    """
    Load YOLO models once and cache them in Streamlit memory.
    Returns a dict containing the loaded model objects.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load each model
    box_model = YOLO(box_path)
    lead_name_model = YOLO(lead_name_path)
    pulse_model = YOLO(pulse_path)
    
    # Move to GPU if available
    box_model.to(device)
    lead_name_model.to(device)
    pulse_model.to(device)
    
    return {
        "box_model": box_model,
        "lead_name_model": lead_name_model,
        "pulse_model": pulse_model,
        "device": device
    }

def run_digitization_pipeline(image_path, config):
    """
    Runs the full digitization pipeline on a single ECG image.
    
    Parameters
    ----------
    image_path : str
        Path to the uploaded image file.
    config : module
        The config module containing paths and settings.
        
    Returns
    -------
    df : pandas.DataFrame
        Digitized ECG signals with lead names as columns and timesteps as rows.
    ecg_instance : ECGImage
        The raw ECGImage instance for accessing signal_grid, masks, or metadata.
    """
    # 1. Validation
    check_models_configured(config)
    check_model_files_exist(config)
    
    # 2. Load models (cached via st.cache_resource)
    models = load_yolo_models(
        config.YOLO_BOX_MODEL_PATH,
        config.YOLO_LEAD_NAME_MODEL_PATH,
        config.YOLO_PULSE_MODEL_PATH
    )
    
    # 3. Instantiate ECGImage
    ecg = ECGImage(
        box_model=models["box_model"],
        segmentation_model=config.YOLO_SEGMENTATION_MODEL_PATH,
        lead_name_model=models["lead_name_model"],
        pulse_model=models["pulse_model"],
        image_path=image_path
    )
    
    # 4. Run the pipeline (without Linux-specific SIGALRM timeouts)
    ecg.run_full_pipeline()
    
    # 5. Extract DataFrame from signal_grid
    df = extract_dataframe_from_ecg(ecg)
    
    # 6. Clear GPU cache if needed
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
            
    return df, ecg

def extract_dataframe_from_ecg(ecg_instance):
    """
    Extracts signal data from an ECGImage instance and converts it to a pandas DataFrame.
    """
    signals, lead_names, max_length = [], [], 0
    for row in ecg_instance.signal_grid:
        for cell in row:
            if 'signal' in cell and 'lead' in cell:
                sig = np.asarray(cell['signal'], dtype=float)
                signals.append(sig)
                lead_names.append(cell['lead'])
                max_length = max(max_length, len(sig))
                
    if not signals:
        raise ValueError("No valid lead signals were extracted from the ECG image.")
        
    # Pad shorter signals with NaN to make them uniform length
    padded = [np.pad(s, (0, max_length - len(s)), mode='constant', constant_values=np.nan) for s in signals]
    
    # Build DataFrame
    df = pd.DataFrame(np.vstack(padded).T, columns=lead_names)
    return df
