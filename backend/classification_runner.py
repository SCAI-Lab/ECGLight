# backend/classification_runner.py
"""
Backend adapter for ECG Classification.
Handles heartbeat segmentation, data preparation for sktime models, and model evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path to ensure digitization and other local modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom Exceptions for dependencies
class DependencyMissingError(Exception):
    """Raised when a required third-party library is not installed."""
    pass

class EmptyDatasetError(Exception):
    """Raised when the uploaded dataset is empty or invalid."""
    pass

# Helper to check dependencies
def check_classification_dependencies():
    """
    Checks if sktime is installed in the current environment.
    Raises DependencyMissingError if it is not.
    """
    missing = []
    try:
        import sktime
    except ImportError:
        missing.append("sktime")
        
    if missing:
        raise DependencyMissingError(
            f"The following libraries are required for classification but are missing in this environment:\n"
            f"{', '.join(missing)}\n\n"
            f"Please run: pip install sktime"
        )

# ==============================================================================
# HEARTBEAT SEGMENTATION
# ==============================================================================
# ==============================================================================
# LOCAL HEARTBEAT SEGMENTATION IMPLEMENTATION (copied from ecg_segmentation.py)
# ==============================================================================

def bandpass_filter(signal_data, lowcut=5, highcut=15, fs=500, order=2):
    """Bandpass filter for ECG signal"""
    from scipy.signal import butter, filtfilt
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal_data)
    except Exception:
        return signal_data


def find_r_peaks(ecg_signal, min_heart_rate=40, max_heart_rate=180, fs=500):
    """Find R-peaks in ECG signal"""
    from scipy.signal import find_peaks
    # Calculate expected distance between peaks based on heart rate
    min_distance = int((60 / max_heart_rate) * fs)  # samples
    max_distance = int((60 / min_heart_rate) * fs)  # samples
    
    # Find peaks
    peaks, properties = find_peaks(ecg_signal, 
                                  height=np.percentile(ecg_signal, 80),
                                  distance=min_distance,
                                  prominence=np.std(ecg_signal) * 0.3)
    
    # Filter peaks based on reasonable heart rate
    if len(peaks) == 0:
        return peaks
        
    valid_peaks = [peaks[0]]
    for i in range(1, len(peaks)):
        distance = peaks[i] - valid_peaks[-1]
        if min_distance <= distance <= max_distance:
            valid_peaks.append(peaks[i])
    
    return np.array(valid_peaks)


def downsample_beat(beat_data, features, downsampling_factor):
    """
    Downsample a single heartbeat by taking every nth sample
    """
    try:
        # Create a new DataFrame for downsampled data
        downsampled_data = []
        
        # Take every nth sample
        for i in range(0, len(beat_data), downsampling_factor):
            if i < len(beat_data):
                row = beat_data.iloc[i]
                downsampled_data.append(row)
        
        if len(downsampled_data) > 0:
            return pd.DataFrame(downsampled_data, columns=beat_data.columns)
        else:
            return None
    except Exception as e:
        print(f"Error in downsampling: {e}")
        return None


def segment_beats_around_r_peaks(subject_data, r_peaks, features, fs=500, target_fs=500, pre_r_ms=150, post_r_ms=250):
    """
    Segment heartbeats around R-peaks with downsampling
    """
    heartbeats = []
    total_samples = len(subject_data)
    
    # Calculate downsampling factor
    downsampling_factor = fs // target_fs
    if downsampling_factor < 1:
        downsampling_factor = 1
    
    # Convert milliseconds to samples at original and target frequencies
    pre_r_original = int((pre_r_ms / 1000) * fs)
    post_r_original = int((post_r_ms / 1000) * fs)
    
    for r_peak in r_peaks:
        start_idx = max(0, r_peak - pre_r_original)
        end_idx = min(total_samples, r_peak + post_r_original)
        
        # Ensure we have enough samples for a complete beat
        if (end_idx - start_idx) >= (pre_r_original + post_r_original) * 0.7:  # at least 70% of expected length
            beat_data_original = subject_data.iloc[start_idx:end_idx].copy()
            
            # Downsample the beat
            beat_data_downsampled = downsample_beat(beat_data_original, features, downsampling_factor)
            
            if beat_data_downsampled is not None:
                heartbeats.append((beat_data_downsampled, r_peak))
    
    return heartbeats


def local_segment_heartbeats(input_csv, output_csv=None, min_heart_rate=40, max_heart_rate=180, 
                             fs=500, target_fs=500):
    """
    Segment ECG data into individual heartbeats for each subject with downsampling.
    Self-contained implementation.
    """
    print("Loading ECG data for heartbeat segmentation...")
    df = pd.read_csv(input_csv, index_col=['subject_id', 'timestamp'])
    features = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
    
    # Fill NaN values in features using linear interpolation and fallback to 0.0
    for feat in features:
        if feat in df.columns:
            df[feat] = df[feat].interpolate(method='linear', limit_direction='both').fillna(0.0)
            
    # Calculate downsampling factor
    downsampling_factor = fs // target_fs
    print(f"Downsampling from {fs} Hz to {target_fs} Hz (factor: {downsampling_factor})")
    
    segmented_data = []
    
    # Get unique subjects
    subject_ids = df.index.get_level_values('subject_id').unique()
    
    for subject_id in subject_ids:
        # Get subject data
        subject_data = df.xs(subject_id, level='subject_id')
        subject_class = subject_data['class'].iloc[0]
        
        # Use Lead II for R-peak detection (commonly used)
        ecg_signal = subject_data['II'].values
        
        # Skip if signal is too short
        if len(ecg_signal) < fs * 2:  # Less than 2 seconds
            continue
            
        # Preprocess signal (apply bandpass filter for robust R-peak detection)
        try:
            ecg_filtered = bandpass_filter(ecg_signal, lowcut=5, highcut=15, fs=fs, order=2)
        except Exception:
            ecg_filtered = ecg_signal
        
        # Find R-peaks
        r_peaks = find_r_peaks(ecg_filtered, min_heart_rate=min_heart_rate, max_heart_rate=max_heart_rate, fs=fs)
        
        if len(r_peaks) < 2:
            # Fallback: if filtered failed or returned no peaks, try on unfiltered signal
            try:
                r_peaks = find_r_peaks(ecg_signal, min_heart_rate=min_heart_rate, max_heart_rate=max_heart_rate, fs=fs)
            except Exception:
                pass
            
        if len(r_peaks) < 2:
            print(f"  Warning: Only {len(r_peaks)} R-peaks found for subject {subject_id}")
            continue
        
        # Segment heartbeats around R-peaks
        heartbeats = segment_beats_around_r_peaks(subject_data, r_peaks, features, fs=fs, target_fs=target_fs)
        
        # Add segmented heartbeats to results
        for i, (beat_data, r_peak_pos) in enumerate(heartbeats):
            beat_id = f"{subject_id}_beat{i+1}"
            
            # Create new index with beat information
            for j, (timestamp, row) in enumerate(beat_data.iterrows()):
                beat_row = {
                    'subject_id': beat_id,
                    'original_subject': subject_id,
                    'beat_number': i + 1,
                    'r_peak_position': int(r_peak_pos),
                    'class': subject_class,
                    'timestamp': j
                }
                
                # Add all features
                for feature in features:
                    beat_row[feature] = float(row[feature])
                
                segmented_data.append(beat_row)
        
    # Create new DataFrame
    if not segmented_data:
        return pd.DataFrame()
        
    segmented_df = pd.DataFrame(segmented_data)
    
    # Set multi-index
    if not segmented_df.empty:
        segmented_df.set_index(['subject_id', 'timestamp'], inplace=True)
    
    # Save if output path provided
    if output_csv and not segmented_df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        segmented_df.to_csv(output_csv)
        print(f"Segmented data saved to: {output_csv}")
    
    print(f"\nSegmentation complete:")
    print(f"Total heartbeats: {segmented_df.index.get_level_values('subject_id').nunique() if not segmented_df.empty else 0}")
    
    return segmented_df


def segment_uploaded_csv(input_csv_path, output_csv_path, target_fs=500):
    """
    Segments a raw digitized dataset CSV into individual heartbeats.
    Uses the local self-contained heartbeat segmentation implementation.
    """
    # Read the first few lines to inspect columns
    df_raw = pd.read_csv(input_csv_path)
    if df_raw.empty:
        raise EmptyDatasetError("The uploaded CSV file is empty.")
        
    # Standardize input formatting for the segmentation module.
    # The segmentation module expects 'subject_id', 'timestamp', 'class' columns.
    required_cols = ['subject_id', 'timestamp', 'class']
    needs_formatting = not all(col in df_raw.columns for col in required_cols)
    
    formatted_csv_path = input_csv_path
    if needs_formatting:
        print("Formatting raw CSV columns to match expected segmentation schema...")
        # Check if we have lead columns
        lead_cols = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
        existing_leads = [col for col in lead_cols if col in df_raw.columns]
        if not existing_leads:
            raise ValueError(
                f"The uploaded CSV does not contain standard ECG lead columns.\n"
                f"Expected some of: {lead_cols}"
            )
            
        # Format df_raw
        df_formatted = df_raw.copy()
        if 'subject_id' not in df_formatted.columns:
            df_formatted['subject_id'] = 'subject_1'
        if 'class' not in df_formatted.columns:
            # Default placeholder label
            df_formatted['class'] = 'Pre-Surgery MI'
        if 'timestamp' not in df_formatted.columns:
            df_formatted['timestamp'] = range(len(df_formatted))
            
        # Ensure subject_id and class are strings
        df_formatted['subject_id'] = df_formatted['subject_id'].astype(str)
        df_formatted['class'] = df_formatted['class'].astype(str)
        
        # Save to temporary path for segmentation input
        formatted_csv_path = input_csv_path + "_formatted.csv"
        df_formatted.to_csv(formatted_csv_path, index=False)
        
    # Run the local segmentation module
    try:
        segmented_df = local_segment_heartbeats(
            formatted_csv_path,
            output_csv=output_csv_path,
            fs=target_fs,
            target_fs=target_fs
        )
    finally:
        # Clean up temporary formatted file if created
        if needs_formatting and os.path.exists(formatted_csv_path):
            try:
                os.remove(formatted_csv_path)
            except Exception:
                pass
                
    if segmented_df is None or segmented_df.empty:
        raise ValueError(
            "Heartbeat segmentation returned no beats. This can happen if the signal is too short or R-peaks were not detected."
        )
        
    return segmented_df


# 


# ==============================================================================
# PRE-TRAINED MODEL LOADING & INFERENCE SUPPORT
# ==============================================================================

def load_pretrained_model_and_metadata(model_key):
    """
    Loads a pre-trained model and its metadata from models/classifier_models/{model_key}.
    """
    import json
    import pickle

    model_dir = os.path.join("models", "classifier_models", model_key)
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Pre-trained model metadata not found: {metadata_path}\n"
            f"Please verify models/classifier_models/ directory contents."
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    model_path = os.path.join(model_dir, metadata['model_file'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pre-trained model pickle not found: {model_path}\n"
            f"Please verify models/classifier_models/ directory contents."
        )

    print(f"Loading {metadata['model_name']} pre-trained model for {metadata['use_case']}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Restore missing classes_ on unpickled scikit-learn models due to version inconsistencies
    def _fix_unpickled_classes(obj, visited=None):
        if obj is None:
            return
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        if hasattr(obj, '_label_binarizer') and hasattr(obj._label_binarizer, 'classes_'):
            if not hasattr(obj, 'classes_') or obj.classes_ is None:
                try:
                    obj.classes_ = obj._label_binarizer.classes_
                except Exception:
                    pass
                    
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _fix_unpickled_classes(item, visited)
        elif isinstance(obj, dict):
            for val in obj.values():
                _fix_unpickled_classes(val, visited)
        else:
            for attr in ['estimator', 'estimator_', 'classifier', 'classifier_', 
                         'estimators', 'estimators_', 'transformers', 'transformers_', 
                         'steps', 'steps_']:
                if hasattr(obj, attr):
                    try:
                        val = getattr(obj, attr)
                        _fix_unpickled_classes(val, visited)
                    except Exception:
                        pass

    _fix_unpickled_classes(model)

    return model, metadata


def preprocess_dataframe_for_inference(df, metadata):
    """
    Preprocesses raw or segmented DataFrame into numpy3D array for the pre-trained model.
    Reshapes each subject's signal to match the exact target length (n_timesteps),
    applies max absolute voltage normalization per subject, and returns the 3D numpy array.
    """
    features = metadata['features']
    n_timesteps = metadata['n_timesteps']
    n_features = metadata['n_features']

    # Standardize input formatting
    # If index is multi-index with subject_id, or if subject_id/timestamp are columns
    df_reset = df.copy()
    if 'subject_id' in df_reset.index.names:
        df_reset = df_reset.reset_index()
    elif 'subject_id' not in df_reset.columns:
        df_reset['subject_id'] = 'subject_1'

    if 'timestamp' in df_reset.index.names:
        df_reset = df_reset.reset_index()
    elif 'timestamp' not in df_reset.columns:
        df_reset['timestamp'] = df_reset.groupby('subject_id').cumcount()

    df_reset = df_reset.set_index(['subject_id', 'timestamp'])

    # Verify required features exist
    missing = [f for f in features if f not in df_reset.columns]
    if missing:
        # Fallback to standard channels if names differ slightly (e.g. casing)
        df_reset.columns = [c.strip() for c in df_reset.columns]
        missing = [f for f in features if f not in df_reset.columns]
        if missing:
            raise ValueError(f"Missing required lead columns in input: {missing}")

    # Normalize features per subject
    def safe_normalize(x):
        max_val = x.abs().max()
        return x / max_val if max_val > 0 else x

    df_X = df_reset[features].copy()
    for col in df_X.columns:
        df_X[col] = df_X.groupby(level='subject_id')[col].transform(safe_normalize)

    # Reshape to 3D array: (n_subjects, n_features, n_timesteps)
    subject_ids = df_X.index.get_level_values('subject_id').unique()
    X_list = []
    valid_subject_ids = []
    y_true_str = []

    for subject_id in subject_ids:
        try:
            subject_data = df_X.xs(subject_id, level='subject_id')
            actual_len = len(subject_data)

            if actual_len >= n_timesteps:
                # Truncate
                arr = subject_data.iloc[:n_timesteps].values.T
            else:
                # Zero-pad
                arr = np.zeros((n_features, n_timesteps))
                arr[:, :actual_len] = subject_data.values.T

            if arr.shape == (n_features, n_timesteps):
                X_list.append(arr)
                valid_subject_ids.append(subject_id)
                
                # Capture ground truth class label if it exists in the source DataFrame
                if 'class' in df_reset.columns:
                    lbl = df_reset.loc[subject_id].iloc[0]['class']
                    y_true_str.append(str(lbl))
        except Exception:
            continue

    if not X_list:
        raise ValueError(
            f"No valid subjects/heartbeats found after preprocessing. "
            f"Ensure data contains at least one subject with some timesteps."
        )

    return np.array(X_list), valid_subject_ids, y_true_str


def run_pretrained_inference(model, X, metadata, subject_ids):
    """
    Runs pre-trained model predictions and returns a results dictionary.
    """
    from joblib import parallel_backend

    with parallel_backend('threading', n_jobs=-1):
        y_pred = model.predict(X)

        has_proba = False
        y_proba = None
        try:
            y_proba = model.predict_proba(X)
            has_proba = True
        except (AttributeError, NotImplementedError):
            pass

    results = {
        'subject_id': subject_ids,
        'predicted_class': y_pred.tolist(),
    }

    if has_proba and y_proba is not None:
        class_labels = list(model.classes_)
        results['class_labels'] = class_labels
        results['probabilities'] = y_proba.tolist()

        # Calculate confidence as the probability of the predicted class
        confidence = []
        for j in range(len(y_pred)):
            try:
                pred_idx = class_labels.index(y_pred[j])
                confidence.append(float(y_proba[j, pred_idx]))
            except ValueError:
                confidence.append(1.0)
        results['confidence'] = confidence

    return results


def calculate_evaluation_metrics(y_true_str, y_pred_str, class_labels, positive_class):
    """
    Calculates accuracy, precision, recall, f1, sensitivity, specificity, and confusion matrix.
    Maps ground-truth and prediction string labels to binary (0/1).
    """
    # Map strings to binary values (0/1) based on positive class
    pos_str = str(positive_class).strip().lower()
    
    # We do a lenient comparison (check substring or exact match)
    y_true_num = [1 if pos_str in str(y).strip().lower() else 0 for y in y_true_str]
    y_pred_num = [1 if pos_str in str(y).strip().lower() else 0 for y in y_pred_str]

    # Handle cases where all targets are of one class
    if len(np.unique(y_true_num)) <= 1 and len(np.unique(y_pred_num)) <= 1:
        # Avoid standard metrics crashing
        accuracy = float(accuracy_score(y_true_num, y_pred_num))
        return {
            "Accuracy": accuracy,
            "Precision": 1.0 if accuracy == 1.0 else 0.0,
            "Recall": 1.0 if accuracy == 1.0 else 0.0,
            "F1": 1.0 if accuracy == 1.0 else 0.0,
            "Sensitivity": 1.0 if accuracy == 1.0 else 0.0,
            "Specificity": 1.0 if accuracy == 1.0 else 0.0,
            "Confusion Matrix": confusion_matrix(y_true_num, y_pred_num, labels=[0, 1]).tolist(),
            "TN": int(accuracy * len(y_true_num)) if y_true_num[0] == 0 else 0,
            "FP": 0,
            "FN": 0,
            "TP": int(accuracy * len(y_true_num)) if y_true_num[0] == 1 else 0
        }

    accuracy = float(accuracy_score(y_true_num, y_pred_num))
    precision = float(precision_score(y_true_num, y_pred_num, zero_division=0))
    recall = float(recall_score(y_true_num, y_pred_num, zero_division=0))
    f1 = float(f1_score(y_true_num, y_pred_num, zero_division=0))

    cm = confusion_matrix(y_true_num, y_pred_num, labels=[0, 1])
    tn, fp, fn, tp = cm.flatten()
    
    sensitivity = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
    specificity = float(tn / (tn + fp) if (tn + fp) > 0 else 0)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Confusion Matrix": cm.tolist(),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

