import os
import argparse
import pickle
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ───────────────────────────────────────────────────────────

MODELS_DIR = 'models'

FEATURES = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5',
            'III', 'aVF', 'V3', 'V6']

USE_CASES = {
    'mi_vs_normal_processed': {
        'name': 'MI vs Normal (Processed)',
        'input_csv': 'data/ptb_xl/ecg_dataset_processed.csv',
        'model_class': Arsenal,
        'model_name': 'Arsenal',
        'model_file': 'arsenal_processed_model.pkl',
        'positive_class': 'MYOCARDIAL_INFARCTION',
        'negative_class': 'NORMAL',
        'hyperparameters': {
            'num_kernels': 10000,
            'random_state': 42,
            'n_jobs': 1,
        },
    },
    'mi_vs_normal_segmented': {
        'name': 'MI vs Normal (Segmented)',
        'input_csv': 'data/ptb_xl/segmented_heartbeats.csv',
        'model_class': Arsenal,
        'model_name': 'Arsenal',
        'model_file': 'arsenal_segmented_model.pkl',
        'positive_class': 'MYOCARDIAL_INFARCTION',
        'negative_class': 'NORMAL',
        'hyperparameters': {
            'num_kernels': 20000,
            'random_state': 42,
            'n_jobs': 1,
        },
    },
    'omi_vs_nonomi': {
        'name': 'OMI vs non-OMI',
        'input_csv': 'data/ecg_matrix_omi_segmented_50_150_90.csv',
        'model_class': RocketClassifier,
        'model_name': 'Rocket',
        'model_file': 'rocket_model.pkl',
        'positive_class': 'OMI',
        'negative_class': 'non-OMI',
        'hyperparameters': {
            'num_kernels': 50000,
            'random_state': 42,
            'n_jobs': -1,
        },
    },
    'ecg_surgery': {
        'name': 'Pre vs Post Procedural MI (Surgery)',
        'input_csv': 'data/ecg_surgery_segmented_50_150_70.csv',
        'model_class': InceptionTimeClassifier,
        'model_name': 'InceptionTime',
        'model_file': 'inceptiontime_model.pkl',
        'positive_class': 'pre-procedural MI',
        'negative_class': 'post-procedural MI',
        'hyperparameters': {
            'n_epochs': 200,
            'batch_size': 64,
            'random_state': 42,
            'verbose': True,
        },
    },
}

# ─── Data Preprocessing ──────────────────────────────────────────────────────

def prepare_data(df):
    """Prepare data for benchmarking in sktime compatible format (numpy3D)"""
    features = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
    
    def safe_normalize(x):
        max_val = x.abs().max()
        # Prevent division by zero which yields NaNs
        return x / max_val if max_val > 0 else x

    print(" Normalizing features...")
    # Normalize features within each subject
    df_X = df[features].copy()
    for col in tqdm(df_X.columns, desc="Normalizing features"):
        df_X[col] = df_X.groupby(level='subject_id')[col].transform(safe_normalize)
    
    # Get unique subjects
    subject_ids = df_X.index.get_level_values('subject_id').unique()
    
    # Check timesteps per subject
    timesteps_per_subject = df_X.groupby(level='subject_id').size()
    print(f" Timesteps per subject stats:")
    print(f"   Min: {timesteps_per_subject.min()}, Max: {timesteps_per_subject.max()}, Mean: {timesteps_per_subject.mean():.1f}")
    
    # Use minimum timesteps to ensure consistent shape
    n_timesteps = timesteps_per_subject.min()
    n_features = len(features)
    
    print(f" Using {n_timesteps} timesteps per subject (minimum)")
    
    # Manually reshape to numpy3D format: (n_instances, n_features, n_timesteps)
    print(" Reshaping data to numpy3D format...")
    X_list = []
    y_list = []
    valid_subject_ids = []
    
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        try:
            subject_data = df_X.xs(subject_id, level='subject_id')
            
            # Ensure we have enough timesteps
            if len(subject_data) >= n_timesteps:
                # Take first n_timesteps and transpose to (n_features, n_timesteps)
                subject_array = subject_data.iloc[:n_timesteps].values.T
                
                # Check shape
                if subject_array.shape == (n_features, n_timesteps):
                    X_list.append(subject_array)
                    
                    # Get label for this subject
                    subject_label = df.loc[subject_id].iloc[0]['class']
                    y_list.append(subject_label)
                    valid_subject_ids.append(subject_id)
            else:
                continue
                
        except Exception as e:
            continue
    
    # Convert to numpy arrays
    X = np.array(X_list)  # Shape: (n_instances, n_features, n_timesteps)
    y = pd.Series(y_list, index=valid_subject_ids)
    
    print(f" Final data shape: {X.shape}")
    print(f" Class distribution:")
    print(y.value_counts())
    
    # Split into train/test
    print(" Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # --- SMOTE-TOMEK INJECTION ---
    print(" Executing SMOTE-Tomek to isolate and synthetically balance training geometries...")
    try:
        # Flatten 3D arrays to 2D for SMOTETomek: (N, features, timesteps) -> (N, features * timesteps)
        n_train, n_feat, n_time = X_train.shape
        X_train_flat = X_train.reshape(n_train, n_feat * n_time)
        
        # Balance classes
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled_flat, y_resampled = smote_tomek.fit_resample(X_train_flat, y_train)
        
        # Deflate back out to precise 3D format for Sktime models
        n_new = X_resampled_flat.shape[0]
        X_train = X_resampled_flat.reshape(n_new, n_feat, n_time)
        y_train = y_resampled
        
        print(f" -> SMOTE Success! Training geometries augmented from {n_train} to {n_new} samples.")
    except Exception as e:
        print(f" [!] SMOTETomek failed (are you missing imbalanced-learn?): {e}")
    # -----------------------------
    
    print(f" Final Training set: {X_train.shape}")
    print(f" Final Test set: {X_test.shape}")
    print(f" y_train balanced distribution: {y_train.value_counts().to_dict()}")
    print(f" y_test strictly isolated distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


# ─── Training Logic ──────────────────────────────────────────────────────────


def train_and_save(use_case_key, config):
    """
    Train a single model for the given use case, evaluate it, and persist
    both the serialised model and a metadata JSON to disk.

    Returns True on success, False on failure.
    """

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {config['name']}")
    print(f"  Model:    {config['model_name']}")
    print(f"{'=' * 60}")

    input_csv = config['input_csv']
    if not os.path.exists(input_csv):
        print(f"[ERROR] Input file not found: {input_csv}")
        return False

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data from {input_csv} ...")
    df = pd.read_csv(input_csv, index_col=['subject_id', 'timestamp'])
    
    # Map labels to Pre/Post Procedural MI if it's the surgery dataset
    if use_case_key == 'ecg_surgery':
        print("  Mapping class labels to 'pre-procedural MI' and 'post-procedural MI' ...")
        df['class'] = df['class'].replace({
            'Pre-Surgery MI': 'pre-procedural MI',
            'Post-Surgery MI': 'post-procedural MI'
        })
        
    print(f"  DataFrame shape: {df.shape}")

    subject_class = df.groupby(level='subject_id').first()['class']
    print(f"  Class distribution (subjects):\n{subject_class.value_counts().to_string()}")

    # ── Prepare data (normalize → reshape → split → SMOTE-Tomek) ──────────
    print("\nPreparing data (normalize, reshape, split, SMOTE-Tomek) ...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    n_timesteps = X_train.shape[2]
    n_features = X_train.shape[1]
    print(f"  n_timesteps = {n_timesteps}")
    print(f"  n_features  = {n_features}")
    print(f"  X_train shape: {X_train.shape}  |  X_test shape: {X_test.shape}")

    # ── Initialise & train model ──────────────────────────────────────────
    model = config['model_class'](**config['hyperparameters'])

    params_str = ""
    if 'num_kernels' in config['hyperparameters']:
        params_str = f" (num_kernels={config['hyperparameters']['num_kernels']})"
    elif 'n_epochs' in config['hyperparameters']:
        params_str = f" (n_epochs={config['hyperparameters']['n_epochs']})"
        if 'batch_size' in config['hyperparameters']:
            params_str += f" (batch_size={config['hyperparameters']['batch_size']})"

    print(f"\nTraining {config['model_name']}{params_str} ...")
    train_start = time.time()
    
    from joblib import parallel_backend
    with parallel_backend('threading'):
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        print(f"  Training completed in {train_time:.1f}s")

        # ── Evaluate on test set ──────────────────────────────────────────────
        print("\nEvaluating on held-out test set ...")
        y_pred = model.predict(X_test)

    positive = config['positive_class']
    y_test_num = (y_test == positive).astype(int)
    y_pred_num = (y_pred == positive).astype(int)

    accuracy    = accuracy_score(y_test_num, y_pred_num)
    precision   = precision_score(y_test_num, y_pred_num, zero_division=0)
    recall      = recall_score(y_test_num, y_pred_num, zero_division=0)
    f1          = f1_score(y_test_num, y_pred_num, zero_division=0)

    cm = confusion_matrix(y_test_num, y_pred_num)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.flatten()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = specificity = 0.0

    test_metrics = {
        'accuracy':         float(accuracy),
        'precision':        float(precision),
        'recall':           float(recall),
        'f1':               float(f1),
        'sensitivity':      float(sensitivity),
        'specificity':      float(specificity),
        'confusion_matrix': cm.tolist(),
    }

    print(f"\n{'-' * 40}")
    print(f"  TEST-SET RESULTS")
    print(f"{'-' * 40}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix:\n    {cm}")

    # ── Save model (pickle) ──────────────────────────────────────────────
    output_dir = os.path.join(MODELS_DIR, use_case_key)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, config['model_file'])
    print(f"\nSaving model to {model_path} ...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model saved  ({model_size_mb:.1f} MB)")

    # ── Save metadata (JSON) ─────────────────────────────────────────────
    metadata = {
        'model_name':       config['model_name'],
        'model_file':       config['model_file'],
        'use_case':         config['name'],
        'positive_class':   config['positive_class'],
        'negative_class':   config['negative_class'],
        'class_labels':     [config['negative_class'], config['positive_class']],
        'features':         FEATURES,
        'n_features':       int(n_features),
        'n_timesteps':      int(n_timesteps),
        'hyperparameters':  config['hyperparameters'],
        'training_data':    input_csv,
        'training_date':    time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': round(train_time, 2),
        'preprocessing': {
            'normalization': 'per_subject_max_abs',
            'smote_tomek':   True,
            'test_size':     0.25,
            'random_state':  42,
        },
        'training_samples': int(X_train.shape[0]),
        'test_samples':     int(X_test.shape[0]),
        'test_metrics':     test_metrics,
    }

    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")

    print(f"\n[OK] {config['name']} - done.")
    return True


# ─── Entry Point ──────────────────────────────────────────────────────────────

def show_saved_metrics():
    """
    Search the MODELS_DIR directory for any saved model metadata JSON files
    and print a beautiful summary of their performance metrics.
    """
    print('=' * 60)
    print('  TRAINED ECG MODEL METRICS SUMMARY')
    print('=' * 60)
    
    found = False
    if os.path.exists(MODELS_DIR):
        for item in sorted(os.listdir(MODELS_DIR)):
            meta_path = os.path.join(MODELS_DIR, item, 'model_metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    found = True
                    print(f"\nUse Case:      {meta.get('use_case', 'Unknown')}")
                    print(f"Model Class:   {meta.get('model_name', 'Unknown')}")
                    print(f"Model File:    {os.path.join(MODELS_DIR, item, meta.get('model_file', ''))}")
                    print(f"Trained On:    {meta.get('training_data', 'Unknown')}")
                    print(f"Training Date: {meta.get('training_date', 'Unknown')}")
                    if 'training_time_seconds' in meta:
                        print(f"Train Time:    {meta['training_time_seconds']:.1f}s")
                    
                    metrics = meta.get('test_metrics', {})
                    print("Test Metrics:")
                    print(f"  - Accuracy:    {metrics.get('accuracy', 0.0):.4f}")
                    print(f"  - Precision:   {metrics.get('precision', 0.0):.4f}")
                    print(f"  - Recall:      {metrics.get('recall', 0.0):.4f}")
                    print(f"  - F1 Score:    {metrics.get('f1', 0.0):.4f}")
                    print(f"  - Sensitivity: {metrics.get('sensitivity', 0.0):.4f}")
                    print(f"  - Specificity: {metrics.get('specificity', 0.0):.4f}")
                    
                    cm = metrics.get('confusion_matrix', None)
                    if cm:
                        print(f"  - Confusion Matrix:\n      {np.array(cm)}")
                    print("-" * 60)
                except Exception as e:
                    print(f"[Error loading metadata from {meta_path}]: {e}")
                    
    if not found:
        print(f"\nNo saved model metadata files found under '{MODELS_DIR}/'.")
        print("Please train models first to generate performance metrics.")
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and save the best ECG classification models')
    parser.add_argument(
        '--use-case', type=str,
        choices=['mi_vs_normal_processed', 'mi_vs_normal_segmented', 'omi_vs_nonomi', 'ecg_surgery', 'all'],
        default='all',
        help='Which use case to train (default: all)')
    parser.add_argument(
        '--show-metrics', action='store_true',
        help='Display performance metrics of already trained/saved models and exit')
    args = parser.parse_args()

    if args.show_metrics:
        show_saved_metrics()
    else:
        cases = list(USE_CASES.keys()) if args.use_case == 'all' else [args.use_case]

        print('=' * 60)
        print('  ECG MODEL TRAINING & SERIALISATION')
        print('=' * 60)

        for key in cases:
            success = train_and_save(key, USE_CASES[key])
            if not success:
                print(f"\n[FAIL] Could not train {USE_CASES[key]['name']}")

        print(f"\n{'=' * 60}")
        print(f"  ALL TRAINING COMPLETE")
        print(f"  Models saved to: {MODELS_DIR}/")
        print(f"{'=' * 60}")
