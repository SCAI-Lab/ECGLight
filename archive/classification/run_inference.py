"""
Run inference on new ECG data using a previously saved model.

Loads the serialised model and its metadata, applies identical preprocessing
(per-subject normalisation, reshape to numpy3D), and outputs per-subject
predictions with confidence scores.

Usage:
    # MI vs Normal
    python run_inference.py --model mi_vs_normal --input data/new_ecg.csv

    # OMI vs non-OMI
    python run_inference.py --model omi_vs_nonomi --input data/new_ecg.csv

    # Custom output path
    python run_inference.py --model mi_vs_normal --input data/new_ecg.csv --output results/preds.csv
"""

import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/classifier_models'))

# ─── Model Loading ───────────────────────────────────────────────────────────


def load_model(use_case):
    """
    Load a saved model and its metadata from the models directory.

    Args:
        use_case: One of 'mi_vs_normal' or 'omi_vs_nonomi'.

    Returns:
        (model, metadata) tuple.
    """
    model_dir = os.path.join(MODELS_DIR, use_case)

    # Metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}\n"
            f"Have you run train_and_save_models.py first?")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Model pickle
    model_path = os.path.join(model_dir, metadata['model_file'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Have you run train_and_save_models.py first?")

    print(f"Loading {metadata['model_name']} model for {metadata['use_case']} ...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"  Model loaded successfully.")
    print(f"  Trained on:      {metadata['training_data']}")
    print(f"  Training date:   {metadata['training_date']}")
    print(f"  Expected input:  {metadata['n_features']} leads x "
          f"{metadata['n_timesteps']} timesteps")
    print(f"  Performance metrics on held-out test set:")
    metrics = metadata.get('test_metrics', {})
    print(f"    - Accuracy:    {metrics.get('accuracy', 0.0):.4f}")
    print(f"    - Precision:   {metrics.get('precision', 0.0):.4f}")
    print(f"    - Recall:      {metrics.get('recall', 0.0):.4f}")
    print(f"    - F1 Score:    {metrics.get('f1', 0.0):.4f}")
    print(f"    - Sensitivity: {metrics.get('sensitivity', 0.0):.4f}")
    print(f"    - Specificity: {metrics.get('specificity', 0.0):.4f}")

    return model, metadata


# ─── Preprocessing ───────────────────────────────────────────────────────────


def preprocess_data(input_csv, metadata):
    """
    Preprocess new ECG data for inference using the same pipeline as training.

    Steps (mirrors prepare_data() in classifier_benchmarking.py):
      1. Per-subject normalisation (divide each lead by its max absolute value)
      2. Reshape to numpy3D  (n_subjects, n_features, n_timesteps)
      3. Truncate or zero-pad each subject to the training n_timesteps

    Args:
        input_csv:  Path to the input CSV.  Must contain columns
                    'subject_id', 'timestamp', and the 12 ECG lead columns.
        metadata:   The model_metadata dict loaded from JSON.

    Returns:
        (X, valid_subject_ids)  where X has shape
        (n_subjects, n_features, n_timesteps).
    """
    features    = metadata['features']
    n_timesteps = metadata['n_timesteps']
    n_features  = metadata['n_features']

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading input data from {input_csv} ...")
    df = pd.read_csv(input_csv, index_col=['subject_id', 'timestamp'])

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required lead columns in input: {missing}")

    # ── Normalise ─────────────────────────────────────────────────────────
    def safe_normalize(x):
        max_val = x.abs().max()
        return x / max_val if max_val > 0 else x

    print("Normalising features per subject ...")
    df_X = df[features].copy()
    for col in tqdm(df_X.columns, desc="Normalising"):
        df_X[col] = df_X.groupby(level='subject_id')[col].transform(safe_normalize)

    # ── Reshape ───────────────────────────────────────────────────────────
    subject_ids = df_X.index.get_level_values('subject_id').unique()
    timesteps_per_subject = df_X.groupby(level='subject_id').size()

    print(f"  Subjects found:          {len(subject_ids)}")
    print(f"  Timesteps per subject — "
          f"min: {timesteps_per_subject.min()}, "
          f"max: {timesteps_per_subject.max()}")
    print(f"  Target timesteps (train): {n_timesteps}")

    X_list = []
    valid_subject_ids = []
    skipped = 0

    for subject_id in tqdm(subject_ids, desc="Reshaping subjects"):
        try:
            subject_data = df_X.xs(subject_id, level='subject_id')
            actual_len = len(subject_data)

            if actual_len >= n_timesteps:
                # Truncate to training length
                arr = subject_data.iloc[:n_timesteps].values.T
            else:
                # Zero-pad short subjects
                arr = np.zeros((n_features, n_timesteps))
                arr[:, :actual_len] = subject_data.values.T
                print(f"  [warn] Subject {subject_id}: {actual_len} timesteps "
                      f"→ padded to {n_timesteps}")

            if arr.shape == (n_features, n_timesteps):
                X_list.append(arr)
                valid_subject_ids.append(subject_id)
            else:
                skipped += 1
        except Exception as e:
            print(f"  [warn] Skipped subject {subject_id}: {e}")
            skipped += 1

    if not X_list:
        raise ValueError("No valid subjects found in the input data!")

    X = np.array(X_list)
    print(f"\n  Preprocessed shape: {X.shape}  "
          f"({len(valid_subject_ids)} subjects, {skipped} skipped)")

    return X, valid_subject_ids


# ─── Inference ───────────────────────────────────────────────────────────────


def run_predictions(model, X, metadata, subject_ids):
    """
    Run model predictions and package them into a DataFrame.

    Returns:
        pd.DataFrame with columns:
            subject_id, predicted_class, confidence,
            probability_<class0>, probability_<class1>
    """
    print(f"\nRunning inference on {len(subject_ids)} subjects ...")

    # Predict labels and probabilities in threading backend to bypass Numba read-only array issue
    from joblib import parallel_backend
    with parallel_backend('threading', n_jobs=-1):
        y_pred = model.predict(X)

        # Predict probabilities (Arsenal & Rocket both support this)
        has_proba = False
        y_proba = None
        try:
            y_proba = model.predict_proba(X)
            has_proba = True
        except (AttributeError, NotImplementedError):
            pass

    # ── Build output DataFrame ────────────────────────────────────────────
    results = {
        'subject_id':      subject_ids,
        'predicted_class':  y_pred,
    }

    if has_proba:
        class_labels = list(model.classes_)
        for i, cls in enumerate(class_labels):
            results[f'probability_{cls}'] = y_proba[:, i]

        # Confidence = probability of the predicted class
        confidence = []
        for j in range(len(y_pred)):
            pred_idx = class_labels.index(y_pred[j])
            confidence.append(float(y_proba[j, pred_idx]))
        results['confidence'] = confidence

    return pd.DataFrame(results)


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Run ECG classification inference with a saved model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --model mi_vs_normal_processed --input data/ptb_xl/ecg_dataset_processed.csv
  python run_inference.py --model mi_vs_normal_segmented --input data/ptb_xl/segmented_heartbeats.csv
  python run_inference.py --model omi_vs_nonomi          --input data/ecg_matrix_omi_segmented_50_150_90.csv
  python run_inference.py --model ecg_surgery            --input data/ecg_surgery_segmented_50_150_70.csv
""")

    parser.add_argument('--model', type=str, required=True,
                        choices=['mi_vs_normal_processed', 'mi_vs_normal_segmented', 'omi_vs_nonomi', 'ecg_surgery'],
                        help='Which saved model to use')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV (subject_id, timestamp, 12 leads)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions CSV '
                             '(default: results/<model>_predictions.csv)')

    args = parser.parse_args()

    # Default output
    if args.output is None:
        os.makedirs('results', exist_ok=True)
        args.output = f'results/{args.model}_predictions.csv'

    print('=' * 60)
    print('  ECG CLASSIFICATION - INFERENCE')
    print('=' * 60)

    # ── Load model ────────────────────────────────────────────────────────
    model, metadata = load_model(args.model)

    # ── Preprocess ────────────────────────────────────────────────────────
    X, subject_ids = preprocess_data(args.input, metadata)

    # ── Predict ───────────────────────────────────────────────────────────
    results_df = run_predictions(model, X, metadata, subject_ids)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  INFERENCE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Model:          {metadata['model_name']} ({metadata['use_case']})")
    print(f"  Total subjects: {len(results_df)}")
    print(f"\n  Predicted class distribution:")
    for cls, count in results_df['predicted_class'].value_counts().items():
        print(f"    {cls}: {count}")

    if 'confidence' in results_df.columns:
        conf = results_df['confidence']
        print(f"\n  Confidence statistics:")
        print(f"    Mean:   {conf.mean():.4f}")
        print(f"    Median: {conf.median():.4f}")
        print(f"    Min:    {conf.min():.4f}")
        print(f"    Max:    {conf.max():.4f}")

    print(f"\n  Results saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
