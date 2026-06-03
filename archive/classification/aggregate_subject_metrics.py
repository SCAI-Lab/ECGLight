import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics import accuracy_score, precision_score, f1_score

def get_base_subject_id(subject_id_with_beat):
    """
    Extracts the base subject ID by removing suffixes like _beatN,  beatN, -index_beatN, etc.
    """
    # Regex to catch variations like _beat1,  beat1, -index_beat1, INDEX_beat1, etc.
    # It looks for 'beat' preceded by common delimiters and captures everything before it.
    match = re.search(r'^(.*?)(?:[_\s-](?:index|INDEX|post|POST|checkpoint|POST PCI|POST PCI \d|POST \d)*)?(?:[_\s-]beat\d+)$', str(subject_id_with_beat))
    if match:
        return match.group(1).strip()
    
    # Fallback: just split by 'beat' if regex is too specific
    if 'beat' in str(subject_id_with_beat):
        return str(subject_id_with_beat).split('beat')[0].rstrip('_ -').strip()
    
    return str(subject_id_with_beat).strip()

def calculate_metrics(df_subject):
    """
    Calculates metrics for subject-level predictions.
    """
    # Ground truth: since all beats for a subject have the same label, we just take the first
    y_true = df_subject['true_label_numeric'].values
    
    # Prediction based on averaged probability for Pre-Surgery MI
    # Assuming probability_Pre-Surgery MI >= 0.5 is Pre-Surgery (1)
    y_pred = (df_subject['avg_prob_pre_surgery'] >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return acc, prec, f1

def main():
    prediction_dir = r'output/benchmark_results/ecg_matrix_dataset_segmented/model_predictions'
    
    if not os.path.exists(prediction_dir):
        print(f"Directory not found: {prediction_dir}")
        return

    # Filter out the combined file if it exists
    files = [f for f in os.listdir(prediction_dir) if f.endswith('.csv') and f != 'all_models_predictions_combined.csv']
    
    results = []
    
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'F1 Score':<10}")
    print("-" * 60)

    for file in files:
        model_name = file.replace('_predictions.csv', '')
        file_path = os.path.join(prediction_dir, file)
        
        try:
            df = pd.read_csv(file_path)
            
            # 1. Extract base subject ID
            df['base_subject_id'] = df['subject_id'].apply(get_base_subject_id)
            
            # 2. Group by base subject ID
            # We average the probability and take the first true label (since it should be invariant for a subject)
            subject_group = df.groupby('base_subject_id').agg({
                'probability_Pre-Surgery MI': 'mean',
                'true_label_numeric': 'first'
            }).reset_index()
            
            subject_group.rename(columns={'probability_Pre-Surgery MI': 'avg_prob_pre_surgery'}, inplace=True)
            
            # 3. Calculate Metrics
            acc, prec, f1 = calculate_metrics(subject_group)
            
            results.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'F1 Score': f1
            })
            
            print(f"{model_name:<20} | {acc:<10.4f} | {prec:<10.4f} | {f1:<10.4f}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Save summary to CSV
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(prediction_dir, 'subject_level_metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
