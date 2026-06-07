import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

# Import common configuration
try:
    from run_segmentation import TARGET_FS
except ImportError:
    TARGET_FS = 500  # Default fallback

def run_lead_importance_test(input_csv, output_dir):
    """
    Test the accuracy of identifying MI using each ECG lead individually.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segmented data
    print(f"Loading data from {input_csv}...")
    df_segmented = pd.read_csv(input_csv, index_col=['subject_id', 'timestamp'])
    
    if df_segmented.empty:
        print("Error: Input data is empty.")
        return
    
    # Define features/leads
    leads = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
    label_map = {'Post-Procedural MI': 0, 'Pre-Procedural MI': 1}
    
    results = []
    
    print(f"\nEvaluating {len(leads)} leads individually at {TARGET_FS} Hz...")
    
    for lead in leads:
        print(f"\n>>> Testing Lead {lead}...")
        
        # 1. Prepare data for this lead
        df_lead = df_segmented[[lead, 'class']].copy()
        
        # Normalize per heartbeat
        df_lead[lead] = df_lead.groupby(level='subject_id')[lead].transform(
            lambda x: x / x.abs().max() if x.abs().max() > 0 else x
        )
        
        # Reshape to (heartbeats, timesteps)
        beat_ids = df_lead.index.get_level_values('subject_id').unique()
        timesteps_per_beat = df_lead.groupby(level='subject_id').size().min()
        
        X_list = []
        y_list = []
        
        for beat_id in beat_ids:
            beat_data = df_lead.xs(beat_id, level='subject_id')
            if len(beat_data) >= timesteps_per_beat:
                X_list.append(beat_data[lead].values[:timesteps_per_beat])
                y_list.append(label_map[beat_data['class'].iloc[0]])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 2. Train and Evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        
        # Collect metrics
        metrics = {
            'Lead': lead,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1_Score': f1_score(y_test, y_pred, zero_division=0)
        }
        results.append(metrics)
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    
    # 3. Save and Visualize Results
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    results_df.to_csv(f'{output_dir}/lead_accuracies.csv', index=False)
    
    print("\n" + "="*40)
    print("LEAD IMPORTANCE TEST RESULTS")
    print("="*40)
    print(results_df.to_string(index=False))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['Lead'], results_df['Accuracy'], color='skyblue', edgecolor='navy')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Baseline (Random)')
    plt.title(f'MLP Accuracy per Single ECG Lead ({TARGET_FS} Hz)', fontsize=15, fontweight='bold')
    plt.xlabel('ECG Lead', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lead_accuracies_plot.png', dpi=300)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    #input_csv = 'data/segmented_heartbeats.csv'
    input_csv = 'data/ecg_matrix_dataset_segmented.csv'
    output_dir = 'output/lead_importance_test'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run segmentation first.")
    else:
        run_lead_importance_test(input_csv, output_dir)
