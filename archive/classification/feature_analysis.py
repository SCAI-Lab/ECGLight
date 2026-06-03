import os
import pandas as pd
import numpy as np
from utils.mlp_feature_downsample import (
    perform_comprehensive_heartbeat_analysis,
    perform_full_signal_analysis,
    DEFAULT_HEARTBEAT_OUTPUT_DIR,
    DEFAULT_TARGET_FS,
)

if __name__ == "__main__":
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Use the processed CSV from the previous step
    combined_input_csv = 'data/ecg_dataset_processed.csv'
    
    # Load and verify the data
    df_combined = pd.read_csv(combined_input_csv)
    print(f"Loaded combined dataset shape: {df_combined.shape}")
    print(f"Unique subjects: {df_combined['subject_id'].nunique()}")

    print("\n" + "="*50)
    print("RUNNING SEGMENTED HEARTBEAT ANALYSIS (500 Hz)")
    print("="*50)
    
    heartbeat_results = perform_comprehensive_heartbeat_analysis(
        input_csv=combined_input_csv,
        output_dir=DEFAULT_HEARTBEAT_OUTPUT_DIR,
        pca_components=50,
        shap_samples=100,
        shap_background_size=50,
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        target_fs=500,
    )
    
    print("\n" + "="*50)
    print("RUNNING FULL SIGNAL ANALYSIS (500 Hz)")
    print("="*50)
    
    full_signal_results = perform_full_signal_analysis(
        input_csv=combined_input_csv,
        output_dir='output/full_signal_analysis',
        pca_components=50,
        shap_samples=100,
        shap_background_size=50,
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        target_fs=500,
    )

    # Print Summary Results
    print("\n" + "#" * 60)
    print("FINAL SUMMARY RESULTS")
    print("#" * 60)
    
    if heartbeat_results is not None and 'metrics' in heartbeat_results:
        m = heartbeat_results['metrics']
        print("\n--- Segmented Heartbeat Analysis ---")
        print(f"  Accuracy:    {m['accuracy']:.4f}")
        print(f"  Precision:   {m['precision']:.4f}")
        print(f"  Recall:      {m['recall']:.4f}")
        print(f"  F1-score:    {m['f1']:.4f}")
        print(f"  Sensitivity: {m['sensitivity']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")
        print("=" * 50)
    '''
    ############################################## Supervised Classifier Benchmarking ##################################
    #print("\nPerforming classifier benchmarking...")
    benchmark_results, models = perform_benchmark(combined_input_csv)
    
    print("\nBenchmarking Results:")
    print(benchmark_results)

    # Save all results to a summary file
    results_summary = {
        #'kmeans_results': kmeans_results,
        'mlp_results': mlp_results,
        'benchmark_results': benchmark_results,
    }
    
    # Save summary to file
    summary_df = pd.DataFrame([
        #{'model': 'K-Means', 'details': str(kmeans_results)},
        #{'model': 'MLP', 'details': str(mlp_results)},
        {'model': 'Benchmark', 'details': str(benchmark_results)},
    ])
    
    summary_df.to_csv('output/analysis_summary.csv', index=False)
    print("\nAnalysis summary saved to 'output/analysis_summary.csv'")
    '''