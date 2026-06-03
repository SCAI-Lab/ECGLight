from utils.ecg_classification import perform_classification_analysis
import os
# Import the universally defined target frequency
from run_segmentation import TARGET_FS

if __name__ == "__main__":
    # Define paths
    input_csv = 'data/ecg_matrix_dataset_segmented.csv'
    output_dir = 'output/heartbeat_analysis_matrix'
    
    # Check if input exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        print("Please run run_segmentation.py first.")
        exit(1)
        
    print(f"Starting classification analysis...")
    print(f"Input: {input_csv}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Frequency: {TARGET_FS} Hz (imported from run_segmentation.py)")
    
    # Run analysis
    # Normalization happens inside this function
    results = perform_classification_analysis(
        segmented_csv=input_csv,
        output_dir=output_dir,
        target_fs=TARGET_FS
    )
    
    if results:
        print("\nAnalysis successful!")
        print(f"Results available in {output_dir}")
    else:
        print("\nAnalysis failed.")
