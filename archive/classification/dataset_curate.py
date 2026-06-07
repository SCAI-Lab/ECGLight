import os
import pandas as pd
from pathlib import Path

def process_ecg_data_wide_format(root_folder):
    """
    Process ECG data in wide format (one row per timestamp with all leads as columns)
    """
    lead_names = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
    
    all_data = []
    base_path = Path(root_folder)
    
    # Process both classes
    for class_name, class_folder in [("Pre-Procedural MI", "Pre"), 
                                    ("Post-Procedural MI", "Post")]:
        class_path = base_path / class_folder
        
        if class_path.exists():
            for csv_file in class_path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    subject_id = csv_file.stem.replace('_hr', '')
                    
                    # Add metadata columns
                    df['subject_id'] = subject_id
                    df['class'] = class_name
                    df['filename'] = csv_file.name
                    df['timestamp'] = range(len(df))
                    
                    all_data.append(df)
                    print(f"Processed {csv_file.name}")
                    print(class_name)
                    
                except Exception as e:
                    print(f"Error processing {csv_file.name}: {str(e)}")
    
    if all_data:
        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns to have metadata first
        metadata_cols = ['subject_id', 'timestamp', 'class', 'filename']
        signal_cols = [col for col in combined_df.columns if col not in metadata_cols]
        combined_df = combined_df[metadata_cols + signal_cols]
        
        return combined_df
    return None

# Usage for wide format:
if __name__ == "__main__":
    folder_path = "."
    
    if os.path.exists(folder_path):
        print("Processing ECG data in wide format...")
        wide_df = process_ecg_data_wide_format(folder_path)
        
        if wide_df is not None:
            wide_df.to_csv("combined_ecg_data_wide.csv", index=False)
            print("Wide format data saved to combined_ecg_data_wide.csv")
            print(f"Shape: {wide_df.shape}")
            print(f"Columns: {list(wide_df.columns)}")
            print("\nSample data:")
            print(wide_df.head())
    else:
        print(f"Folder '{folder_path}' not found.")