import pandas as pd
import os

if __name__ == "__main__":
    # ── CONFIGURATION ──
    # Input CSV file path
    input_csv = 'data/combined_ecg_matrix_data_segmented.csv'
    
    # Output CSV file path
    output_csv = 'data/combined_ecg_matrix_data_filtered.csv'

    # Option 1: Only include specific classes (set to None to use all)
    # e.g. ['Pre', 'Post', 'Index']
    include = None

    # Option 2: Exclude specific classes (set to None to skip)
    # e.g. ['Dimi']
    exclude = None

    # Option 3: Combine/merge classes under a new label (set to None to skip)
    # e.g. {'Surgery': ['Pre', 'Post'], 'Other': ['Dimi', 'Discharge']}
    combine = {'Post-Surgery': ['Discharge', 'Dimi', 'Post'], 'Pre-Surgery': ['Index', 'Pre']}

    print(f"Loading data from '{input_csv}'...")
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        exit(1)

    df = pd.read_csv(input_csv)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows.")
    
    if 'class' not in df.columns:
        print("Error: 'class' column not found in input CSV.")
        exit(1)

    print(f"\nOriginal classes: {sorted(df['class'].dropna().unique())}")
    for cls in sorted(df['class'].dropna().unique()):
        print(f"  {cls}: {len(df[df['class'] == cls])} rows")

    # ── 1. Combine/Merge Classes ──
    if combine:
        print("\nApplying class combinations...")
        for new_name, old_names in combine.items():
            df.loc[df['class'].isin(old_names), 'class'] = new_name
            print(f"  Combined {old_names} -> '{new_name}'")

    # ── 2. Include Classes ──
    if include:
        print(f"\nIncluding only: {include}")
        df = df[df['class'].isin(include)]

    # ── 3. Exclude Classes ──
    if exclude:
        print(f"\nExcluding: {exclude}")
        df = df[~df['class'].isin(exclude)]

    if len(df) == 0:
        print("\nERROR: No data remaining after class filtering!")
        exit(1)

    print(f"\nData processing complete. Remaining rows: {len(df)} ({(len(df)/initial_rows)*100:.1f}%)")
    
    final_classes = sorted(df['class'].dropna().unique())
    print(f"\nFinal classes hierarchy: {final_classes}")
    for cls in final_classes:
        print(f"  {cls}: {len(df[df['class'] == cls])} rows")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved filtered data to '{output_csv}'")
