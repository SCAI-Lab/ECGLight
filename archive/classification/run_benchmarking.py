import argparse
import os
from utils.classifier_benchmarking import perform_benchmark

def run_benchmarking(input_file, output_folder=None):
    """
    Run benchmarking on the specified input file.
    
    Args:
        input_file: Path to input CSV
        output_folder: Directory to save results. If None, derives from input filename.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Determine output folder if not specified
    if output_folder is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_folder = f'output/benchmark_results/{base_name}'

    print(f"\n{'='*60}")
    print(f"STARTING BENCHMARKING")
    print(f"Input:  {input_file}")
    print(f"Output: {output_folder}")
    print(f"{'='*60}")
    
    try:
        perform_benchmark(input_file, output_dir=output_folder)
        print(f"\nBenchmarking complete for {input_file}")
        print(f"Results saved to: {output_folder}")
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ECG Classifier Benchmarking")
    parser.add_argument('--input', type=str, help='Path to input CSV file (optional)')
    parser.add_argument('--all', action='store_true', help='Run on both standard datasets (segmented and full)')
    
    args = parser.parse_args()
    
    # Default files
    segmented_file = 'data/segmented_heartbeats.csv'
    processed_file = 'data/ecg_dataset_processed.csv'
    
    # Define explicit output paths for standard runs
    output_segmented = 'output/benchmark_results/segmented_heartbeats'
    output_processed = 'output/benchmark_results/ecg_dataset_processed'
    
    if args.input:
        # Run on user specified file (folder derived from filename)
        run_benchmarking(args.input)
    elif args.all:
        # Run on both with specific folders
        print("Running comprehensive benchmarking on ALL datasets...")
        run_benchmarking(segmented_file, output_folder=output_segmented)
        run_benchmarking(processed_file, output_folder=output_processed)
    else:
        # Default run
        print("No input specified. Defaulting to processed ECG data.")
        print(f"Usage: python run_benchmarking.py --input <path> OR --all")
        run_benchmarking(processed_file, output_folder=output_processed)
