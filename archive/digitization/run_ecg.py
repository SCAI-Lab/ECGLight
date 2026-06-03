# Force all BLAS / OpenMP thread pools to a single thread before any other
# import. This must happen before numpy, torch, or cv2 are imported, otherwise
# the libraries have already spawned their thread pools and the env vars have
# no effect. Without this, BLAS and PyTorch worker threads compete for CPU
# cores and can deadlock on multi-core servers.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import signal
import torch
import pandas as pd
from ultralytics import YOLO
from digitization import ECGImage

# -------------------------
# Configuration
# -------------------------

# Root of the organised input dataset. Each immediate subdirectory is treated
# as one hospital and must itself contain subfolders named after CATEGORIES.
ORGANIZED_DIR = "../ecg_files/ECG_organized_all"

# Root where output CSVs are written, mirroring the hospital/category structure
# of ORGANIZED_DIR. Created automatically if it does not exist.
OUTPUT_DIR    = "../ecg_files/ECG_digitized"

# Subfolders to process within each hospital directory.
# Any subfolder not listed here is silently skipped.
# Add or remove entries to match your dataset's naming convention.
CATEGORIES    = ["pre", "index", "post", "discharge", "dimi"]

# Per-image wall-clock time limit in seconds, enforced via SIGALRM.
# Images that exceed this budget are logged as timeouts and skipped.
# Increase for very large images or slow CPU-only machines.
# Note: SIGALRM is only available on Linux.
TIMEOUT       = 600

# -------------------------
# Torch / GPU settings
# -------------------------

if torch.cuda.is_available():
    # benchmark=True lets cuDNN auto-tune convolution algorithms at the first
    # forward pass. Speeds up subsequent passes when input sizes are fixed,
    # which is the case here (images are resized to a fixed height).
    torch.backends.cudnn.benchmark     = True

    # deterministic=False allows cuDNN to use non-deterministic (but faster)
    # algorithms. Set to True if exact reproducibility is required.
    torch.backends.cudnn.deterministic = False

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n",
          flush=True)
else:
    print("No GPU found — running on CPU.\n", flush=True)

# -------------------------
# Timeout (Linux only)
# -------------------------

class TimeoutException(Exception):
    """Raised by timeout_handler when SIGALRM fires."""
    pass

def timeout_handler(signum, frame):
    """Signal handler that converts SIGALRM into a TimeoutException.

    Registered below with signal.signal() so that any blocking call
    inside process_folder() can be interrupted after TIMEOUT seconds.
    """
    raise TimeoutException("Processing timed out")

# Register the handler for SIGALRM. signal.alarm(n) starts a countdown;
# if the process has not called signal.alarm(0) within n seconds, this
# handler fires and raises TimeoutException.
signal.signal(signal.SIGALRM, timeout_handler)

# -------------------------
# Load models once (on GPU if available)
# -------------------------

# All four models are loaded once before the processing loop so that the
# weights are only read from disk once, regardless of how many images are
# processed. Loading inside the loop would add several seconds of overhead
# per image.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models...", flush=True)

# Detection model for full lead bounding boxes (one box per lead region).
box_model       = YOLO("Submission code/3. Model training/runs/yolo11_full/weights/best.pt")

# Classification model for lead name labels (I, II, III, aVR, aVL, aVF, V1–V6).
lead_name_model = YOLO("Submission code/3. Model training/runs/yolo11_lead/weights/best.pt")

# Detection model for calibration (reference) pulse boxes.
pulse_model     = YOLO("Submission code/3. Model training/runs/yolo11_pulse/weights/best.pt")

box_model.to(DEVICE)
lead_name_model.to(DEVICE)
pulse_model.to(DEVICE)

# The segmentation model is kept as a file path rather than a loaded model
# object because it is consumed by patched_yolo_infer.MakeCropsDetectThem,
# which loads it internally on each call. This avoids a second copy of the
# weights living in VRAM alongside the three detection models above.
segmentation_model = "Submission code/3. Model training/runs/yolo11_patch/weights/best.pt"

print(f"Models loaded on {DEVICE.upper()}.\n", flush=True)

# -------------------------
# Process a single folder
# -------------------------

def process_folder(input_dir, output_dir):
    """Digitize all ECG images in input_dir and save CSVs to output_dir.

    Iterates over all .png / .jpg / .jpeg files in input_dir (sorted
    alphabetically). Each image is processed by ECGImage.run_full_pipeline()
    and the resulting signals are exported as a CSV file with the same base
    name as the image.

    Parameters
    ----------
    input_dir : str
        Directory containing the input ECG images.
    output_dir : str
        Directory where output CSV files are written (created if absent).

    Returns
    -------
    success : int
        Number of images successfully digitized.
    total : int
        Total number of images found in input_dir.
    """
    # Collect all supported image formats; sort for reproducible ordering.
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort()

    # Return early without creating the output directory if the folder is empty.
    if not files:
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    success = errors = timeouts = 0

    for i, file_path in enumerate(files):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"    [{i+1}/{len(files)}] {base_name}", flush=True)
        try:
            # Start the per-image countdown. If the block below takes longer
            # than TIMEOUT seconds, SIGALRM fires and TimeoutException is raised.
            signal.alarm(TIMEOUT)

            # Instantiate the digitizer for this image. Models are passed in
            # so they are not reloaded on every iteration.
            ecg = ECGImage(
                box_model=box_model,
                segmentation_model=segmentation_model,
                lead_name_model=lead_name_model,
                pulse_model=pulse_model,
                image_path=file_path,
            )

            # Run all pipeline stages: preprocessing → segmentation → YOLO
            # detection (×3) → scale calibration → grid construction →
            # signal extraction. Internally retries at different image sizes
            # if any detection returns empty.
            ecg.run_full_pipeline()

            # Write one CSV per image: columns = lead names, rows = time samples.
            ecg.save_signals_as_csv(base_name, directory=output_dir)
            success += 1

        except TimeoutException:
            # The image took longer than TIMEOUT seconds. Counted separately
            # from errors so they can be investigated independently.
            timeouts += 1
            print(f"    ⏰ TIMEOUT: {base_name}", flush=True)

        except Exception as e:
            # Any other failure (bad image, detection failure, calibration
            # error, etc.) is caught here so the loop continues with the
            # remaining images.
            errors += 1
            print(f"    ❌ ERROR: {base_name} → {e}", flush=True)

        finally:
            # Always cancel the alarm, even if an exception was raised, to
            # prevent a stale countdown from interrupting the next image.
            signal.alarm(0)

            # Release VRAM after each image to prevent fragmentation across
            # a long batch run. synchronize() ensures all GPU ops are complete
            # before empty_cache() frees the memory.
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    total = len(files)
    print(f"    → {success}/{total} ok | {errors} errors | {timeouts} timeouts\n",
          flush=True)
    return success, total

# -------------------------
# Main loop
# -------------------------

# Discover hospital directories. Each immediate child directory of
# ORGANIZED_DIR is treated as one hospital; files at the root level are ignored.
hospitals = sorted([
    name for name in os.listdir(ORGANIZED_DIR)
    if os.path.isdir(os.path.join(ORGANIZED_DIR, name))
])

# Accumulate per-hospital/category counts for the summary report.
log_rows = []

print(f"Found {len(hospitals)} hospital(s). Starting digitization...\n", flush=True)

for hospital in hospitals:
    print(f"{'='*60}", flush=True)
    print(f"Hospital: {hospital}", flush=True)
    print(f"{'='*60}", flush=True)

    for cat in CATEGORIES:
        input_dir  = os.path.join(ORGANIZED_DIR, hospital, cat)
        output_dir = os.path.join(OUTPUT_DIR,    hospital, cat)

        # Skip silently if this category does not exist for this hospital.
        if not os.path.isdir(input_dir):
            continue

        print(f"  [{cat.upper()}]", flush=True)
        success, total = process_folder(input_dir, output_dir)

        # Only log categories that had at least one image, to keep the
        # summary table free of zero-row entries.
        if total > 0:
            log_rows.append({
                "hospital": hospital,
                "category": cat,
                "total"   : total,
                "success" : success,
                "failed"  : total - success,
                "rate"    : f"{100 * success / total:.1f}%",
            })

# -------------------------
# Summary report
# -------------------------

print("\n" + "="*60, flush=True)
print("DIGITIZATION SUMMARY", flush=True)
print("="*60, flush=True)

df = pd.DataFrame(log_rows)

if df.empty:
    print("No images were processed.", flush=True)
else:
    # Flat per-hospital/category breakdown.
    print("\nPer hospital / category:\n")
    print(df.to_string(index=False))

    # Pivot to a hospital × category matrix of successful digitizations,
    # with a TOTAL row and column for quick inspection.
    print("\n\nSuccessful digitizations — distribution:\n")
    success_pivot = (
        df.pivot_table(index="hospital", columns="category",
                       values="success", aggfunc="sum", fill_value=0)
          .reindex(columns=CATEGORIES, fill_value=0)  # preserve CATEGORIES column order
    )
    success_pivot["TOTAL"] = success_pivot.sum(axis=1)   # row totals
    success_pivot.loc["TOTAL"] = success_pivot.sum(axis=0)  # grand total row
    print(success_pivot.to_string())

    # Overall counts across the entire dataset.
    total_imgs   = df["total"].sum()
    total_ok     = df["success"].sum()
    total_failed = df["failed"].sum()
    print(f"\n✅ Overall: {total_ok}/{total_imgs} successful "
          f"({100*total_ok/total_imgs:.1f}%) | "
          f"{total_failed} failed", flush=True)

    # Persist the pivot table as a CSV for later analysis.
    summary_path = os.path.join(OUTPUT_DIR, "digitization_summary.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_pivot.to_csv(summary_path)
    print(f"📄 Summary saved to {summary_path}", flush=True)

print("\nAll processing complete.", flush=True)
