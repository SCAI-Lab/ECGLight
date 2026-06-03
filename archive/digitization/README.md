# archive/digitization

Legacy ECG image digitization scripts that have been **superseded** by the
Streamlit dashboard in the workspace root.

## Contents

| File | Status | Reason for archival |
| --- | --- | --- |
| `run_ecg.py` | Legacy batch digitizer | Processes a directory tree of ECG paper scans (one CSV per image). Uses Linux-only `SIGALRM` timeouts and hard-coded model paths under `Submission code/...`. Replaced by the interactive pipeline in `backend/digitization_runner.py` invoked from `app.py`. |

## Current digitization entry point

Use the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open the **"📷 Image Digitizer"** (now **"ECG Image Digitizer"**) page and
either upload an ECG scan or check the sample image option.
