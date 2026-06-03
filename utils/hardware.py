# utils/hardware.py
"""
Cached hardware detection for the ECG Dashboard sidebar.
Uses @st.cache_data to avoid re-running expensive subprocess/torch calls on every Streamlit rerun.
"""

import platform
import subprocess
import streamlit as st


@st.cache_data(show_spinner=False)
def get_cpu_name() -> str:
    """Detect CPU name. Cached so it only runs once per session."""
    try:
        return subprocess.check_output(
            "wmic cpu get name", shell=True
        ).decode().split("\n")[1].strip()
    except Exception:
        return platform.processor().split(',')[0].strip()


@st.cache_data(show_spinner=False)
def get_gpu_info() -> dict:
    """
    Detect GPU availability and details.
    Cached so the heavy `import torch` only happens once per session.
    Returns dict with keys: available, name, vram_gb.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
    except ImportError:
        pass
    return {"available": False, "name": None, "vram_gb": None}


def render_sidebar_hardware():
    """Render the hardware info panel in the sidebar using cached data."""
    st.sidebar.markdown("### 🔍 Clinical Workstation Info")

    cpu = get_cpu_name()
    st.sidebar.markdown(f"**🖥️ CPU:**\n`{cpu}`")

    gpu = get_gpu_info()
    if gpu["available"]:
        st.sidebar.markdown(
            f"**🚀 GPU:**\n`{gpu['name']}`\n"
            f"`CUDA Active | VRAM: {gpu['vram_gb']:.1f} GB`"
        )
    else:
        st.sidebar.markdown("**🚀 GPU:**\n`Not Detected (Running on CPU)`")
