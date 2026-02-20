"""
In Vitro Calcium Imaging — Synchronized Network Burst Analysis

This script detects synchronized network bursts in in vitro calcium imaging
datasets acquired using the genetically encoded calcium indicator jRCaMP1a.

Acquisition details:
    - Sensor: jRCaMP1a
    - Sampling frequency: FS = 4.0 Hz
    - Imaging was acquired in multiple temporal segments.
      Raw segments were stitched prior to Suite2p processing
      to ensure consistent ROI tracking across time.
    - ROIs were identified in Suite2p and exported as F.npy.

Analysis pipeline:
    1. Compute ΔF/F₀ using rolling-minimum baseline (10 s window).
    2. Filter active cells (max ΔF/F₀ ≥ 0.05).
    3. Split recording into 3000-frame segments (750 s per segment at 4 Hz).
    4. Apply Gaussian smoothing (σ = 2 frames).
    5. Detect per-cell events:
         - Threshold = baseline (30th percentile) + 3σ
         - Minimum event length ≥ 3 frames
         - Absolute ΔF/F₀ floor ≥ 0.01
    6. Construct binary peak raster (1 at event peak frame).
    7. Compute fraction of active cells per frame.
    8. Define synchronized network bursts when fraction_active ≥ 0.3.

Input:
    F.npy — 2D array (cells × frames), exported from Suite2p.

Outputs (per dataset folder):
    binary_segN.csv
    fraction_active_segN.csv
    burst_summary.csv

Author: Huaxun Fan
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d, gaussian_filter1d

# ---------------------------
# Parameters
# ---------------------------
MASTER_DIR = "xxx"   # <--- CHANGE THIS
FS = 4.0                              # sampling frequency (Hz)
ROLLING_MIN_SEC = 10.0                # baseline window (s)
ACTIVE_CUTOFF = 0.05                  # keep cells with max ΔF/F₀ ≥ this
SIGMA_SMOOTH = 2.0                    # Gaussian σ (frames)
SIGMA_MULT = 3.0                      # baseline + 3σ
BASELINE_PCT = 30.0                   # percentile for baseline
ABS_FLOOR = 0.01                      # absolute ΔF/F₀ floor
MIN_LEN = 3                           # minimum event length (frames)
SEG_LEN = 3000                        # frames per segment
SYNC_THRESHOLD = 0.3                  # fraction-active threshold for synchronized bursts

# ---------------------------
# Helper functions
# ---------------------------
def rolling_min_baseline(trace, window_frames):
    return minimum_filter1d(trace, size=window_frames, mode="nearest")

def compute_baseline_sigma(trace, pct=30.0):
    base = np.percentile(trace, pct)
    noise_vals = trace[trace <= base]
    sigma = np.std(noise_vals) if noise_vals.size > 0 else np.std(trace)
    return base, sigma

def detect_events(trace):
    base, sigma = compute_baseline_sigma(trace, BASELINE_PCT)
    thr = base + SIGMA_MULT * sigma
    above = trace >= thr
    diff = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    evs = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= MIN_LEN:
            seg = trace[s:e+1]
            pk_rel = np.argmax(seg)
            pk_idx = s + pk_rel
            pk_val = trace[pk_idx]
            if pk_val >= ABS_FLOOR:
                evs.append((s, e, pk_idx, pk_val))
    return evs

def compute_dff(F):
    win = int(ROLLING_MIN_SEC * FS)
    F0 = np.array([rolling_min_baseline(f, win) for f in F])
    return (F - F0) / (F0 + 1e-6)

def segment_array(arr):
    n_cells, n_frames = arr.shape
    n_segs = n_frames // SEG_LEN
    return [arr[:, i*SEG_LEN:(i+1)*SEG_LEN] for i in range(n_segs)]

# ---------------------------
# Dataset processing
# ---------------------------
def process_dataset(fpath: Path):
    folder = fpath.parent
    print(f"[RUN] {folder}")

    F = np.load(fpath, allow_pickle=True)
    if F.ndim == 1:
        F = np.expand_dims(F, 0)
    n_cells, n_frames = F.shape

    # Compute ΔF/F₀
    dff = compute_dff(F)

    # Filter active cells
    active_mask = dff.max(axis=1) >= ACTIVE_CUTOFF
    if not np.any(active_mask):
        print("  No active cells, skipped.")
        return
    dff_active = dff[active_mask]

    # Split before smoothing
    segments = segment_array(dff_active)
    summary_rows = []

    for si, seg in enumerate(segments, start=1):
        seg_smooth = gaussian_filter1d(seg, sigma=SIGMA_SMOOTH, axis=1)
        n_cells, n_frames = seg_smooth.shape

        # --- Event detection and binary matrix ---
        binary = np.zeros_like(seg_smooth, dtype=int)
        for i, tr in enumerate(seg_smooth):
            evs = detect_events(tr)
            for (_, _, pk_idx, _) in evs:
                binary[i, pk_idx] = 1

        # Save binary matrix
        binary_csv = folder / f"binary_seg{si}.csv"
        np.savetxt(binary_csv, binary, fmt="%d", delimiter=",")

        # --- Fraction-active trace ---
        frac_active = np.sum(binary, axis=0) / n_cells
        frac_csv = folder / f"fraction_active_seg{si}.csv"
        np.savetxt(frac_csv, frac_active, fmt="%.6f", delimiter=",")

        # --- Compute synchronized burst stats ---
        above = frac_active >= SYNC_THRESHOLD
        diff = np.diff(above.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        burst_count = len(starts)
        burst_freq = burst_count / (SEG_LEN / FS)
        burst_peaks = []
        for s, e in zip(starts, ends):
            burst_peaks.append(np.max(frac_active[s:e+1]) if e > s else 0)
        mean_burst_height = np.mean(burst_peaks) if burst_peaks else 0

        # Segment-level summary
        mean_frac = np.mean(frac_active)
        max_frac = np.max(frac_active)
        std_frac = np.std(frac_active)
        summary_rows.append({
            "segment_id": si,
            "burst_count": burst_count,
            "burst_frequency_Hz": burst_freq,
            "mean_burst_height": mean_burst_height,
            "mean_fraction_active": mean_frac,
            "max_fraction_active": max_frac,
            "std_fraction_active": std_frac
        })

        print(f"  Segment {si}: bursts={burst_count}, freq={burst_freq:.3f} Hz, mean height={mean_burst_height:.3f}")

    # --- Save per-dataset summary ---
    summary_df = pd.DataFrame(summary_rows)
    out_csv = folder / "burst_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved summary: {out_csv}\n")

# ---------------------------
# Loop over all datasets
# ---------------------------
def main():
    master = Path(MASTER_DIR)
    all_files = list(master.rglob("F.npy"))
    if not all_files:
        print(f"No F.npy files found under {master}")
        return
    for fpath in all_files:
        try:
            process_dataset(fpath)
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")

if __name__ == "__main__":
    main()
