"""
In Vivo Calcium Imaging Analysis

This script processes in vivo calcium imaging fluorescence traces
extracted from Suite2p (F.npy).

Experimental details:
    - Sensor: jRGECO
    - Imaging frequency: 1 Hz
    - ROIs were manually curated in Suite2p
      No additional iscell-based filtering is applied in this script.

Analysis steps:
    1. Baseline estimation (F0) using a rolling minimum window (20 frames).
    2. ΔF/F0 calculation: (F − F0) / F0
    3. Peak detection using scipy.signal.find_peaks
       - Height threshold: 0.2 ΔF/F0
       - Minimum peak distance: 1 frame (1 second at 1 Hz)

Input:
    F.npy — 2D array (cells x timepoints), exported from Suite2p

Output:
    *_DeltaF.csv — normalized ΔF/F0 traces
    *_Peaks.csv  — per-cell peak statistics:
        - Peak_Count
        - Peak_Frequency (events per second; 1 Hz acquisition)
        - Avg_Peak_Amplitude

Author: Huaxun Fan
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Function to compute F0 using the minimum in a rolling window
def compute_F0_min(trace, window_size=20):
    """Compute F0 as the minimum value in a rolling window."""
    F0_values = np.zeros_like(trace)
    for i in range(len(trace)):
        start_idx = max(0, i - window_size + 1)
        rolling_window_values = trace[start_idx:i+1]
        F0_values[i] = np.min(rolling_window_values) if len(rolling_window_values) > 0 else trace[i]
    return F0_values

# Function to process a single file
def process_single_file(file_path, output_folder, sampling_rate=1):
    # Peak detection parameters
    peak_distance = 1 * sampling_rate  # Minimum distance between peaks (in seconds)
    peak_threshold = 0.2  # Fixed peak height threshold
    window_size = 20  # Rolling window for F0 calculation

    try:
        data = np.load(file_path, allow_pickle=True)

        # Ensure the data is a 2D array
        if data.ndim != 2:
            print(f"Skipping {file_path}: Not a 2D array")
            return None, None

        num_cells, num_seconds = data.shape

        # Compute ΔF/F0 for all cells
        deltaF_F0_all = np.zeros_like(data, dtype=np.float32)
        for i in range(num_cells):
            F0_min = compute_F0_min(data[i], window_size)
            deltaF_F0_all[i] = (data[i] - F0_min) / np.maximum(F0_min, 1e-6)  # Avoid division by zero

        # Run peak detection
        peak_data = []
        for i in range(num_cells):
            peaks, properties = find_peaks(deltaF_F0_all[i], height=peak_threshold, distance=peak_distance)
            num_peaks = len(peaks)
            peak_freq = num_peaks / num_seconds  # Peaks per second
            avg_peak_amplitude = np.mean(properties["peak_heights"]) if num_peaks > 0 else 0  # Avg peak height
            peak_data.append([i, num_peaks, peak_freq, avg_peak_amplitude])

        return deltaF_F0_all, peak_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Function to process all valid F.npy files in subdirectories
def process_npy_folders(root_folder, sampling_rate=1):
    results = {}

    # Traverse all subdirectories
    for subdir, _, files in os.walk(root_folder):
        if "F.npy" in files:
            file_path = os.path.join(subdir, "F.npy")

            # Extract B (the second-level folder)
            relative_path = os.path.relpath(subdir, root_folder)  # Get relative path
            path_parts = relative_path.split(os.sep)  # Split by folder structure
            
            if len(path_parts) >= 3:  # Ensure path is at least A/B/C/D
                output_folder = os.path.join(root_folder, path_parts[0])  # B folder

                # Process the file
                deltaF_F0_all, peak_data = process_single_file(file_path, output_folder, sampling_rate)

                if deltaF_F0_all is not None:
                    # Save ΔF/F0 results
                    deltaF_output_path = os.path.join(output_folder, f"{path_parts[0]}_DeltaF.csv")
                    pd.DataFrame(deltaF_F0_all).to_csv(deltaF_output_path, index=False)

                    # Save peak statistics
                    peaks_output_path = os.path.join(output_folder, f"{path_parts[0]}_Peaks.csv")
                    pd.DataFrame(peak_data, columns=['Cell_Index', 'Peak_Count', 'Peak_Frequency', 'Avg_Peak_Amplitude']).to_csv(peaks_output_path, index=False)

                    print(f"Processed {file_path} -> Saved results to {output_folder}")
                    results[file_path] = (deltaF_output_path, peaks_output_path)

    return results

# Example usage: Update with your actual root folder
root_folder = "xxx"
process_npy_folders(root_folder)

