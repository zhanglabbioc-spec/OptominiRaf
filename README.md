# Calcium Imaging Analysis Code  

Analysis scripts used for the manuscript:

**[Title]**  
**[Authors]**

This repository contains the exact code used to compute the quantitative results reported in the paper.  
It is provided for transparency and reproducibility and is not a general-purpose software package.

---

## Data Input

Suite2p-extracted fluorescence traces:

- `F.npy` (cells × frames)

ROI selection was performed during Suite2p processing.

---

## Analysis Pipelines

### 1. In Vivo Activity Analysis

- Sensor: jRGECO1a  
- Sampling frequency: 1 Hz  
- Rolling-minimum baseline (20 s window)  
- ΔF/F₀ calculation  
- Peak detection (threshold = 0.2 ΔF/F₀, minimum distance = 1 s)  

Outputs:
- ΔF/F₀ traces  
- Per-cell peak count, frequency, and amplitude  

---

### 2. In Vitro Synchronized Network Analysis

- Sensor: jRCaMP1a  
- Sampling frequency: 4 Hz  
- Rolling-minimum baseline (10 s window)  
- Gaussian smoothing (σ = 2 frames)  
- Per-cell event detection (baseline + 3σ, min length ≥ 3 frames)  
- Functional cell filtering (max ΔF/F₀ ≥ 0.05)  
- Synchronized burst detection (fraction active ≥ 0.3)  

Note: Imaging was acquired in temporal segments.  
Raw segments were stitched prior to Suite2p processing to maintain consistent ROI tracking across time.  

Outputs:
- Binary peak matrices  
- Fraction-active traces  
- Network burst summary statistics  

---
## Contact

[Kai Zhang，
University of Illinois at Urbana-Champaign，
kaizkaiz@illinios.edu]

[Huaxun Fan，
University of Illinois at Urbana-Champaign，
huaxunf2@illinois.edu]



