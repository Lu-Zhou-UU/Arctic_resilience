# Arctic Sea Ice Resilience: Reproducibility Guide

This repository contains analysis scripts used for the manuscript on Arctic sea ice resilience and tipping behavior.

## Scope

- **Main goal**: reproduce figures, metrics, and robustness checks reported in the manuscript.
- **Current status**: this GitHub repo started as a raw code snapshot and has been extended with a reproducibility-oriented structure and script map.

## Repository Structure

- `Plot-codes.ipynb`  
  Core model visualization workflows (e.g., bifurcation/potential conceptual plots).
- `Sea_ice_energy_new.py`  
  Core empirical-potential pipeline from observed SIV-derived energy.
- `combined_lag_new.py`  
  Rolling AC1 diagnostics for SIE/SIV (April/October).
- `SIE2resilient.py`, `SIV_lag_new.py`, `AVHRR_1982_2024_SIV2energy.py`, `Harmony_check.py`, `combine.py`, `compare_albedo.py`  
  Supporting preprocessing and diagnostic scripts.
- `extract_CS_thickness.m`, `Plot_NH.m`  
  MATLAB helper scripts.

## Data Requirements

Place input files either:
1. in the repository root, or
2. update file paths in scripts to your local locations.

Typical required files include:

- `SIE_monthly_197901_202506.txt`
- `C3_ice_volume_April.txt`
- `C3_ice_volume_October.txt`
- `PIOMAS_ice_volume_April.txt`
- `PIOMAS_ice_volume_October.txt`

## Python Environment

Recommended: Python 3.10+ (tested with 3.12).

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels jupyter
```

## Reproduction Order (Suggested)

1. **Volume harmonization / preprocessing**
   - `Harmony_check.py`
   - `AVHRR_1982_2024_SIV2energy.py`
2. **AC1 diagnostics (main text Figure 4-style)**
   - `combined_lag_new.py`
3. **Empirical potential and bootstrap metrics (main text Figure 5-style)**
   - `Sea_ice_energy_new.py`
4. **Conceptual model and potential plots (Figure 2/3/7-style)**
   - `Plot-codes.ipynb` and associated model scripts
5. **Robustness/sensitivity tests**
   - (see manuscript-updated scripts listed below)

## Script-to-Figure Map (Manuscript)

- **Figure 2 (bifurcation)**: `Plot-codes.ipynb`
- **Figure 3 (resilience potential panels)**: `Plot-codes.ipynb` (updated forcing panels)
- **Figure 4 (AC1 SIE/SIV)**: `combined_lag_new.py`
- **Figure 5 (empirical potential + bootstrap summary)**: empirical potential workflow (`Sea_ice_energy_new.py` and manuscript-updated scripts)
- **Figure 6 (robustness/alternative hypotheses)**: manuscript robustness scripts
- **Figure 7 (stochastic potential / escape-time illustration)**: manuscript stochastic scripts

## Manuscript-Updated Scripts (Newer Local Versions)

The manuscript revision used newer local scripts that may be ahead of the original GitHub snapshot.
Common examples include:

- `empirical_resilience_potential.py`
- `robustness_alternative_hypotheses.py`
- `escape_time_white_vs_ou.py`
- `compare_obs_stroeve2012.py`
- `combined_lag_extra_figures.py`

If these are available locally (e.g., `/Volumes/user/`), sync them into this repo before final reruns.

## Reproducibility Notes

- Use fixed analysis windows and epoch definitions exactly as in manuscript (`1982--2005`, `2006--2024` where applicable).
- Keep KDE bandwidth/sigma options consistent across runs when comparing figures.
- Report bootstrap settings (`nboot`, block size, and barrier-identification criteria) in figure captions or methods.
- Keep caption values synchronized with plotted panel values (especially Figures 3, 5, and 7).

## Output Tracking

Recommended: keep generated outputs in `outputs/` and avoid committing large binary files unless needed for release.

Example:

```bash
mkdir -p outputs
python combined_lag_new.py
```

## Citation and License

- License: GPL-2.0 (`LICENSE`)
- Repository: <https://github.com/Lu-Zhou-UU/Arctic_resilience>
