"""
Extended Data Fig: Consistency check with Stroeve et al. 2012 (CMIP5 models).
Compares observed SIV trend (and optional 0-D schematic) to the distribution of
model trends from the table. Replace STROEVE_PRIMARY with exact values from the paper.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# -----------------------------------------------------------------------------
# Stroeve et al. 2012 table: 20 models, primary metric (Column 5)
# Replace with exact values from the table (same units as your obs trend)
STROEVE_MODEL_NAMES = [
    "CanESM2", "CSIRO Mk3-6-0", "CNRM-CM5", "GFDL-CM3", "GISS-E2-R",
    "HadCM3", "HadGEM2-CC", "HadGEM2-ES", "INMCM4", "IPSL-CM5A-LR",
    "IPSL-CM5A-MR", "MIROC4h", "MIROC5", "MIROC-ESM", "MIROC-ESM-CHEM",
    "MPI-ESM-LR", "MRI-CGCM3", "NCAR-CCSM4", "NCAR-CESM", "NorESM1-M",
]
# Primary metric (Column 5) – replace with your actual table values
# STROEVE_PRIMARY = np.array([
#     -0.384, -0.103, -0.534, -0.678, -0.234, -1.092,
#     -0.45, -0.62, -0.31, -0.55, -0.48, -0.72, -0.39, -0.58, -0.41,
#     -0.50, -0.67, -0.44, -0.52, -0.36,
# ])

STROEVE_PRIMARY = np.array([
    -0.678, -0.234, -1.092, -0.486, -0.740, -0.424,
    -0.742, -0.081, -0.082, -0.385, -0.51, -0.821, -0.735, -0.75, -0.358,
    -0.34, -0.596, -0.456, -0.685, -0.410
])

# -----------------------------------------------------------------------------
# Your observed data
DATA_DIR = "/Volumes/Yotta_1"
OCT_SIV_FILE = os.path.join(DATA_DIR, "october_siv_ensemble.csv")
USE_SYNTHETIC_OBS = False  # set True if no file; then set OBS_TREND
OBS_TREND = -0.52  # same units as STROEVE_PRIMARY (e.g. % per decade)


def load_observed_siv(filepath):
    """Load Year, Volume (or Ensemble). Return years and volume array."""
    try:
        df = pd.read_csv(filepath)
        if "Volume" not in df.columns:
            if "Ensemble" in df.columns:
                df = df.rename(columns={"Ensemble": "Volume"})
            elif "Vol_Ensemble" in df.columns:
                df = df.rename(columns={"Vol_Ensemble": "Volume"})
        years = df["Year"].values.astype(float)
        vol = df["Volume"].values.astype(float)
        return years, vol
    except Exception as e:
        print(f"Could not load {filepath}: {e}")
        return None, None


def trend_per_decade(years, y_vals, year_min=1979, year_max=2011):
    """Linear trend: change in y per decade (slope * 10)."""
    mask = (years >= year_min) & (years <= year_max)
    x, y = years[mask], y_vals[mask]
    if len(x) < 5:
        return np.nan
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope * 10


def main():
    year_min, year_max = 1979, 2011
    obs_trend = np.nan
    if not USE_SYNTHETIC_OBS:
        years_obs, vol_obs = load_observed_siv(OCT_SIV_FILE)
        if years_obs is not None and len(years_obs) > 5:
            trend_vol_decade = trend_per_decade(years_obs, vol_obs, year_min, year_max)  # km³/decade
            mean_vol = np.nanmean(vol_obs[(years_obs >= year_min) & (years_obs <= year_max)])
            if mean_vol and np.isfinite(trend_vol_decade):
                obs_trend = (trend_vol_decade / mean_vol) * 100  # % per decade
            else:
                obs_trend = trend_vol_decade
        else:
            obs_trend = OBS_TREND
    else:
        obs_trend = OBS_TREND

    print(f"Observed trend (1979-2011): {obs_trend:.3f}")
    print(f"Stroeve models: mean = {np.mean(STROEVE_PRIMARY):.3f}, std = {np.std(STROEVE_PRIMARY):.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        [STROEVE_PRIMARY],
        positions=[0],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.8),
        medianprops=dict(color="darkblue", lw=2),
    )
    ax.scatter([0], [obs_trend], s=120, color="red", zorder=5, edgecolors="black", linewidths=2, label="This study (observed)")
    ax.axhline(0, color="gray", ls="--", alpha=0.7)
    ax.set_ylabel("Trend (same units as Stroeve et al. 2012)", fontsize=12)
    ax.set_xticks([0])
    ax.set_xticklabels(["CMIP5 models\n(Stroeve et al. 2012)"])
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    # ax.set_title("Consistency check: observed trend vs CMIP5 model distribution")
    plt.tight_layout()
    out_path = "Extended_Fig_scope_check.png"
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(base, out_path)
    except Exception:
        pass
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    df_ref = pd.DataFrame({"Model": STROEVE_MODEL_NAMES, "Primary_metric": STROEVE_PRIMARY})
    csv_path = out_path.replace(".png", "_stroeve2012_ref.csv")
    df_ref.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
