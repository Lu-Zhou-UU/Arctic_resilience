import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Physical constants for SIV ----
RHO_I = 917.0
L_I = 3.34e5
KAPPA = RHO_I * L_I

# ---- Reference area ----
AREF_KM2 = 25.0 * 25.0
AREF_M2 = AREF_KM2 * 1e6


def load_monthly_extent(filepath, start_date="1979-01-01"):
    df = pd.read_csv(filepath, header=None, names=["Extent_km2"])
    dates = pd.date_range(start=pd.Timestamp(start_date), periods=len(df), freq="MS")
    df["Date"] = dates
    # Known discontinuity in this local file: monthly phase flips around 1986-04.
    # Apply a 4-month timestamp shift from the breakpoint onward.
    breakpoint = pd.Timestamp("1986-04-01")
    mask = df["Date"] >= breakpoint
    if mask.any():
        df.loc[mask, "Date"] = df.loc[mask, "Date"] + pd.DateOffset(months=4)
        df = df.sort_values("Date").drop_duplicates(subset="Date", keep="first")
    df["Extent_million_km2"] = df["Extent_km2"] / 1e6
    return df[["Date", "Extent_million_km2"]].sort_values("Date").reset_index(drop=True)


def extract_month_series(df, month):
    s = df.loc[df["Date"].dt.month == month, ["Date", "Extent_million_km2"]].copy()
    s["Year"] = s["Date"].dt.year
    s = s.drop_duplicates(subset="Year", keep="last").sort_values("Year")
    return s[["Year", "Extent_million_km2"]]


def rolling_variance_annual(values, years, window_years):
    out_year = []
    out_var = []
    vals = np.asarray(values, dtype=float)
    yrs = np.asarray(years, dtype=int)
    for i in range(window_years - 1, len(vals)):
        sub = vals[i - window_years + 1:i + 1]
        if np.isfinite(sub).sum() >= 5:
            out_year.append(int(yrs[i]))
            out_var.append(float(np.var(sub, ddof=1)))
    return np.asarray(out_year), np.asarray(out_var)


def load_ice_volume_data(filename):
    return pd.read_csv(filename, sep=" ", header=None, names=["Year", "Volume"]).dropna()


def harmonize_volume(c3_df, piomas_df):
    c3 = c3_df[["Year", "Volume"]].rename(columns={"Volume": "Vol_C3S"}).copy()
    pm = piomas_df[["Year", "Volume"]].rename(columns={"Volume": "Vol_PIOMAS"}).copy()
    overlap = pd.merge(c3, pm, on="Year", how="inner")

    if len(overlap) >= 5:
        A = np.vstack([overlap["Vol_PIOMAS"].values, np.ones(len(overlap))]).T
        a, b = np.linalg.lstsq(A, overlap["Vol_C3S"].values, rcond=None)[0]
    else:
        a, b = 1.0, 0.0

    pm["Vol_PIOMAS_bc"] = a * pm["Vol_PIOMAS"] + b
    merged = pd.merge(c3, pm[["Year", "Vol_PIOMAS_bc"]], on="Year", how="outer").sort_values("Year")
    merged["Vol_ens_km3"] = merged[["Vol_C3S", "Vol_PIOMAS_bc"]].mean(axis=1)
    return merged


def volume_to_energy_jm2(vol_km3):
    v_m3 = np.asarray(vol_km3, dtype=float) * 1e9
    h_m = v_m3 / AREF_M2
    return -KAPPA * h_m


def main():
    out_dir = "/Users/jay"
    # ---------- Inputs ----------
    sie_file = "/Volumes/Yotta_1/SIE_monthly_197901_202506.txt"
    c3_apr_file = "/Volumes/Yotta_1/C3_ice_volume_April.txt"
    pm_apr_file = "/Volumes/Yotta_1/PIOMAS_ice_volume_April.txt"
    c3_oct_file = "/Volumes/Yotta_1/C3_ice_volume_October.txt"
    pm_oct_file = "/Volumes/Yotta_1/PIOMAS_ice_volume_October.txt"

    # ---------- SIE annual series (April/October) ----------
    sie = load_monthly_extent(sie_file)
    sie_apr = extract_month_series(sie, month=4)
    sie_oct = extract_month_series(sie, month=10)

    # ---------- SIV annual series (April/October), harmonized ----------
    siv_apr = harmonize_volume(load_ice_volume_data(c3_apr_file), load_ice_volume_data(pm_apr_file))
    siv_oct = harmonize_volume(load_ice_volume_data(c3_oct_file), load_ice_volume_data(pm_oct_file))

    # ---------- Figure Sx: underlying annual series ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(sie_oct["Year"], sie_oct["Extent_million_km2"], color="#D4AF37", marker="s", lw=2, ms=5, label="October")
    ax1.plot(sie_apr["Year"], sie_apr["Extent_million_km2"], color="#6A4C93", marker="o", lw=2, ms=5, label="April")
    ax1.set_ylabel("SIE (million km$^2$)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Annual month-specific SIE series", loc="left", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(framealpha=0.95)

    ax2.plot(siv_oct["Year"], siv_oct["Vol_ens_km3"], color="#D4AF37", marker="s", lw=2, ms=5, label="October")
    ax2.plot(siv_apr["Year"], siv_apr["Vol_ens_km3"], color="#6A4C93", marker="o", lw=2, ms=5, label="April")
    ax2.set_ylabel("SIV (km$^3$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Annual month-specific SIV series (harmonized ensemble)", loc="left", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, linestyle="--")

    for yr in [2007, 2012, 2020]:
        ax1.axvline(yr, color="darkslategray", linestyle="--", alpha=0.5)
        ax2.axvline(yr, color="darkslategray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/Supplementary_Fig_Sx_Annual_SIE_SIV_series.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure Sy: rolling variance ----------
    # Windows matched to AC1 analysis: 10-year for SIE, 11-year for SIV
    yrs_sie_oct, var_sie_oct = rolling_variance_annual(sie_oct["Extent_million_km2"].values, sie_oct["Year"].values, 10)
    yrs_sie_apr, var_sie_apr = rolling_variance_annual(sie_apr["Extent_million_km2"].values, sie_apr["Year"].values, 10)

    e_apr = volume_to_energy_jm2(siv_apr["Vol_ens_km3"].values)
    e_oct = volume_to_energy_jm2(siv_oct["Vol_ens_km3"].values)
    yrs_siv_oct, var_siv_oct = rolling_variance_annual(e_oct, siv_oct["Year"].values, 11)
    yrs_siv_apr, var_siv_apr = rolling_variance_annual(e_apr, siv_apr["Year"].values, 11)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(yrs_sie_oct, var_sie_oct, color="#D4AF37", marker="s", lw=2, ms=5, label="October")
    ax1.plot(yrs_sie_apr, var_sie_apr, color="#6A4C93", marker="o", lw=2, ms=5, label="April")
    ax1.set_ylabel("Rolling variance (SIE)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) SIE rolling variance (10-year window)", loc="left", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(framealpha=0.95)

    ax2.plot(yrs_siv_oct, var_siv_oct, color="#D4AF37", marker="s", lw=2, ms=5, label="October")
    ax2.plot(yrs_siv_apr, var_siv_apr, color="#6A4C93", marker="o", lw=2, ms=5, label="April")
    ax2.set_ylabel("Rolling variance (October/April $E_{obs}$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax2.set_title("(b) SIV-derived energy rolling variance (11-year window)", loc="left", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, linestyle="--")

    for yr in [2007, 2012, 2020]:
        ax1.axvline(yr, color="darkslategray", linestyle="--", alpha=0.5)
        ax2.axvline(yr, color="darkslategray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/Supplementary_Fig_Sy_Rolling_Variance_SIE_SIV.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_dir}/Supplementary_Fig_Sx_Annual_SIE_SIV_series.png")
    print(f"Saved {out_dir}/Supplementary_Fig_Sy_Rolling_Variance_SIE_SIV.png")


if __name__ == "__main__":
    main()
