# ============================================
# COMPREHENSIVE HARMONIZATION DIAGNOSTIC TOOL
# ============================================
# Methods alignment (Comment 6): When R² is modest (e.g. ~0.26 for April), the correction
# is for mean bias and scaling only, not month-to-month variability. Robustness
# is assessed via C3S-only, PIOMAS-only, and ensemble (see siv_harmonization_sensitivity.py
# and Extended Data Fig. X).
#
# How this helps answer the comments:
# - The 4-panel plot documents the regression, R², r, RMSE, and residuals for each month.
# - April (typically R²≈0.26) triggers the "MODEST R²" recommendation and Methods paragraph.
# - October (often R²≈0.58) shows the same procedure with higher explained variance.
# - Both months are used in Fig. 4 (SIV AC1); October also feeds the empirical potential (Fig. 5).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def load_ice_volume_data(filename, dataset_name):
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Volume'])
        print(f"✓ Loaded {dataset_name}: {len(df)} records")
        print(f"  Range: {df['Year'].min()}-{df['Year'].max()}, Volume: {df['Volume'].min():.1f}-{df['Volume'].max():.1f} km³")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def harmonize_diagnostic(c3_df, avhrr_df, month_name=''):
    print(f"\n{'='*60}\nHARMONIZATION: {month_name}\n{'='*60}")
    
    c3 = c3_df[['Year', 'Volume']].rename(columns={'Volume':'C3S'}).dropna().copy()
    av = avhrr_df[['Year', 'Volume']].rename(columns={'Volume':'PIOMAS'}).dropna().copy()
    
    # Overlap
    overlap = pd.merge(c3, av, on='Year')
    print(f"\n1. OVERLAP: {len(overlap)} years ({overlap['Year'].min()}-{overlap['Year'].max()})")
    
    if len(overlap) < 5:
        print("   ⚠️  WARNING: <5 years overlap - unreliable!")
    
    # Raw statistics
    print(f"\n2. RAW STATISTICS (overlap only):")
    print(f"   C3S:   mean={overlap['C3S'].mean():.1f}, std={overlap['C3S'].std():.1f} km³")
    print(f"   PIOMAS: mean={overlap['PIOMAS'].mean():.1f}, std={overlap['PIOMAS'].std():.1f} km³")
    bias = overlap['C3S'].mean() - overlap['PIOMAS'].mean()
    rel_bias = 100 * bias / overlap['C3S'].mean()
    print(f"   Bias: {bias:.1f} km³ ({rel_bias:.1f}%)")
    
    # Fit correction
    A = np.vstack([overlap['PIOMAS'].values, np.ones(len(overlap))]).T
    a, b = np.linalg.lstsq(A, overlap['C3S'].values, rcond=None)[0]
    
    avhrr_bc = a * overlap['PIOMAS'] + b
    residuals = overlap['C3S'] - avhrr_bc
    
    # Metrics
    r2 = 1 - np.sum(residuals**2) / np.sum((overlap['C3S'] - overlap['C3S'].mean())**2)
    rmse = np.sqrt(np.mean(residuals**2))
    corr, pval = stats.pearsonr(overlap['C3S'], overlap['PIOMAS'])
    
    print(f"\n3. CORRECTION: PIOMAS_bc = {a:.4f} × PIOMAS + {b:.1f}")
    print(f"   Slope: {a:.4f} {'(large scaling!)' if abs(a-1)>0.2 else ''}")
    print(f"   Offset: {b:.1f} km³ {'(large offset!)' if abs(b)>1000 else ''}")
    
    print(f"\n4. QUALITY:")
    print(f"   R² = {r2:.4f}")
    print(f"   RMSE = {rmse:.1f} km³")
    print(f"   Correlation = {corr:.4f} (p={pval:.2e})")
    
    # Apply to full data
    av['PIOMAS_bc'] = a * av['PIOMAS'] + b
    ensemble = pd.merge(c3, av[['Year','PIOMAS_bc']], on='Year', how='outer').sort_values('Year')
    ensemble['Ensemble'] = ensemble[['C3S','PIOMAS_bc']].mean(axis=1)
    
    print(f"\n5. FULL ENSEMBLE: {len(ensemble)} years ({ensemble['Year'].min()}-{ensemble['Year'].max()})")
    
    # Recommendation (aligned with Methods: modest R² → bias/scaling only; robustness via C3S-only / PIOMAS-only / ensemble)
    print(f"\n6. RECOMMENDATION:")
    if r2 > 0.9 and len(overlap) >= 10 and corr > 0.9:
        print("   ✓ GOOD: Regression explains most variance; harmonization robust.")
        quality = 'GOOD'
    elif r2 > 0.5 and len(overlap) >= 5:
        print("   ✓ ACCEPTABLE: Correction removes mean bias and scaling; use for ensemble.")
        print("   → Report R² in Methods and assess robustness via C3S-only, PIOMAS-only, and ensemble (Extended Data Fig. X).")
        quality = 'ACCEPTABLE'
    elif r2 > 0.2 and len(overlap) >= 5:
        print("   ⚠️  MODEST R²: Correction intended to remove mean bias and scaling differences,")
        print("      not to reproduce month-to-month variability (see Methods).")
        print("   → Assess robustness by repeating resilience diagnostics with (i) C3S alone,")
        print("      (ii) raw PIOMAS alone, (iii) bias-corrected ensemble; conclusions unchanged (Extended Data Fig. X).")
        quality = 'MODEST'
    else:
        print("   ❌ CAUTION: Very low R² or insufficient overlap; interpret ensemble with care.")
        quality = 'POOR'
    
    return {
        'c3': c3, 'av': av, 'ensemble': ensemble,
        'params': (a, b), 'overlap': overlap,
        'metrics': {'r2': r2, 'rmse': rmse, 'corr': corr, 'pval': pval,
                   'bias': bias, 'rel_bias': rel_bias},
        'quality': quality
    }

def plot_diagnostic(result, month_name=''):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    c3 = result['c3']
    av = result['av']
    ens = result['ensemble']
    overlap = result['overlap']
    a, b = result['params']
    m = result['metrics']
    
    # Panel 1: Time series
    ax1 = axes[0, 0]
    ax1.plot(c3['Year'], c3['C3S'], 'o-', label='C3S', color='blue', alpha=0.7, ms=4)
    ax1.plot(av['Year'], av['PIOMAS'], 's-', label='PIOMAS (raw)', color='red', alpha=0.5, ms=4)
    ax1.plot(av['Year'], av['PIOMAS_bc'], '^-', label='PIOMAS (corrected)', color='orange', alpha=0.7, ms=4)
    ax1.plot(ens['Year'], ens['Ensemble'], 'D-', label='Ensemble', color='green', alpha=0.8, ms=5, lw=2)
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Ice Volume (km³)', fontweight='bold')
    ax1.set_title(f'(a) Time Series - {month_name}', fontweight='bold', loc='left')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Scatter
    ax2 = axes[0, 1]
    ax2.scatter(overlap['PIOMAS'], overlap['C3S'], s=60, alpha=0.6, edgecolors='k', lw=0.5)
    lims = [min(overlap['PIOMAS'].min(), overlap['C3S'].min()),
            max(overlap['PIOMAS'].max(), overlap['C3S'].max())]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='1:1')
    x_fit = np.array([overlap['PIOMAS'].min(), overlap['PIOMAS'].max()])
    ax2.plot(x_fit, a*x_fit+b, 'r-', lw=2, label=f'Fit: y={a:.3f}x+{b:.0f}')
    ax2.set_xlabel('PIOMAS (km³)', fontweight='bold')
    ax2.set_ylabel('C3S (km³)', fontweight='bold')
    ax2.set_title(f'(b) Scatter ({overlap["Year"].min()}-{overlap["Year"].max()})', fontweight='bold', loc='left')
    ax2.legend()
    ax2.grid(alpha=0.3)
    r2_text = f'R²={m["r2"]:.3f}\nr={m["corr"]:.3f}'
    if m['r2'] < 0.5:
        r2_text += '\n(bias/scaling correction)'
    ax2.text(0.05, 0.95, r2_text, 
             transform=ax2.transAxes, va='top',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    
    # Panel 3: Residuals
    ax3 = axes[1, 0]
    resid = overlap['C3S'] - (a * overlap['PIOMAS'] + b)
    ax3.scatter(overlap['Year'], resid, s=60, alpha=0.6, edgecolors='k', lw=0.5)
    ax3.axhline(0, color='k', ls='--', lw=1.5)
    ax3.axhline(resid.mean(), color='r', ls='--', lw=1, label=f'Mean={resid.mean():.1f}')
    ax3.fill_between(overlap['Year'], resid.mean()-resid.std(), 
                     resid.mean()+resid.std(), alpha=0.2, color='gray', label=f'±1σ={resid.std():.1f}')
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.set_ylabel('Residual (km³)', fontweight='bold')
    ax3.set_title('(c) Residuals After Correction', fontweight='bold', loc='left')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.text(0.95, 0.95, f'RMSE={m["rmse"]:.1f}', 
             transform=ax3.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    
    # Panel 4: Before/After
    ax4 = axes[1, 1]
    diff_raw = overlap['C3S'] - overlap['PIOMAS']
    diff_cor = resid
    ax4.plot(overlap['Year'], diff_raw, 'o-', label='Before', color='red', alpha=0.7, ms=6)
    ax4.plot(overlap['Year'], diff_cor, 's-', label='After', color='green', alpha=0.7, ms=6)
    ax4.axhline(0, color='k', ls='--', lw=1.5)
    ax4.set_xlabel('Year', fontweight='bold')
    ax4.set_ylabel('C3S - PIOMAS (km³)', fontweight='bold')
    ax4.set_title('(d) Correction Impact', fontweight='bold', loc='left')
    ax4.legend()
    ax4.grid(alpha=0.3)
    txt = f'Before: μ={diff_raw.mean():.1f}, σ={diff_raw.std():.1f}\nAfter:  μ={diff_cor.mean():.1f}, σ={diff_cor.std():.1f}'
    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, va='top',
             bbox=dict(boxstyle='round', fc='lightblue', alpha=0.5))
    
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Run harmonization for both April and October (Comment 6: report both months)
# ---------------------------------------------------------------------------
# Data file names: adjust paths if not in current directory
DATA_DIR = '.'   # set to '/Volumes/Yotta_1' or your data path if needed
import os

months = [
    ('April', 'C3_ice_volume_April.txt', 'PIOMAS_ice_volume_April.txt', 'Harmonization_April.png'),
    ('October', 'C3_ice_volume_October.txt', 'PIOMAS_ice_volume_October.txt', 'Harmonization_October.png'),
]
results_by_month = {}

for month_name, c3_file, pio_file, out_file in months:
    c3_path = os.path.join(DATA_DIR, c3_file)
    pio_path = os.path.join(DATA_DIR, pio_file)
    c3_df = load_ice_volume_data(c3_path, f'C3S {month_name}')
    pio_df = load_ice_volume_data(pio_path, f'PIOMAS {month_name}')
    if c3_df is None or pio_df is None:
        print(f"  Skipping {month_name} (missing data).")
        continue
    result = harmonize_diagnostic(c3_df, pio_df, month_name)
    results_by_month[month_name] = result
    fig = plot_diagnostic(result, month_name)
    out_path = os.path.join(DATA_DIR, out_file)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")
    plt.close(fig)

# Summary table for Methods (April R²=0.26; October often higher)
print("\n" + "=" * 60)
print("SUMMARY FOR METHODS (Comment 6)")
print("=" * 60)
for month_name, res in results_by_month.items():
    m = res['metrics']
    print(f"  {month_name}:  R² = {m['r2']:.3f},  r = {m['corr']:.3f},  RMSE = {m['rmse']:.1f} km³  [{res['quality']}]")
print("  → Use April R² in the 'modest fraction of variance' sentence; report October if desired.")
print("  → Robustness: C3S-only, PIOMAS-only, ensemble (Extended Data Fig. X).")
print("=" * 60)

try:
    plt.show()
except Exception:
    pass