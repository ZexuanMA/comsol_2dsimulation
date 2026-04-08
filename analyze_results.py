#!/usr/bin/env python3
"""
Analyze COMSOL evanescent wave simulation results and generate plots.
Compares with Song et al., Nat. Commun. 12, 4101 (2021).
"""

import csv, pathlib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE / "results_mean.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── read data (mean across seeds) ──
rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        row = {k: float(v) for k, v in r.items()}
        # map mean columns to T/A/R for compatibility
        row['T'] = row['T_mean']
        row['A'] = row['A_mean']
        row['R'] = row['R_mean']
        rows.append(row)

# ── helpers ──
CASES = {
    0.034: {'za': 114.3, 'label': 'Low  (p=0.034)', 'color': '#2196F3', 'marker': 'o'},
    0.206: {'za': 52.9,  'label': 'Med  (p=0.206)', 'color': '#FF9800', 'marker': 's'},
}
LAMBDA_NM = 365.0
N_Q = 1.46

def penetration_depth(theta_deg, n_ext):
    theta = np.radians(theta_deg)
    arg = N_Q**2 * np.sin(theta)**2 - n_ext**2
    if arg <= 0:
        return np.inf
    return LAMBDA_NM / (4 * np.pi * np.sqrt(arg))

def select(p_val, n_ext_val):
    sub = [r for r in rows if abs(r['p'] - p_val) < 0.001 and abs(r['n_ext'] - n_ext_val) < 0.01]
    sub.sort(key=lambda r: r['theta'])
    return sub

# Paper's analytical model: E_E,dis' ∝ exp(-z_a / Λ) per bounce
def paper_model_TA(theta_arr, p, za, n_ext):
    """Simplified single-bounce evanescent absorption from paper Eq.(2)."""
    result = []
    for th in theta_arr:
        L = penetration_depth(th, n_ext)
        result.append(np.exp(-za / L))
    return np.array(result)


# ================================================================
# Fig 1: T+A vs θ — air and water side by side
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax, n_ext, title in [(ax1, 1.0, 'Air (n$_{ext}$=1.0)'),
                          (ax2, 1.33, 'Water (n$_{ext}$=1.33)')]:
    for p_val, info in CASES.items():
        sub = select(p_val, n_ext)
        if not sub:
            continue
        thetas = [r['theta'] for r in sub]
        ta = [(r['T'] + r['A']) * 100 for r in sub]
        ax.plot(thetas, ta, marker=info['marker'], color=info['color'],
                label=info['label'], markersize=4, linewidth=1.5)

    if n_ext == 1.0:
        ax.axvline(43.2, color='gray', ls=':', alpha=0.5, label=r'$\theta_c$=43.2°')
    else:
        ax.axvline(65.6, color='gray', ls=':', alpha=0.5, label=r'$\theta_c$=65.6°')

    ax.set_xlabel('Incidence angle θ (°)', fontsize=12)
    ax.set_ylabel('T + A (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Single-bounce energy loss (T+A) vs incidence angle — 6-seed mean', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_TA_vs_theta.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 1 saved: fig1_TA_vs_theta.png")


# ================================================================
# Fig 2: T and A separately — 2×2 grid
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (n_ext, title_suffix) in enumerate([(1.0, 'Air'), (1.33, 'Water')]):
    ax_t = axes[0, col]
    ax_a = axes[1, col]
    for p_val, info in CASES.items():
        sub = select(p_val, n_ext)
        if not sub:
            continue
        thetas = [r['theta'] for r in sub]
        ts = [r['T'] * 100 for r in sub]
        als = [r['A'] * 100 for r in sub]
        ax_t.plot(thetas, ts, marker=info['marker'], color=info['color'],
                  label=info['label'], markersize=4, linewidth=1.5)
        ax_a.plot(thetas, als, marker=info['marker'], color=info['color'],
                  label=info['label'], markersize=4, linewidth=1.5)

    ax_t.set_title(f'Transmittance T — {title_suffix}', fontsize=12, fontweight='bold')
    ax_a.set_title(f'Absorptance A — {title_suffix}', fontsize=12, fontweight='bold')
    for ax in (ax_t, ax_a):
        ax.set_xlabel('θ (°)', fontsize=11)
        ax.set_ylabel('%', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

fig.suptitle('Transmittance and Absorptance vs θ — 6-seed mean', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_T_A_separate.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 2 saved: fig2_T_A_separate.png")


# ================================================================
# Fig 3: Water / Air ratio of T+A at overlapping angles (66°–84°)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))

for p_val, info in CASES.items():
    sub_air = {r['theta']: r for r in select(p_val, 1.0)}
    sub_wat = {r['theta']: r for r in select(p_val, 1.33)}
    common = sorted(set(sub_air) & set(sub_wat))
    if not common:
        continue
    ratios = []
    for th in common:
        ta_air = sub_air[th]['T'] + sub_air[th]['A']
        ta_wat = sub_wat[th]['T'] + sub_wat[th]['A']
        ratios.append(ta_wat / ta_air if ta_air > 1e-8 else np.nan)
    ax.plot(common, ratios, marker=info['marker'], color=info['color'],
            label=info['label'], markersize=5, linewidth=1.5)

ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Water = Air')
ax.set_xlabel('θ (°)', fontsize=12)
ax.set_ylabel('(T+A)$_{water}$ / (T+A)$_{air}$', fontsize=12)
ax.set_title('Water-to-Air loss ratio vs θ', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_water_air_ratio.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 3 saved: fig3_water_air_ratio.png")


# ================================================================
# Fig 4: Compare with paper's exp(-za/Λ) model
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, n_ext, title in [(ax1, 1.0, 'Air'), (ax2, 1.33, 'Water')]:
    for p_val, info in CASES.items():
        sub = select(p_val, n_ext)
        if not sub:
            continue
        thetas = np.array([r['theta'] for r in sub])
        ta = np.array([(r['T'] + r['A']) for r in sub])

        # paper model (normalized to match)
        za = CASES[p_val]['za']
        model = paper_model_TA(thetas, p_val, za, n_ext)
        scale = np.mean(ta) / np.mean(model) if np.mean(model) > 0 else 1
        model_scaled = model * scale * 100

        ax.plot(thetas, ta * 100, marker=info['marker'], color=info['color'],
                label=f'{info["label"]} COMSOL', markersize=4, linewidth=1.5)
        ax.plot(thetas, model_scaled, '--', color=info['color'],
                label=f'{info["label"]} exp(-z$_a$/Λ)', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('θ (°)', fontsize=12)
    ax.set_ylabel('T+A (%)', fontsize=12)
    ax.set_title(f'COMSOL vs Paper Model — {title}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_comsol_vs_model.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 4 saved: fig4_comsol_vs_model.png")


# ================================================================
# Fig 5: Penetration depth Λ vs θ
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))
thetas_air = np.linspace(44, 89, 200)
thetas_wat = np.linspace(66, 89, 200)

L_air = [penetration_depth(t, 1.0) for t in thetas_air]
L_wat = [penetration_depth(t, 1.33) for t in thetas_wat]
ax.plot(thetas_air, L_air, 'b-', linewidth=2, label='Air (n$_{ext}$=1.0)')
ax.plot(thetas_wat, L_wat, 'r-', linewidth=2, label='Water (n$_{ext}$=1.33)')

# mark za values
for p_val, info in CASES.items():
    za = CASES[p_val]['za']
    ax.axhline(za, color=info['color'], ls=':', alpha=0.6,
               label=f'z$_a$ = {za} nm ({info["label"][:4]})')

ax.set_xlabel('θ (°)', fontsize=12)
ax.set_ylabel('Λ (nm)', fontsize=12)
ax.set_title('Penetration depth Λ vs incidence angle', fontsize=13, fontweight='bold')
ax.set_ylim(0, 300)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_penetration_depth.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 5 saved: fig5_penetration_depth.png")


# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*60)
print("Summary statistics (6-seed mean)")
print("="*60)

for n_ext, medium in [(1.0, 'Air'), (1.33, 'Water')]:
    print(f"\n--- {medium} (n_ext={n_ext}) ---")
    for p_val, info in CASES.items():
        sub = select(p_val, n_ext)
        if not sub:
            continue
        ta = [(r['T'] + r['A']) * 100 for r in sub]
        t_vals = [r['T'] * 100 for r in sub]
        a_vals = [r['A'] * 100 for r in sub]
        print(f"  {info['label']}:")
        print(f"    T+A:  mean={np.mean(ta):.2f}%  min={np.min(ta):.2f}%  max={np.max(ta):.2f}%")
        print(f"    T:    mean={np.mean(t_vals):.2f}%  A: mean={np.mean(a_vals):.2f}%")
        print(f"    A/T ratio: {np.mean(a_vals)/np.mean(t_vals):.2f}")

# Water/Air ratio (common angles 66-84)
print(f"\n--- Water/Air T+A ratio (θ = 66°–84°) ---")
for p_val, info in CASES.items():
    sub_air = {r['theta']: r for r in select(p_val, 1.0)}
    sub_wat = {r['theta']: r for r in select(p_val, 1.33)}
    common = sorted(set(sub_air) & set(sub_wat))
    if not common:
        continue
    ta_air_avg = np.mean([(sub_air[t]['T'] + sub_air[t]['A']) for t in common])
    ta_wat_avg = np.mean([(sub_wat[t]['T'] + sub_wat[t]['A']) for t in common])
    print(f"  {info['label']}: air={ta_air_avg*100:.2f}%  water={ta_wat_avg*100:.2f}%"
          f"  ratio={ta_wat_avg/ta_air_avg:.2f}x")

# Paper comparison
print(f"\n--- Paper comparison (Low vs Med) ---")
for env, n_ext in [('Air', 1.0), ('Water', 1.33)]:
    means = {}
    for p_val, info in CASES.items():
        sub = select(p_val, n_ext)
        if sub:
            means[info['label'][:4].strip()] = np.mean([(r['T'] + r['A']) for r in sub])
    ranking = sorted(means.items(), key=lambda x: -x[1])
    print(f"  {env}: " + " > ".join(f"{n}({v*100:.1f}%)" for n, v in ranking))

print("\nDone.")
