#!/usr/bin/env python3
"""
Compare COMSOL results across different random seeds.
Reads results_seed{13,29,37,41,53,67}.csv.
Outputs figures into figures/seed_comparison/.
"""

import csv, pathlib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures" / "seed_comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── load all seed CSVs ──
def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

SEED_IDS = [13, 29, 37, 41, 53, 67]
seed_rows = {}
for sid in SEED_IDS:
    p = HERE / f"results_seed{sid}.csv"
    if p.exists():
        seed_rows[sid] = load_csv(p)
        print(f"seed={sid}: {len(seed_rows[sid])} records")
    else:
        print(f"seed={sid}: file not found, skipping")

# ── helpers ──
CASES = {
    0.034: {'label': 'Low  (p=0.034)', 'color': '#2196F3'},
    0.206: {'label': 'Med  (p=0.206)', 'color': '#FF9800'},
}

SEED_STYLES = {
    13: {'ls': '-',  'marker': 'o', 'label': 'seed=13'},
    29: {'ls': ':',  'marker': 's', 'label': 'seed=29'},
    37: {'ls': '-.', 'marker': 'D', 'label': 'seed=37'},
    41: {'ls': '-',  'marker': 'v', 'label': 'seed=41'},
    53: {'ls': '--', 'marker': '^', 'label': 'seed=53'},
    67: {'ls': ':',  'marker': 'P', 'label': 'seed=67'},
}

def select(rows, p_val, n_ext_val):
    sub = [r for r in rows if abs(r['p'] - p_val) < 0.001 and abs(r['n_ext'] - n_ext_val) < 0.01]
    sub.sort(key=lambda r: r['theta'])
    return sub


# ================================================================
# Fig 1: T+A vs θ — seed=13 vs seed=53, 2 panels (air / water)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for ax, n_ext, title in [(ax1, 1.0, 'Air (n$_{ext}$=1.0)'),
                          (ax2, 1.33, 'Water (n$_{ext}$=1.33)')]:
    for p_val, cinfo in CASES.items():
        for sid, srows in seed_rows.items():
            sub = select(srows, p_val, n_ext)
            if not sub:
                continue
            thetas = [r['theta'] for r in sub]
            ta = [(r['T'] + r['A']) * 100 for r in sub]
            sty = SEED_STYLES[sid]
            lbl = f"{cinfo['label'][:4]} s{sid}"
            ax.plot(thetas, ta, ls=sty['ls'], marker=sty['marker'],
                    color=cinfo['color'], label=lbl, markersize=3, linewidth=1.3, alpha=0.85)

    ax.set_xlabel('θ (°)', fontsize=12)
    ax.set_ylabel('T+A (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

seed_desc = ', '.join(f's{s}' for s in seed_rows)
fig.suptitle(f'T+A vs θ — {seed_desc}', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_TA_seed_compare.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 1 saved: fig1_TA_seed_compare.png")


# ================================================================
# Fig 2: T and A separately — 2 rows (Low/Med) × 2 cols (air/water)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (p_val, cinfo) in enumerate(CASES.items()):
    for col, (n_ext, medium) in enumerate([(1.0, 'Air'), (1.33, 'Water')]):
        ax = axes[row, col]
        for sid, srows in seed_rows.items():
            sub = select(srows, p_val, n_ext)
            if not sub:
                continue
            thetas = [r['theta'] for r in sub]
            ts = [r['T'] * 100 for r in sub]
            als = [r['A'] * 100 for r in sub]
            sty = SEED_STYLES[sid]
            ax.plot(thetas, ts, ls=sty['ls'], marker=sty['marker'],
                    color='#1976D2', label=f'T s{sid}', markersize=3, linewidth=1.2, alpha=0.8)
            ax.plot(thetas, als, ls=sty['ls'], marker=sty['marker'],
                    color='#D32F2F', label=f'A s{sid}', markersize=3, linewidth=1.2, alpha=0.8)

        ax.set_title(f'{cinfo["label"]} — {medium}', fontsize=11, fontweight='bold')
        ax.set_xlabel('θ (°)', fontsize=10)
        ax.set_ylabel('%', fontsize=10)
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)

fig.suptitle('T (blue) and A (red) — all seeds', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_TA_detail_seed_compare.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 2 saved: fig2_TA_detail_seed_compare.png")


# ================================================================
# Fig 3: Spread across seeds — (max - min) / mean for T+A
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, n_ext, title in [(ax1, 1.0, 'Air'), (ax2, 1.33, 'Water')]:
    for p_val, cinfo in CASES.items():
        # collect data per theta from all seeds
        by_theta = {}
        for sid, srows in seed_rows.items():
            for r in select(srows, p_val, n_ext):
                th = r['theta']
                by_theta.setdefault(th, []).append(r['T'] + r['A'])
        # only keep thetas present in at least 2 seeds
        thetas_common = sorted(th for th, vals in by_theta.items() if len(vals) >= 2)
        if not thetas_common:
            continue
        spread = []
        for th in thetas_common:
            vals = np.array(by_theta[th])
            mean_val = np.mean(vals)
            if mean_val > 1e-8:
                spread.append((np.max(vals) - np.min(vals)) / mean_val * 100)
            else:
                spread.append(0)
        ax.plot(thetas_common, spread, marker='o', color=cinfo['color'],
                label=cinfo['label'], markersize=3, linewidth=1.3)

    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('θ (°)', fontsize=12)
    ax.set_ylabel('Spread (max−min)/mean (%)', fontsize=12)
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Seed-to-seed spread in T+A', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_relative_diff.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 3 saved: fig3_relative_diff.png")


# ================================================================
# Fig 4: Scatter — each seed vs seed-mean (correlation)
# ================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

from matplotlib.lines import Line2D

SEED_MARKERS = {13: 'o', 29: 's', 37: 'D', 41: 'v', 53: '^', 67: 'P'}

for ax, qty, qlabel in [(ax1, 'T', 'Transmittance T'),
                          (ax2, 'A', 'Absorptance A'),
                          (ax3, lambda r: r['T']+r['A'], 'T+A')]:
    for p_val, cinfo in CASES.items():
        for n_ext in [1.0, 1.33]:
            # build {theta: {seed: value}}
            by_theta = {}
            for sid, srows in seed_rows.items():
                for r in select(srows, p_val, n_ext):
                    th = r['theta']
                    val = qty(r) * 100 if callable(qty) else r[qty] * 100
                    by_theta.setdefault(th, {})[sid] = val
            # only thetas with all seeds
            common = sorted(th for th, d in by_theta.items() if len(d) >= 2)
            if not common:
                continue
            for sid in seed_rows:
                means = []
                vals = []
                for th in common:
                    if sid not in by_theta[th]:
                        continue
                    others = [v for s, v in by_theta[th].items() if s != sid]
                    if not others:
                        continue
                    means.append(np.mean(list(by_theta[th].values())))
                    vals.append(by_theta[th][sid])
                if vals:
                    ax.scatter(means, vals, c=cinfo['color'],
                               marker=SEED_MARKERS.get(sid, 'o'), s=12, alpha=0.5)

    lims = ax.get_xlim()
    ax.plot([0, max(lims[1], 35)], [0, max(lims[1], 35)], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel(f'{qlabel} mean (%)', fontsize=11)
    ax.set_ylabel(f'{qlabel} per seed (%)', fontsize=11)
    ax.set_title(qlabel, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# custom legend
handles = [Line2D([0],[0], marker='o', color=c['color'], ls='', markersize=6, label=c['label'][:4])
           for c in CASES.values()]
for sid, mk in SEED_MARKERS.items():
    if sid in seed_rows:
        handles.append(Line2D([0],[0], marker=mk, color='gray', ls='', markersize=6, label=f's{sid}'))
fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.06))
fig.suptitle('Seed-to-mean correlation (each point = one angle)', fontsize=14, y=1.1)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_scatter_correlation.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig 4 saved: fig4_scatter_correlation.png")


# ================================================================
# Statistics
# ================================================================
print("\n" + "="*60)
print("Seed comparison statistics")
print("="*60)

for n_ext, medium in [(1.0, 'Air'), (1.33, 'Water')]:
    print(f"\n--- {medium} ---")
    for p_val, cinfo in CASES.items():
        # collect per-theta from all seeds
        by_theta = {}
        for sid, srows in seed_rows.items():
            for r in select(srows, p_val, n_ext):
                th = r['theta']
                by_theta.setdefault(th, {})[sid] = r['T'] + r['A']

        common = sorted(th for th, d in by_theta.items() if len(d) >= 2)
        if not common:
            continue

        # per-seed means
        seed_means = {}
        for sid in seed_rows:
            vals = [by_theta[th][sid] for th in common if sid in by_theta[th]]
            if vals:
                seed_means[sid] = np.mean(vals) * 100

        # spread stats
        spreads = []
        for th in common:
            vals = np.array(list(by_theta[th].values()))
            mean_val = np.mean(vals)
            if mean_val > 1e-8:
                spreads.append((np.max(vals) - np.min(vals)) / mean_val * 100)
        spreads = np.array(spreads)

        print(f"  {cinfo['label']}:")
        mean_str = '  '.join(f's{s}={m:.2f}%' for s, m in sorted(seed_means.items()))
        print(f"    Mean T+A:  {mean_str}")
        print(f"    Spread (max-min)/mean:  mean={np.mean(spreads):.1f}%  max={np.max(spreads):.1f}%")

        # pairwise correlations
        sids = sorted(seed_rows.keys())
        for i in range(len(sids)):
            for j in range(i+1, len(sids)):
                s1, s2 = sids[i], sids[j]
                common_pair = [th for th in common if s1 in by_theta[th] and s2 in by_theta[th]]
                if len(common_pair) < 2:
                    continue
                v1 = np.array([by_theta[th][s1] for th in common_pair])
                v2 = np.array([by_theta[th][s2] for th in common_pair])
                corr = np.corrcoef(v1, v2)[0, 1]
                print(f"    Corr s{s1}-s{s2}: r={corr:.4f}")

print("\nDone.")
