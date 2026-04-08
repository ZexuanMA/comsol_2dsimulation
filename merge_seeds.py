#!/usr/bin/env python3
"""
Merge 6 seed CSVs (seed 13, 29, 53, 37, 41, 67) into a single results_mean.csv.
Only keeps Low (p=0.034) and Med (p=0.206) cases.

Groups by (p, za, theta, n_ext, pol) and computes mean ± std of T, A, R across
seeds. Polarization handling:
  • Old seed CSVs (no `pol` column) are treated as TE for backward compat.
  • New TM data lives alongside TE in the same row keyspace.
  • multi_bounce.py averages TE+TM at the load step (the unpolarized LED case).
"""

import csv, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

SEED_IDS = [13, 29, 53, 37, 41, 67]
KEEP_P = {0.034, 0.206}  # exclude High (0.528)

NUMERIC_FIELDS = {'p', 'za', 'theta', 'n_ext', 'seed', 'T', 'A', 'R'}

# Load all seeds
all_rows = []
for sid in SEED_IDS:
    path = HERE / f"results_seed{sid}.csv"
    if not path.exists():
        print(f"Warning: {path.name} not found, skipping")
        continue
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {k: (float(v) if k in NUMERIC_FIELDS else v)
                   for k, v in r.items()}
            row.setdefault('pol', 'TE')   # backward compat for old CSVs
            if row['p'] in KEEP_P or any(abs(row['p'] - pk) < 0.001 for pk in KEEP_P):
                all_rows.append(row)
    print(f"Loaded {path.name}")

# Group by (p, za, theta, n_ext, pol)
groups = {}
for r in all_rows:
    key = (r['p'], r['za'], r['theta'], r['n_ext'], r['pol'])
    groups.setdefault(key, []).append(r)

# Compute mean and std
out_rows = []
for key, rows in sorted(groups.items()):
    p, za, theta, n_ext, pol = key
    Ts = np.array([r['T'] for r in rows])
    As = np.array([r['A'] for r in rows])
    Rs = np.array([r['R'] for r in rows])
    n_seeds = len(rows)

    out_rows.append({
        'p': p, 'za': za, 'theta': theta, 'n_ext': n_ext, 'pol': pol,
        'n_seeds': n_seeds,
        'T_mean': np.mean(Ts), 'T_std': np.std(Ts, ddof=1) if n_seeds > 1 else 0,
        'A_mean': np.mean(As), 'A_std': np.std(As, ddof=1) if n_seeds > 1 else 0,
        'R_mean': np.mean(Rs), 'R_std': np.std(Rs, ddof=1) if n_seeds > 1 else 0,
    })

# Write CSV
out_path = HERE / "results_mean.csv"
fields = ['p', 'za', 'theta', 'n_ext', 'pol', 'n_seeds',
          'T_mean', 'T_std', 'A_mean', 'A_std', 'R_mean', 'R_std']
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

print(f"\nWrote {len(out_rows)} rows to {out_path.name}")
print(f"  Cases: {sorted(set(r['p'] for r in out_rows))}")
print(f"  Pols : {sorted(set(r['pol'] for r in out_rows))}")
print(f"  Seeds per point: {sorted(set(r['n_seeds'] for r in out_rows))}")

# Quick summary
for p_val in sorted(KEEP_P):
    sub = [r for r in out_rows if abs(r['p'] - p_val) < 0.001]
    for pol_val in sorted(set(r['pol'] for r in sub)):
        sub2 = [r for r in sub if r['pol'] == pol_val]
        n_air = sum(1 for r in sub2 if abs(r['n_ext'] - 1.0) < 0.01)
        n_water = sum(1 for r in sub2 if abs(r['n_ext'] - 1.33) < 0.01)
        print(f"  p={p_val} {pol_val}: {n_air} air + {n_water} water angles")
