#!/usr/bin/env python3
"""
Collect COMSOL evanescent-wave energy-balance results into a single CSV.

Reads all  energy_balance_*.txt  files in the current directory (or a given
directory), extracts parameters (p, za, theta, n_ext, seed) and results
(T, A, R), and writes a tidy CSV sorted by (p, n_ext, theta, seed).

Usage
-----
    python collect_results.py                 # scan ./result/
    python collect_results.py /path/to/txts   # scan given dir
    python collect_results.py -o out.csv      # custom output name
"""

import argparse, pathlib, re, sys
import csv

# ---------- parsing helpers ----------

# Filename pattern:
#   TE  →  energy_balance_p_{p}_za_{za}_{theta}_{n_ext}_s{seed}.txt
#   TM  →  energy_balance_p_{p}_za_{za}_{theta}_{n_ext}_TM_s{seed}.txt
# The optional `_TM` marker comes from evanescent_sim.make_tag(pol='TM').
FN_RE = re.compile(
    r"energy_balance_p_(?P<p>[\d.]+)_za_(?P<za>[\d.]+)"
    r"_(?P<theta>[\d.]+)_(?P<n_ext>[\d.]+)"
    r"(?P<polmark>_TM)?"
    r"_s(?P<seed>\d+)\.txt"
)

# Lines inside the file that carry T, A, R
T_RE = re.compile(r"Transmittance\s+T\s*=\s*([\d.eE+-]+)")
A_RE = re.compile(r"Absorptance\s+A\s*=\s*([\d.eE+-]+)")
R_RE = re.compile(r"Reflectance\s+R\s*=\s*([\d.eE+-]+)")


def parse_file(path: pathlib.Path) -> dict | None:
    """Return a dict with all fields, or None on failure."""
    m = FN_RE.match(path.name)
    if not m:
        return None

    g = m.groupdict()
    rec = {
        "p":     float(g["p"]),
        "za":    float(g["za"]),
        "theta": float(g["theta"]),
        "n_ext": float(g["n_ext"]),
        "seed":  int(g["seed"]),
    }

    text = path.read_text()
    t = T_RE.search(text)
    a = A_RE.search(text)
    r = R_RE.search(text)
    if not (t and a and r):
        print(f"  [WARN] incomplete data in {path.name}, skipped", file=sys.stderr)
        return None

    rec["T"] = float(t.group(1))
    rec["A"] = float(a.group(1))
    rec["R"] = float(r.group(1))
    return rec


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Collect energy-balance txt → CSV")
    ap.add_argument("directory", nargs="?", default=".",
                    help="directory containing energy_balance_*.txt (default: current dir)")
    ap.add_argument("-o", "--output", default="results.csv",
                    help="output CSV filename (default: results.csv)")
    args = ap.parse_args()

    src = pathlib.Path(args.directory)
    files = sorted(src.glob("energy_balance_*.txt"))
    if not files:
        print(f"No energy_balance_*.txt files found in {src.resolve()}")
        sys.exit(1)

    rows = []
    for f in files:
        rec = parse_file(f)
        if rec:
            rows.append(rec)

    # sort: p → n_ext → theta → seed
    rows.sort(key=lambda r: (r["p"], r["n_ext"], r["theta"], r["seed"]))

    script_dir = pathlib.Path(__file__).resolve().parent
    out = script_dir / args.output
    fields = ["p", "za", "theta", "n_ext", "seed", "T", "A", "R"]
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Collected {len(rows)} records from {len(files)} files → {out}")

    # quick summary
    ps = sorted(set(r["p"] for r in rows))
    ns = sorted(set(r["n_ext"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    thetas = sorted(set(r["theta"] for r in rows))
    print(f"  p      : {ps}")
    print(f"  n_ext  : {ns}")
    print(f"  theta  : {thetas[0]}–{thetas[-1]} ({len(thetas)} values)")
    print(f"  seeds  : {seeds}")


if __name__ == "__main__":
    main()
