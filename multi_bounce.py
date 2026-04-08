#!/usr/bin/env python3
"""
Multi-bounce energy balance model for TiO₂-coated optical fibers.

Implements two approaches:
  1. COMSOL-based: uses our simulated T(θ), A(θ) per single bounce
  2. Paper analytical: uses Eqs. (3)-(6) from Song et al., Nat. Commun. 12, 4101 (2021)

Compares cumulative E_dis with experimental data (Fig. 3a).

Parameters (from paper Methods + Supplementary Note 5A):
  - Fiber: FT1000UMT (Thorlabs), d = 1 mm core diameter
  - Coating length: L = 6.5 cm
  - θ range: 0.376π to 0.495π (67.68° to 89.1°), step 0.0001π
  - Light intensity: E₀ = 7.02 mW/cm²
  - Irradiation: 4 h
  - λ = 365 nm, n_q = 1.46, n_T = 2.5 - 0.018i
"""

import csv, pathlib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures" / "multi_bounce"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Physical parameters
# ══════════════════════════════════════════════════════════════
LAMBDA_NM = 365.0
N_Q = 1.46
N_T = 2.5           # TiO₂ real part
D_CM = 0.1           # fiber diameter = 1 mm = 0.1 cm
L_CM = 6.5           # coating length
E0_MW_CM2 = 7.02     # incident irradiance mW/cm²
T_HOURS = 4.0        # irradiation time

CASES = {
    'Low':  {'p': 0.034, 'za_nm': 114.3},
    'Med':  {'p': 0.206, 'za_nm': 52.9},
}

# Paper's θ grid: 0.376π to 0.495π, step 0.0001π → ~1190 angles
THETA_MIN = 0.376 * np.pi  # rad
THETA_MAX = 0.495 * np.pi  # rad
DTHETA = 0.0001 * np.pi    # rad
THETAS_RAD = np.arange(THETA_MIN, THETA_MAX + DTHETA/2, DTHETA)
THETAS_DEG = np.degrees(THETAS_RAD)
N_ANGLES = len(THETAS_RAD)

# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════

def penetration_depth(theta_rad, n_ext):
    """Λ in nm — Eq. (1)."""
    arg = N_Q**2 * np.sin(theta_rad)**2 - n_ext**2
    arg = np.maximum(arg, 1e-30)
    return LAMBDA_NM / (4 * np.pi * np.sqrt(arg))


def fresnel_T_quartz_tio2(theta_rad):
    """
    Fresnel transmission T at quartz/TiO₂ interface — Eq. (3).
    TE + TM averaged. Used for the refracted-light channel (Mode 1/2).
    """
    cos_i = np.cos(theta_rad)
    sin_i = np.sin(theta_rad)
    ratio = N_Q / N_T
    sin_t = ratio * sin_i
    # For angles where sin_t > 1, total internal reflection → T = 0
    mask = sin_t < 1.0
    T = np.zeros_like(theta_rad)

    cos_t = np.sqrt(np.maximum(1 - sin_t**2, 0))
    # TE (s-pol)
    rs = (N_Q * cos_i - N_T * cos_t) / (N_Q * cos_i + N_T * cos_t)
    # TM (p-pol)
    rp = (N_T * cos_i - N_Q * cos_t) / (N_T * cos_i + N_Q * cos_t)
    T_fresnel = 1.0 - 0.5 * (rs**2 + rp**2)
    T[mask] = T_fresnel[mask]
    return T


def n_bounces(theta_rad):
    """Number of TIR bounces in coating length L — N = L / (d · tanθ)."""
    return L_CM / (D_CM * np.tan(theta_rad))


# ══════════════════════════════════════════════════════════════
# Approach 1: Paper's analytical model — Eqs. (4)-(6)
# ══════════════════════════════════════════════════════════════

def paper_model(n_ext):
    """
    Compute E_dis per angle using the paper's analytical Eqs. (4)-(6).
    Returns dict with E_E_dis, E_R_dis, E_dis per case.
    """
    results = {}
    T_fresnel = fresnel_T_quartz_tio2(THETAS_RAD)
    Lambda = penetration_depth(THETAS_RAD, n_ext)
    N_b = n_bounces(THETAS_RAD)

    for name, case in CASES.items():
        p = case['p']
        za = case['za_nm']

        exp_za_L = np.exp(-za / Lambda)   # exp(-z_a / Λ)

        # Per-bounce survival: fraction of light NOT lost per bounce
        # From Eqs (4)-(5): the denominator term
        surv = (1 - p) * (1 - exp_za_L) + p * (1 - T_fresnel)
        # This is the fraction lost per bounce; survival = 1 - surv
        # But the paper writes it differently. Let me re-derive from Eqs (4)-(5).
        #
        # Eq (4): E_E,dis' = E₀ × [(1-p)·exp(-za/Λ) × {1 - [(1-p)(1-exp(-za/Λ)) + p(1-T)]^N}]
        #                         / [(1-p)·exp(-za/Λ) + p·T]
        #
        # Eq (5): E_R,dis' = E₀ × [p·T × {1 - [(1-p)(1-exp(-za/Λ)) + p(1-T)]^N}]
        #                         / [(1-p)·exp(-za/Λ) + p·T]

        # survival fraction per bounce (what's in the brackets raised to N):
        q = (1 - p) * (1 - exp_za_L) + p * (1 - T_fresnel)
        # But wait, this should be < 1 for it to make sense as survival^N.
        # Actually re-reading: the expression inside [...]^{L/(d tanθ)} is the
        # survival per bounce. Let me be more careful.
        #
        # The paper writes:
        #   [(1-p)(1-e^{-za/Λ}) + p(1-T)]^{L/(d tanθ)}
        # This is the fraction of light that SURVIVES after N bounces.
        # Wait no — (1-p)(1-e^{-za/Λ}) is evanescent loss fraction on non-contact area,
        # and p(1-T) is refraction loss on contact area. So their sum is total loss per bounce.
        # Then survival = 1 - loss... but actually let me re-check.
        #
        # Actually, I think the paper's notation is:
        #   The term inside []^N represents what fraction of each ray's energy
        #   is NOT dissipated and continues to propagate. So it should be the
        #   survival fraction.
        #
        # For that to work, we need:
        #   survival = 1 - (fraction lost per bounce)
        # where fraction lost = p*T_fresnel (refracted into TiO₂ and lost)
        #                     + (1-p)*exp(-za/Λ) ... wait, that doesn't add up either.
        #
        # Let me just carefully transcribe the paper's formulas:
        #
        # The quantity raised to power N is:
        #   (1-p)(1-e^{-za/Λ}) + p(1-T)
        #
        # Hmm, (1-p)(1-e^{-za/Λ}) → for non-contact area, the fraction that the
        # evanescent wave deposits into the coating.
        # p(1-T) → for contact area, the fraction reflected back at quartz/TiO₂.
        #
        # Wait, I think this IS the survival fraction. Let me think again...
        # When light hits non-contact area: evanescent wave is generated.
        #   - Fraction exp(-za/Λ) reaches the TiO₂ coating → absorbed/dissipated
        #   - Fraction (1-exp(-za/Λ)) returns to fiber → survives? No...
        #   - Actually exp(-za/Λ) is how much reaches TiO₂, so dissipated = exp(-za/Λ),
        #     survived = 1-exp(-za/Λ)? That doesn't make physical sense either.
        #
        # Let me re-read the paper more carefully. Eq (2) says:
        #   E_{E,dis} = E_i × e^{-z_n/Λ}
        # This is the evanescent energy dissipated — it's the energy that REACHES the
        # TiO₂ layer at distance z_n. So the fraction dissipated is e^{-za/Λ}.
        # Wait, that's also weird — exp(-za/Λ) → when za is large, exp is small → less
        # dissipation. That makes sense: if particles are far away, less evanescent field
        # reaches them.
        #
        # So for non-contact area (fraction 1-p of the surface):
        #   dissipated fraction = exp(-za/Λ)  [evanescent field reaching TiO₂]
        #   survived fraction = 1 - exp(-za/Λ)  [returned to fiber]
        #
        # For contact area (fraction p of the surface):
        #   dissipated fraction = T_fresnel [refracted into TiO₂]
        #   survived fraction = 1 - T_fresnel [reflected back]
        #
        # Total survival per bounce:
        #   S = (1-p) × [1 - exp(-za/Λ)] + p × (1-T)
        #     = (1-p)(1-exp(-za/Λ)) + p(1-T)
        #
        # YES! This matches the expression in the paper raised to power N.
        # So S^N = fraction surviving after N bounces.
        # Total dissipated = 1 - S^N.
        #
        # The paper then splits the dissipated energy into evanescent vs refracted:
        #   E_E,dis' = E₀ × [(1-p)exp(-za/Λ) / ((1-p)exp(-za/Λ) + pT)] × (1 - S^N)
        #   E_R,dis' = E₀ × [pT / ((1-p)exp(-za/Λ) + pT)] × (1 - S^N)

        S = (1 - p) * (1 - exp_za_L) + p * (1 - T_fresnel)  # survival per bounce
        S_N = S ** N_b                                          # survival after N bounces
        total_diss = 1 - S_N                                   # total dissipated fraction

        denom = (1 - p) * exp_za_L + p * T_fresnel
        frac_evan = (1 - p) * exp_za_L / denom    # fraction of loss going to evanescent
        frac_refr = p * T_fresnel / denom          # fraction of loss going to refraction

        E_E = total_diss * frac_evan   # evanescent dissipation fraction per angle
        E_R = total_diss * frac_refr   # refracted dissipation fraction per angle

        results[name] = {
            'S': S, 'N_bounce': N_b,
            'total_diss': total_diss,
            'E_E': E_E, 'E_R': E_R,
            'E_dis': E_E + E_R,   # = total_diss (sanity check)
        }

    return results


# ══════════════════════════════════════════════════════════════
# Approach 2: COMSOL-based multi-bounce
# ══════════════════════════════════════════════════════════════

def load_comsol_data():
    """Load results_mean.csv and average over polarization.

    The CSV may contain TE only (legacy data) or both TE and TM (after Phase
    1.2). Real LED light is unpolarized, so when both polarizations are
    present at a given (p, n_ext, θ) we average them. The returned dict has
    one entry per (p, n_ext, θ); each entry stores T, A, plus a `pols` list
    indicating which polarizations contributed to that average.
    """
    path = HERE / "results_mean.csv"
    by_pol = {}  # {(p, n_ext, θ): {pol: {T, A}}}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (float(r['p']), float(r['n_ext']), float(r['theta']))
            pol = r.get('pol', 'TE') or 'TE'
            by_pol.setdefault(key, {})[pol] = {
                'T': float(r['T_mean']),
                'A': float(r['A_mean']),
            }

    averaged = {}
    pol_combos = {}
    for key, pols in by_pol.items():
        Ts = [v['T'] for v in pols.values()]
        As = [v['A'] for v in pols.values()]
        averaged[key] = {
            'T': sum(Ts) / len(Ts),
            'A': sum(As) / len(As),
            'pols': sorted(pols.keys()),
        }
        combo = "+".join(sorted(pols.keys()))
        pol_combos[combo] = pol_combos.get(combo, 0) + 1

    summary = ", ".join(f"{c}: {n}" for c, n in sorted(pol_combos.items()))
    print(f"  Polarization coverage: {summary}")
    return averaged


def comsol_multi_bounce(n_ext, comsol_data):
    """
    Use COMSOL T(θ), A(θ) interpolated onto the paper's θ grid.
    R(θ) = 1 - T - A, then survival^N_bounce.
    """
    results = {}
    N_b = n_bounces(THETAS_RAD)

    for name, case in CASES.items():
        p = case['p']

        # Get COMSOL data points for this case
        comsol_thetas = []
        comsol_T = []
        comsol_A = []
        for (p_key, nex, th), vals in comsol_data.items():
            if abs(p_key - p) < 0.001 and abs(nex - n_ext) < 0.01:
                comsol_thetas.append(th)
                comsol_T.append(vals['T'])
                comsol_A.append(vals['A'])

        if not comsol_thetas:
            continue

        idx = np.argsort(comsol_thetas)
        ct = np.array(comsol_thetas)[idx]
        cT = np.array(comsol_T)[idx]
        cA = np.array(comsol_A)[idx]

        # Interpolate onto paper's fine θ grid
        T_interp = np.interp(THETAS_DEG, ct, cT, left=cT[0], right=cT[-1])
        A_interp = np.interp(THETAS_DEG, ct, cA, left=cA[0], right=cA[-1])
        R_interp = 1.0 - T_interp - A_interp

        # Multi-bounce: survival after N bounces
        R_interp = np.clip(R_interp, 0, 1)
        S_N = R_interp ** N_b
        total_diss = 1 - S_N

        # Split into T and A proportionally
        ta_sum = T_interp + A_interp
        ta_sum = np.maximum(ta_sum, 1e-15)
        frac_T = T_interp / ta_sum
        frac_A = A_interp / ta_sum

        results[name] = {
            'R': R_interp, 'N_bounce': N_b,
            'total_diss': total_diss,
            'diss_T': total_diss * frac_T,
            'diss_A': total_diss * frac_A,
            'comsol_thetas': ct,
            'comsol_TA': cT + cA,
        }

    return results


# ══════════════════════════════════════════════════════════════
# Compute E_dis in Joules (to compare with Fig. 3a)
# ══════════════════════════════════════════════════════════════

## ─────────────────────────────────────────────────────────────
## LED angular weighting models
## ─────────────────────────────────────────────────────────────
##
## Each model defines w(θ), the fraction of total input energy that arrives
## at incidence angle θ. After normalization Σ_θ w(θ) = 1.
##
## Physics rationale:
##   • uniform   — paper's implicit assumption: equal energy per angle bin.
##   • cos       — full mode mixing in a multimode fiber (NA=0.50): rays
##                 uniformly distributed in solid angle inside the cone, so
##                 dN/dα ∝ sin(α) = cos(θ), where θ is incidence on the wall
##                 and α = 90° − θ is angle from the fiber axis. The same
##                 result follows from a Lambertian LED face-coupled to the
##                 fiber, to leading order in α.
##   • sin2     — w(θ) ∝ sin(2θ) ∝ sin(θ)cos(θ): a stronger Lambertian-like
##                 model that drops faster as θ → 90°. Useful for sensitivity.
##
## All three weightings keep the same total input energy (the irradiance × area
## × time product); they only redistribute it across θ.

WEIGHT_MODELS = {
    'uniform': lambda th_rad: np.ones_like(th_rad),
    'cos':     lambda th_rad: np.cos(th_rad),
    'sin2':    lambda th_rad: np.sin(2 * th_rad),
}


def angular_weights(model_name):
    """Return normalized weights w(θ) on the THETAS_RAD grid (sums to 1)."""
    w = WEIGHT_MODELS[model_name](THETAS_RAD)
    w = np.clip(w, 0.0, None)
    s = np.sum(w)
    if s <= 0:
        raise ValueError(f"Weight model {model_name!r} normalizes to zero")
    return w / s


def compute_E_dis_joules(diss_fractions, weight_model='uniform'):
    """
    Convert per-angle dissipation fractions to total E_dis in Joules.

    Total input energy is fixed (irradiance × area × time). The angular
    weighting only changes how that energy is distributed across θ:

        E_dis = total_energy_J × Σ_θ [ w(θ) · diss(θ) ]

    where w(θ) is normalized so Σ_θ w(θ) = 1. The default 'uniform' model
    reproduces the paper's implicit assumption of equal energy per angle bin.

    Total input energy = irradiance × area × time
      = 7.02 mW/cm² × π(d/2)² × 4h
      = 7.02e-3 W/cm² × π(0.05)² cm² × 14400 s
    """
    area = np.pi * (D_CM / 2)**2  # cm²
    total_energy_J = E0_MW_CM2 * 1e-3 * area * T_HOURS * 3600  # Joules

    w = angular_weights(weight_model)
    E_dis = total_energy_J * np.sum(w * diss_fractions)
    return E_dis, total_energy_J


# ══════════════════════════════════════════════════════════════
# Main computation and plotting
# ══════════════════════════════════════════════════════════════

COLORS = {'Low': '#2196F3', 'Med': '#FF9800'}

comsol_data = load_comsol_data()
print(f"COMSOL data: {len(comsol_data)} unique (p, n_ext, θ) points")
print(f"Paper θ grid: {N_ANGLES} angles, {np.degrees(THETA_MIN):.1f}°–{np.degrees(THETA_MAX):.1f}°")
print(f"Fiber: d={D_CM*10:.0f} mm, L={L_CM} cm")
print(f"Bounces: {n_bounces(THETA_MIN):.0f} (θ_min) to {n_bounces(THETA_MAX):.0f} (θ_max)")

for n_ext, medium in [(1.0, 'Air'), (1.33, 'Water')]:
    print(f"\n{'='*60}")
    print(f"  {medium} (n_ext = {n_ext})")
    print(f"{'='*60}")

    paper_res = paper_model(n_ext)
    comsol_res = comsol_multi_bounce(n_ext, comsol_data)

    # ── Print E_dis comparison ──
    print(f"\n  {'Case':<6} | {'Paper E_dis (J)':>15} | {'COMSOL E_dis (J)':>16} | {'Paper Fig3a (J)':>15}")
    print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*16}-+-{'-'*15}")

    # Experimental values from Fig. 3a (approximate)
    exp_vals = {
        1.0:  {'Med': 0.63, 'Low': 0.33},
        1.33: {'Med': 0.73, 'Low': 0.66},
    }

    for name in ['Med', 'Low']:
        # Paper analytical
        if name in paper_res:
            e_paper, total_E = compute_E_dis_joules(paper_res[name]['total_diss'])
        else:
            e_paper = 0

        # COMSOL-based
        if name in comsol_res:
            e_comsol, _ = compute_E_dis_joules(comsol_res[name]['total_diss'])
        else:
            e_comsol = 0

        e_exp = exp_vals.get(n_ext, {}).get(name, 0)
        print(f"  {name:<6} | {e_paper:>15.4f} | {e_comsol:>16.4f} | {e_exp:>15.2f}")

    print(f"\n  Total input energy: {total_E:.4f} J")

    # ────────────────────────────────────────────────────────
    # Fig: Dissipated fraction vs θ
    # ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    for name in ['Med', 'Low']:
        c = COLORS[name]

        if name in paper_res:
            ax1.plot(THETAS_DEG, paper_res[name]['total_diss'] * 100,
                     color=c, linewidth=1.5, label=name)

        if name in comsol_res:
            ax2.plot(THETAS_DEG, comsol_res[name]['total_diss'] * 100,
                     color=c, linewidth=1.5, label=name)

    for ax, title in [(ax1, f'Paper analytical model — {medium}'),
                       (ax2, f'COMSOL-based model — {medium}')]:
        ax.set_xlabel('θ (°)', fontsize=12)
        ax.set_ylabel('Cumulative dissipation (%)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    fig.suptitle(f'Multi-bounce cumulative dissipation — {medium}', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig1_diss_vs_theta_{medium.lower()}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig saved: fig1_diss_vs_theta_{medium.lower()}.png")

    # ────────────────────────────────────────────────────────
    # Fig: N_bounces and single-bounce survival
    # ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    N_b = n_bounces(THETAS_RAD)
    ax1.plot(THETAS_DEG, N_b, 'k-', linewidth=2)
    ax1.set_xlabel('θ (°)', fontsize=12)
    ax1.set_ylabel('N bounces', fontsize=12)
    ax1.set_title(f'Number of bounces (d={D_CM*10:.0f}mm, L={L_CM}cm)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    for name in ['Med', 'Low']:
        if name in paper_res:
            ax2.plot(THETAS_DEG, paper_res[name]['S'] * 100,
                     color=COLORS[name], linewidth=1.5, label=f'{name} (paper)')
        if name in comsol_res:
            ax2.plot(THETAS_DEG, comsol_res[name]['R'] * 100,
                     '--', color=COLORS[name], linewidth=1.5, label=f'{name} (COMSOL)', alpha=0.8)

    ax2.set_xlabel('θ (°)', fontsize=12)
    ax2.set_ylabel('Survival per bounce (%)', fontsize=12)
    ax2.set_title(f'Single-bounce survival R(θ) — {medium}', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig2_bounces_survival_{medium.lower()}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig saved: fig2_bounces_survival_{medium.lower()}.png")


# ══════════════════════════════════════════════════════════════
# Fig: Bar chart comparing E_dis — Paper vs COMSOL vs Experiment
# ══════════════════════════════════════════════════════════════
exp_vals = {
    1.0:  {'Med': 0.63, 'Low': 0.33},
    1.33: {'Med': 0.73, 'Low': 0.66},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, n_ext, medium in [(ax1, 1.0, 'Air'), (ax2, 1.33, 'Water')]:
    paper_res = paper_model(n_ext)
    comsol_res = comsol_multi_bounce(n_ext, comsol_data)

    names = ['Med', 'Low']
    x = np.arange(len(names))
    width = 0.25

    e_exp = [exp_vals[n_ext][n] for n in names]
    e_paper = [compute_E_dis_joules(paper_res[n]['total_diss'])[0] for n in names]
    e_comsol = [compute_E_dis_joules(comsol_res[n]['total_diss'])[0] if n in comsol_res else 0
                for n in names]

    bars1 = ax.bar(x - width, e_exp, width, label='Experiment (Fig.3a)', color='#333333', alpha=0.8)
    bars2 = ax.bar(x, e_paper, width, label='Paper model (Eq.4-6)', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x + width, e_comsol, width, label='COMSOL multi-bounce', color='#2196F3', alpha=0.8)

    # value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('E$_{dis}$ (J)', fontsize=12)
    ax.set_title(f'{medium}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Cumulative radiant energy dissipation E$_{dis}$ — comparison',
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_Edis_bar_comparison.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("\nFig saved: fig3_Edis_bar_comparison.png")


# ══════════════════════════════════════════════════════════════
# Phase 4.2 — LED angular weighting comparison
# ══════════════════════════════════════════════════════════════
# COMSOL is systematically -4% to -15% below experiment (worst on Low). One
# possible cause is that the multi-bounce model assumes uniform energy per θ
# bin, while a real LED-coupled multimode fiber concentrates power closer to
# the axis (lower θ). Below we re-compute E_dis using:
#   • uniform   — current paper-style weighting
#   • cos       — full mode mixing or Lambertian-coupled fiber: w(θ) ∝ cos(θ)
#   • sin2      — stronger Lambertian-like model:        w(θ) ∝ sin(2θ)
# and compare against experiment.

WEIGHT_LABELS = {
    'uniform': 'Uniform (paper)',
    'cos':     'cos(θ)  [mode-mixed / Lambertian]',
    'sin2':    'sin(2θ) [strong Lambertian]',
}
WEIGHT_BAR_COLORS = {
    'uniform': '#90A4AE',
    'cos':     '#2196F3',
    'sin2':    '#9C27B0',
}

print("\n" + "═" * 78)
print("  Phase 4.2 — LED angular weighting comparison")
print("═" * 78)

weighted_table = {}  # {n_ext: {case: {model: (paper_J, comsol_J)}}}

for n_ext, medium in [(1.0, 'Air'), (1.33, 'Water')]:
    paper_res  = paper_model(n_ext)
    comsol_res = comsol_multi_bounce(n_ext, comsol_data)

    print(f"\n  {medium} (n_ext = {n_ext})")
    header = (f"    {'Case':<5} | {'Model':<32} | {'Paper E (J)':>11} | "
              f"{'COMSOL E (J)':>12} | {'Exp (J)':>8}")
    print(header)
    print("    " + "-" * (len(header) - 4))

    weighted_table[n_ext] = {}
    for name in ['Med', 'Low']:
        weighted_table[n_ext][name] = {}
        e_exp = exp_vals[n_ext][name]
        for model in WEIGHT_MODELS:
            e_paper, _  = compute_E_dis_joules(paper_res[name]['total_diss'],
                                               weight_model=model)
            e_comsol, _ = (compute_E_dis_joules(comsol_res[name]['total_diss'],
                                                weight_model=model)
                           if name in comsol_res else (0.0, 0.0))
            weighted_table[n_ext][name][model] = (e_paper, e_comsol)
            print(f"    {name:<5} | {WEIGHT_LABELS[model]:<32} | "
                  f"{e_paper:>11.4f} | {e_comsol:>12.4f} | {e_exp:>8.2f}")
        print("    " + "-" * (len(header) - 4))

# ─────────────────────────────────────────────────────────────
# Bar chart: E_dis vs experiment under each weighting model
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (n_ext, medium) in enumerate([(1.0, 'Air'), (1.33, 'Water')]):
    for row, source in enumerate(['paper', 'comsol']):
        ax = axes[row, col]
        names = ['Med', 'Low']
        x = np.arange(len(names))

        models = list(WEIGHT_MODELS.keys())
        n_models = len(models)
        width = 0.8 / (n_models + 1)  # +1 for the experiment bar

        # experiment bar (always first)
        e_exp = [exp_vals[n_ext][n] for n in names]
        ax.bar(x - width * (n_models / 2), e_exp, width,
               label='Experiment', color='#333333', alpha=0.85)

        # one bar per weighting model
        src_idx = 0 if source == 'paper' else 1
        for k, model in enumerate(models):
            heights = [weighted_table[n_ext][n][model][src_idx] for n in names]
            offset = (k - n_models / 2 + 1) * width
            bars = ax.bar(x + offset, heights, width,
                          label=WEIGHT_LABELS[model],
                          color=WEIGHT_BAR_COLORS[model], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

        # also annotate experiment bars
        for xi, ev in zip(x - width * (n_models / 2), e_exp):
            ax.text(xi, ev + 0.01, f'{ev:.2f}',
                    ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel('E$_{dis}$ (J)', fontsize=11)
        ax.set_title(f'{source.capitalize()} model — {medium}',
                     fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.2, axis='y')
        if row == 0 and col == 1:
            ax.legend(fontsize=8, loc='upper right')

fig.suptitle('Phase 4.2 — E$_{dis}$ under different LED angular weightings',
             fontsize=14, y=1.00)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_Edis_weighted.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("\nFig saved: fig5_Edis_weighted.png")

# ─────────────────────────────────────────────────────────────
# Plot the weighting functions themselves for transparency
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for model in WEIGHT_MODELS:
    w = angular_weights(model)
    ax.plot(THETAS_DEG, w / w.max(), linewidth=2,
            label=WEIGHT_LABELS[model], color=WEIGHT_BAR_COLORS[model])
ax.set_xlabel('θ (°)', fontsize=12)
ax.set_ylabel('w(θ)  (peak-normalized)', fontsize=12)
ax.set_title('LED angular weighting models on the paper θ grid',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(THETAS_DEG[0], THETAS_DEG[-1])
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_weight_models.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig saved: fig6_weight_models.png")


# ══════════════════════════════════════════════════════════════
# Fig: E_dis vs coating length L (like Fig. 5c)
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

L_range = np.linspace(0.5, 30, 200)  # cm

for ax, n_ext, medium in [(ax1, 1.0, 'Air'), (ax2, 1.33, 'Water')]:
    Lambda = penetration_depth(THETAS_RAD, n_ext)
    T_fresnel = fresnel_T_quartz_tio2(THETAS_RAD)
    comsol_res_base = comsol_multi_bounce(n_ext, comsol_data)

    for name in ['Med', 'Low']:
        case = CASES[name]
        p, za = case['p'], case['za_nm']
        exp_za_L = np.exp(-za / Lambda)
        S_paper = (1 - p) * (1 - exp_za_L) + p * (1 - T_fresnel)

        e_paper_L = []
        e_comsol_L = []
        for L in L_range:
            N_b = L / (D_CM * np.tan(THETAS_RAD))

            # Paper model
            S_N = S_paper ** N_b
            diss = 1 - S_N
            e, _ = compute_E_dis_joules(diss)
            e_paper_L.append(e)

            # COMSOL model
            if name in comsol_res_base:
                R_c = comsol_res_base[name]['R']
                S_N_c = R_c ** N_b
                diss_c = 1 - S_N_c
                e_c, _ = compute_E_dis_joules(diss_c)
                e_comsol_L.append(e_c)

        ax.plot(L_range, e_paper_L, color=COLORS[name], linewidth=1.5, label=f'{name} paper')
        if e_comsol_L:
            ax.plot(L_range, e_comsol_L, '--', color=COLORS[name], linewidth=1.5,
                    label=f'{name} COMSOL', alpha=0.8)

    # Mark L=6.5 cm
    ax.axvline(6.5, color='gray', ls=':', alpha=0.5, label='L=6.5cm (expt)')
    ax.set_xlabel('Coating length L (cm)', fontsize=12)
    ax.set_ylabel('E$_{dis}$ (J)', fontsize=12)
    ax.set_title(f'{medium}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('E$_{dis}$ vs coating length — saturation behavior', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_Edis_vs_length.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Fig saved: fig4_Edis_vs_length.png")

print("\nAll done.")
