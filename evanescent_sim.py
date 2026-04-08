#!/usr/bin/env python3
"""
2D Wave Optics (COMSOL via MPh) simulation of evanescent wave interaction
with randomly distributed TiO2 nanoparticles on a quartz optical fiber surface.

Validates the energy balance model of Song et al., Nat. Commun. 12, 4101 (2021).

Usage:
    python evanescent_sim.py [--seed N] [--pol TE|TM] [--no-mph] \
        <theta_deg> <n_ext> <case> [case2 ...]
    python evanescent_sim.py 75 1.33 low
    python evanescent_sim.py 75 1.33 all
    python evanescent_sim.py 60 1.0 low med high
    python evanescent_sim.py --pol TM 75 1.33 low

Arguments:
    --seed N   : random seed for particle generation (default 42)
    --pol TE/TM: polarization. TE = E perpendicular to plane of incidence
                 (out-of-plane Ez, default — backward compatible).
                 TM = E in plane of incidence (in-plane Ex,Ey; H out-of-plane).
                 Real LED light is unpolarized → average TE and TM.
    --no-mph   : do not save the .mph file
    theta_deg  : incidence angle in degrees (e.g. 75)
    n_ext      : external medium refractive index (air=1.0, water=1.33)
    case       : low, med, high, or all
"""

import mph
import numpy as np
import json
import sys
import os
import jpype
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ====================================================================
# Physical parameters (fixed)
# ====================================================================
LAMBDA_NM = 365.0          # wavelength [nm]
N_Q       = 1.46           # quartz refractive index
N_T_RE    = 2.5            # TiO2 refractive index (real part)
N_T_IM    = 0.017689       # TiO2 extinction coefficient
R_NM      = 10.5           # TiO2 particle radius [nm]

# ====================================================================
# Domain dimensions [nm]
# ====================================================================
W_NM  = 10000.0            # domain width  (10 μm)
HQ_NM = 800.0              # quartz slab height
HA_NM = 600.0              # external medium region height
HP_NM = 400.0              # PML thickness (top and bottom)

# ====================================================================
# Case definitions (Song et al. Table 1 / Supplementary)
# ====================================================================
CASES = {
    'low':  {'p': 0.034, 'za_nm': 114.3, 'label': 'TiO2-QOF-Low'},
    'med':  {'p': 0.206, 'za_nm': 52.9,  'label': 'TiO2-QOF-Med'},
    'high': {'p': 0.528, 'za_nm': 7.7,   'label': 'TiO2-QOF-High'},
}


# ====================================================================
# Naming helper
# ====================================================================
def make_tag(p, za, theta_deg, n_ext, seed=42, pol='TE'):
    """Build file/folder naming tag.

    TE (default) keeps the historical naming, e.g. 'p_0.034_za_114.3_75_1.33'
    or 'p_0.034_za_114.3_75_1.33_s7', so existing data on disk is unaffected.

    TM appends a '_TM' marker before the seed:
        'p_0.034_za_114.3_75_1.33_TM' / '..._TM_s7'.
    """
    base = f"p_{p:g}_za_{za:g}_{theta_deg:g}_{n_ext:g}"
    if pol == 'TM':
        base += "_TM"
    return base if seed == 42 else f"{base}_s{seed}"


# ====================================================================
# Particle generation
# ====================================================================
def generate_particles(p, za_nm, r, w, seed=42):
    """
    Generate random 2D TiO2 particle positions.

    Interface at y=0.  Quartz at y>0, external medium at y<0.

    Contact particles: centers at y=0 (straddling interface), each covers
    a chord of 2r on the surface → patchiness = n_contact * 2r / w.

    Floating particles: centers at y = -(r + gap), gap ~ Exp(za_nm).
    These don't touch the surface; their average gap ≈ za_nm.

    Returns list of (x, y) in nm.
    """
    rng = np.random.default_rng(seed)

    # --- contact particles ---
    n_contact = max(1, round(p * w / (2 * r)))
    xs = []
    for _ in range(n_contact * 500):
        if len(xs) >= n_contact:
            break
        x = rng.uniform(r, w - r)
        if all(abs(x - xc) > 2.5 * r for xc in xs):
            xs.append(x)
    particles = [(x, 0.0) for x in xs]
    actual_p = len(xs) * 2 * r / w
    print(f"    Contact: {len(xs)}/{n_contact} placed, actual p={actual_p:.4f}")

    # --- floating particles ---
    n_float = max(5, round(len(xs) * 1.5))
    placed = 0
    for _ in range(n_float * 20):
        if placed >= n_float:
            break
        x = rng.uniform(r, w - r)
        gap = max(1.0, rng.exponential(za_nm))
        y = -(r + gap)
        if abs(y) + r > HA_NM - 10:
            continue
        if all((x - px)**2 + (y - py)**2 > (2.2 * r)**2
               for px, py in particles):
            particles.append((x, y))
            placed += 1
    print(f"    Floating: {placed}")
    print(f"    Total particles: {len(particles)}")
    return particles


# ====================================================================
# Matplotlib field plot
# ====================================================================
def plot_field_matplotlib(data_path, img_path, label, particles, n_ext):
    """Read COMSOL-exported field data and plot with matplotlib."""
    # Parse COMSOL text export (header lines start with %)
    rows = []
    with open(data_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            try:
                vals = [float(v) for v in s.split()]
                if len(vals) >= 3 and all(np.isfinite(vals)):
                    rows.append(vals)
            except ValueError:
                continue
    arr = np.array(rows)
    x, y, E = arr[:, 0], arr[:, 1], arr[:, 2]

    # Auto-detect unit: if coordinates are in meters, convert to nm
    if x.max() < 1.0:
        x, y = x * 1e9, y * 1e9

    # Exclude PML regions for a clean view
    keep = (y >= -HA_NM) & (y <= HQ_NM)
    x, y, E = x[keep], y[keep], E[keep]

    tri = Triangulation(x, y)
    vmax = np.percentile(E, 99.5)

    fig, ax = plt.subplots(figsize=(18, 5))
    levels = np.linspace(0, vmax, 120)
    tcf = ax.tricontourf(tri, E, levels=levels, cmap='hot', extend='max')
    fig.colorbar(tcf, ax=ax, label='|E| (V/m)', pad=0.02, shrink=0.9)

    # Interface line and region labels
    ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.6)
    ax.text(W_NM * 0.01, HQ_NM * 0.8, f'Quartz (n={N_Q})',
            color='white', fontsize=11, fontweight='bold')
    ax.text(W_NM * 0.01, -HA_NM * 0.8, f'External (n={n_ext})',
            color='white', fontsize=11, fontweight='bold')

    # Draw particle outlines
    for cx, cy in particles:
        ax.add_patch(plt.Circle((cx, cy), R_NM, fill=False,
                                ec='cyan', lw=0.4, alpha=0.8))

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title(f'Electric field |E| — {label}')
    ax.set_xlim(0, W_NM)
    ax.set_ylim(-HA_NM, HQ_NM)
    ax.set_aspect('equal')
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [10a] Matplotlib plot saved: {img_path}")


# ====================================================================
# COMSOL model builder
# ====================================================================
def build_model(case_name, client, theta_deg, n_ext, seed=42, save_mph=True,
                pol='TE'):
    """Build, mesh, solve a COMSOL 2D ewfd model.  Return the model.

    Parameters
    ----------
    pol : 'TE' or 'TM'
        TE = E perpendicular to plane of incidence (out-of-plane Ez).
             Background field uses TE Fresnel (rs, ts) for TIR; |E₀| = 1 V/m.
        TM = E in plane of incidence (in-plane Ex, Ey; H out-of-plane).
             Background field derived from |H₀| = 1 A/m via Maxwell:
                E = (1/(iωε)) ∇×H.
             TM Fresnel reflection at TIR:
                rp = (a + ib)/(a - ib),  a = ε_ext·ky_q,  b = ε_q·κ_ev.
             tp = 1 + rp (from Hz continuity at the interface).
        Real LED light is unpolarized; the analysis pipeline averages
        TE+TM after both have been simulated.
    """
    if pol not in ('TE', 'TM'):
        raise ValueError(f"pol must be 'TE' or 'TM', got {pol!r}")

    case  = CASES[case_name]
    p     = case['p']
    za    = case['za_nm']
    label = case['label']
    tag   = make_tag(p, za, theta_deg, n_ext, seed, pol=pol)

    print(f"\n{'=' * 60}")
    print(f"  {label}:  p = {p},  za = {za} nm")
    print(f"  theta = {theta_deg}°,  n_ext = {n_ext},  pol = {pol}")
    print(f"  tag = {tag}")
    print(f"{'=' * 60}")

    # --- output directories ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    case_dir   = os.path.join(script_dir, tag)
    os.makedirs(case_dir, exist_ok=True)

    # --- particle positions ---
    particles = generate_particles(p, za, R_NM, W_NM, seed=seed)
    with open(os.path.join(case_dir, f'particles_{tag}.json'), 'w') as f:
        json.dump({'case': case_name, 'p': p, 'za_nm': za,
                   'theta_deg': theta_deg, 'n_ext': n_ext,
                   'r_nm': R_NM, 'n': len(particles),
                   'xy_nm': particles}, f, indent=2)

    # --- create model ---
    model = client.create(f'ev_{case_name}')
    j = model.java                              # raw Java API handle

    # ================================================================
    # 1.  Global parameters
    # ================================================================
    P = j.param()
    P.set("lda0",     f"{LAMBDA_NM}[nm]")
    P.set("k0",       "2*pi/lda0")
    P.set("f0",       "c_const/lda0")
    P.set("n_q",      str(N_Q))
    P.set("n_ext",    str(n_ext))
    P.set("n_t_re",   str(N_T_RE))
    P.set("n_t_im",   str(N_T_IM))
    P.set("theta_i",  f"{theta_deg}[deg]")
    # derived wave-vector components
    P.set("kx_f",     "n_q*k0*sin(theta_i)")
    P.set("ky_q",     "n_q*k0*cos(theta_i)")
    P.set("kappa_ev", "sqrt(kx_f^2 - (n_ext*k0)^2)")
    # Fresnel TE reflection / transmission for TIR at quartz→external
    P.set("alp",      "n_q*cos(theta_i)")
    P.set("bet",      "sqrt(n_q^2*sin(theta_i)^2 - n_ext^2)")
    P.set("den",      "alp^2 + bet^2")
    P.set("rs_re",    "(alp^2 - bet^2)/den")
    P.set("rs_im",    "-2*alp*bet/den")
    P.set("ts_re",    "2*alp^2/den")
    P.set("ts_im",    "-2*alp*bet/den")
    # ---- TM (p-pol) Fresnel + background-field prefactors ----
    # rp = (a + ib)/(a - ib)  with  a = ε_ext·ky_q,  b = ε_q·κ_ev
    # ε₀ cancels in the ratio so we use n²·k instead.
    P.set("a_tm",     "n_ext^2*ky_q")
    P.set("b_tm",     "n_q^2*kappa_ev")
    P.set("d_tm",     "a_tm^2 + b_tm^2")
    P.set("rp_re",    "(a_tm^2 - b_tm^2)/d_tm")
    P.set("rp_im",    "2*a_tm*b_tm/d_tm")
    P.set("tp_re",    "1 + rp_re")        # tp = 1 + rp (Hz continuity)
    P.set("tp_im",    "rp_im")
    # Background-E prefactors for TM with H₀ = 1 A/m:
    #   E = (1/(iωε)) ∇×H  →  Ex,Ey have units of impedance × A/m = V/m.
    P.set("omega",    "2*pi*f0")
    P.set("eps_q_si", "epsilon0_const*n_q^2")
    P.set("eps_e_si", "epsilon0_const*n_ext^2")
    P.set("Cqx",      "ky_q/(omega*eps_q_si)")     # E_x prefactor in quartz
    P.set("Cqy",      "kx_f/(omega*eps_q_si)")     # E_y prefactor in quartz
    P.set("Cex",      "kappa_ev/(omega*eps_e_si)") # E_x prefactor below (×−i)
    P.set("Cey",      "kx_f/(omega*eps_e_si)")     # E_y prefactor below
    print("  [1] Parameters set.")

    # ================================================================
    # 2.  Component — 2-D
    # ================================================================
    j.component().create("comp1", True)

    # ================================================================
    # 3.  Geometry  (length unit = nm)
    # ================================================================
    geom = j.component("comp1").geom().create("geom1", 2)
    geom.lengthUnit("nm")

    # cumulative selections for later material / BC assignment
    for sel_tag in ("csel_q", "csel_ext", "csel_pml_top", "csel_pml_bot", "csel_tio2"):
        geom.selection().create(sel_tag, "CumulativeSelection")

    # four background rectangles
    rects = [
        ("r_pml_top", [W_NM, HP_NM],  [0, HQ_NM],               "csel_pml_top"),
        ("r_quartz",  [W_NM, HQ_NM],  [0, 0],                   "csel_q"),
        ("r_ext",     [W_NM, HA_NM],  [0, -HA_NM],              "csel_ext"),
        ("r_pml_bot", [W_NM, HP_NM],  [0, -(HA_NM + HP_NM)],   "csel_pml_bot"),
    ]
    for rect_tag, sz, pos, csel in rects:
        geom.create(rect_tag, "Rectangle")
        geom.feature(rect_tag).set("size", sz)
        geom.feature(rect_tag).set("pos", pos)
        geom.feature(rect_tag).set("contributeto", csel)

    # TiO2 circles
    for i, (cx, cy) in enumerate(particles):
        ctag = f"c{i}"
        geom.create(ctag, "Circle")
        geom.feature(ctag).set("r", R_NM)
        geom.feature(ctag).set("pos", [cx, cy])
        geom.feature(ctag).set("contributeto", "csel_tio2")

    geom.run("fin")
    print(f"  [3] Geometry built  ({len(particles)} particles).")

    # ================================================================
    # 4.  Definitions: box selections + PML coordinate systems
    # ================================================================
    defs = j.component("comp1")

    # box selections for left / right periodic boundaries
    y_lo = -(HA_NM + HP_NM + 1)
    y_hi =   HQ_NM + HP_NM + 1

    JInt = jpype.JInt   # explicit Java int to avoid JPype overload ambiguity

    defs.selection().create("bx_left", "Box")
    defs.selection("bx_left").set("entitydim", JInt(1))
    defs.selection("bx_left").set("xmin", -1.0)
    defs.selection("bx_left").set("xmax",  1.0)
    defs.selection("bx_left").set("ymin", y_lo)
    defs.selection("bx_left").set("ymax", y_hi)
    defs.selection("bx_left").set("condition", "allvertices")

    defs.selection().create("bx_right", "Box")
    defs.selection("bx_right").set("entitydim", JInt(1))
    defs.selection("bx_right").set("xmin", W_NM - 1.0)
    defs.selection("bx_right").set("xmax", W_NM + 1.0)
    defs.selection("bx_right").set("ymin", y_lo)
    defs.selection("bx_right").set("ymax", y_hi)
    defs.selection("bx_right").set("condition", "allvertices")

    defs.selection().create("bx_periodic", "Union")
    defs.selection("bx_periodic").set("entitydim", JInt(1))
    defs.selection("bx_periodic").set("input", ["bx_left", "bx_right"])

    # PML coordinate stretching
    defs.coordSystem().create("pml_top", "PML")
    defs.coordSystem("pml_top").selection().named("geom1_csel_pml_top_dom")

    defs.coordSystem().create("pml_bot", "PML")
    defs.coordSystem("pml_bot").selection().named("geom1_csel_pml_bot_dom")

    print("  [4] Selections & PML done.")

    # ================================================================
    # 5.  Materials
    # ================================================================
    def add_mat(mtag, lbl, sel_name, n_expr, ki_expr):
        m = j.component("comp1").material().create(mtag, "Common")
        m.label(lbl)
        m.selection().named(sel_name)
        m.propertyGroup().create("RefractiveIndex", "Refractive index")
        m.propertyGroup("RefractiveIndex").set("n",  [n_expr])
        m.propertyGroup("RefractiveIndex").set("ki", [ki_expr])

    add_mat("m_q",   "Quartz",           "geom1_csel_q_dom",       "n_q",    "0")
    add_mat("m_qp",  "Quartz (PML)",     "geom1_csel_pml_top_dom", "n_q",    "0")
    add_mat("m_ext", "External medium",   "geom1_csel_ext_dom",     "n_ext",  "0")
    add_mat("m_ep",  "External (PML)",    "geom1_csel_pml_bot_dom", "n_ext",  "0")
    add_mat("m_t",   "TiO2",            "geom1_csel_tio2_dom",    "n_t_re", "n_t_im")
    print("  [5] Materials assigned.")

    # ================================================================
    # 6.  Physics — Electromagnetic Waves, Frequency Domain  (TE)
    # ================================================================
    ewfd = j.component("comp1").physics().create(
        "ewfd", "ElectromagneticWavesFrequencyDomain", "geom1")

    # Choose which E-field components to solve for, based on polarization.
    # Default (TE) keeps the historical out-of-plane Ez vector.
    if pol == 'TM':
        # In-plane vector: solve for (Ex, Ey); Hz is the natural out-of-plane H.
        # The 2D ewfd "Electric field components solved for" setting lives in
        # the lowercase property group `components`, with the same lowercase
        # key. Valid values: "outofplane" (TE, default), "inplane" (TM),
        # "threeComp" (full three-component vector).
        ewfd.prop("components").set("components", "inplane")

    # scattered-field formulation
    ewfd.prop("BackgroundField").set("SolveFor", "scatteredField")

    if pol == 'TE':
        # background E: analytical TIR solution (out-of-plane Ez)
        #   quartz (y>0) : E_inc + r_s * E_refl
        #   ext    (y<0) : t_s * evanescent
        ebz = (
            "if(y>0,"
            "  exp(-i*(kx_f*x - ky_q*y))"
            "  + (rs_re + i*rs_im)*exp(-i*(kx_f*x + ky_q*y)),"
            "  (ts_re + i*ts_im)*exp(-i*kx_f*x)*exp(kappa_ev*y)"
            ")"
        )
        ewfd.prop("BackgroundField").set("Eb", ["0", "0", ebz])
    else:  # pol == 'TM'
        # Background E for TM with |H₀| = 1 A/m:
        #   Above (y>0):
        #     Hz = exp(-i(kx·x − ky·y)) + rp · exp(-i(kx·x + ky·y))
        #     Ex = (ky/(ωε_q)) · [ exp(...inc) − rp · exp(...refl) ]
        #     Ey = (kx/(ωε_q)) · [ exp(...inc) + rp · exp(...refl) ]
        #   Below (y<0):
        #     Hz = tp · exp(-i kx·x) · exp(κ·y)
        #     Ex = −i (κ/(ωε_ext)) · tp · exp(...) · exp(κy)
        #     Ey =     (kx/(ωε_ext)) · tp · exp(...) · exp(κy)
        ebx_above = (
            "Cqx*( exp(-i*(kx_f*x - ky_q*y))"
            " - (rp_re + i*rp_im)*exp(-i*(kx_f*x + ky_q*y)) )"
        )
        eby_above = (
            "Cqy*( exp(-i*(kx_f*x - ky_q*y))"
            " + (rp_re + i*rp_im)*exp(-i*(kx_f*x + ky_q*y)) )"
        )
        ebx_below = (
            "-i*Cex*(tp_re + i*tp_im)*exp(-i*kx_f*x)*exp(kappa_ev*y)"
        )
        eby_below = (
            "Cey*(tp_re + i*tp_im)*exp(-i*kx_f*x)*exp(kappa_ev*y)"
        )
        ebx = f"if(y>0, {ebx_above}, {ebx_below})"
        eby = f"if(y>0, {eby_above}, {eby_below})"
        ewfd.prop("BackgroundField").set("Eb", [ebx, eby, "0"])

    # Floquet periodic condition on left + right boundaries
    pc = ewfd.create("pc1", "PeriodicCondition", 1)
    pc.selection().named("bx_periodic")
    pc.set("PeriodicType", "Floquet")
    pc.set("kFloquet", ["kx_f", "0", "0"])

    print(f"  [6] Physics configured (ewfd, scattered field, Floquet, pol={pol}).")

    # ================================================================
    # 7.  Mesh
    # ================================================================
    mesh = j.component("comp1").mesh().create("mesh1")

    # fine size on TiO2 particles
    sz_t = mesh.create("sz_t", "Size")
    sz_t.selection().named("geom1_csel_tio2_dom")
    sz_t.set("custom", "on")
    sz_t.set("hmax", "3")
    sz_t.set("hmin", "0.5")

    # global max element size  (λ/n ≈ 250nm → λ/(8n) ≈ 31nm)
    mesh.feature("size").set("custom", "on")
    mesh.feature("size").set("hmax", "35")
    mesh.feature("size").set("hmin", "0.5")

    mesh.create("ftri1", "FreeTri")
    mesh.run()
    print("  [7] Mesh generated.")

    # ================================================================
    # 8.  Study — frequency domain
    # ================================================================
    std = j.study().create("std1")
    std.create("freq", "Frequency")
    std.feature("freq").set("plist", "f0")
    print("  [8] Study created.")

    # ================================================================
    # 9.  Solve
    # ================================================================
    print("  [9] Solving  (this may take a few minutes) ...")
    j.study("std1").run()
    print("  [9] Solved!")

    # ================================================================
    # 10. Results — field plot (PNG with colorbar) + T / R / A
    # ================================================================
    res = j.result()

    # ---- Export field data → matplotlib plot with axes & colorbar ----
    img_path  = os.path.join(case_dir, f"field_{tag}.png")
    data_path = os.path.join(case_dir, f"field_data_{tag}.txt")
    try:
        exp_data = res.export().create("data1", "Data")
        exp_data.set("expr", ["ewfd.normE"])
        exp_data.set("filename", data_path)
        exp_data.run()
        print(f"  [10a] Field data exported: {data_path}")
        plot_field_matplotlib(data_path, img_path, label, particles, n_ext)
    except Exception as e:
        print(f"  [10a] Data export / plot failed ({e})")
        import traceback; traceback.print_exc()

    # ---- Cut-line datasets for Poynting-vector flux ----
    y_cut_q = HQ_NM - 50.0            # in quartz, 50 nm below top PML
    y_cut_a = -(HA_NM - 20.0)         # in ext medium, 20 nm above bottom PML

    ds_q = res.dataset().create("cut_q", "CutLine2D")
    ds_q.set("genpoints", [[0.0, y_cut_q], [W_NM, y_cut_q]])

    ds_a = res.dataset().create("cut_a", "CutLine2D")
    ds_a.set("genpoints", [[0.0, y_cut_a], [W_NM, y_cut_a]])

    # ---- Numerical integrations ----
    int_q = res.numerical().create("int_q", "IntLine")
    int_q.set("data", "cut_q")
    int_q.set("expr", ["ewfd.Poavy"])

    int_a = res.numerical().create("int_a", "IntLine")
    int_a.set("data", "cut_a")
    int_a.set("expr", ["ewfd.Poavy"])

    int_abs = res.numerical().create("int_abs", "IntSurface")
    int_abs.selection().named("geom1_csel_tio2_dom")
    int_abs.set("expr", ["ewfd.Qh"])

    # ---- Analytical incident power ----
    # TE: |E₀| = 1 V/m, ⟨S_y⟩ = ky_q / (2 ω μ₀)
    # TM: |H₀| = 1 A/m, ⟨S_y⟩ = ky_q / (2 ω ε_q)  with ε_q = ε₀ n_q²
    # Both yield W/m² downward; multiplying by domain width W gives W/m
    # (per metre into the page, which is the 2-D convention).
    mu0   = 4.0 * np.pi * 1e-7       # H/m
    eps0  = 8.8541878128e-12         # F/m
    c0    = 299792458.0              # m/s
    lda_m = LAMBDA_NM * 1e-9
    omega = 2.0 * np.pi * c0 / lda_m
    theta_rad = np.radians(theta_deg)
    ky_q_val = N_Q * (omega / c0) * np.cos(theta_rad)

    if pol == 'TM':
        eps_q_si = eps0 * N_Q ** 2
        S_inc_y  = ky_q_val / (2.0 * omega * eps_q_si)
    else:  # TE (default)
        S_inc_y  = ky_q_val / (2.0 * omega * mu0)

    P_inc = W_NM * 1e-9 * S_inc_y
    print(f"  [10b] Analytical P_inc ({pol}) = {P_inc:.6e} W/m")

    # ---- Read fluxes and compute T, R, A ----
    def _read_scalar(num_node):
        """Extract a scalar from COMSOL numerical result."""
        num_node.setResult()
        raw = num_node.getReal()
        try:
            return float(raw[0][0])
        except (TypeError, IndexError):
            return float(raw[0])

    try:
        P_trans = _read_scalar(int_a)
        P_abs   = _read_scalar(int_abs)
        P_net_q = _read_scalar(int_q)

        T_frac = abs(P_trans) / P_inc
        A_frac = abs(P_abs)  / P_inc
        R_frac = 1.0 - T_frac - A_frac

        print(f"\n  {'='*50}")
        print(f"  Energy balance — {label}")
        print(f"  theta={theta_deg}°, n_ext={n_ext}")
        print(f"  {'='*50}")
        print(f"    P_inc  (analytical)  = {P_inc:.6e} W/m")
        print(f"    P_trans (ext line)   = {P_trans:.6e} W/m")
        print(f"    P_abs   (TiO2)      = {P_abs:.6e} W/m")
        print(f"    P_net   (quartz line)= {P_net_q:.6e} W/m")
        print(f"  ---")
        print(f"    Transmittance  T = {T_frac:.6f}  ({T_frac*100:.4f}%)")
        print(f"    Absorptance    A = {A_frac:.6f}  ({A_frac*100:.4f}%)")
        print(f"    Reflectance    R = {R_frac:.6f}  ({R_frac*100:.4f}%)")
        print(f"    (R = 1 − T − A, energy conservation)")
        print(f"  {'='*50}\n")

        # Energy balance txt → script directory (not case subdirectory)
        txt_path = os.path.join(script_dir, f"energy_balance_{tag}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Energy balance — {label}\n")
            f.write(f"Case: {case_name},  p = {p},  za = {za} nm\n")
            f.write(f"theta = {theta_deg} deg,  n_ext = {n_ext},  pol = {pol}\n")
            f.write(f"{'='*50}\n")
            f.write(f"P_inc  (analytical)   = {P_inc:.6e} W/m\n")
            f.write(f"P_trans (ext line)    = {P_trans:.6e} W/m\n")
            f.write(f"P_abs   (TiO2)       = {P_abs:.6e} W/m\n")
            f.write(f"P_net   (quartz line) = {P_net_q:.6e} W/m\n")
            f.write(f"---\n")
            f.write(f"Transmittance  T = {T_frac:.6f}  ({T_frac*100:.4f}%)\n")
            f.write(f"Absorptance    A = {A_frac:.6f}  ({A_frac*100:.4f}%)\n")
            f.write(f"Reflectance    R = {R_frac:.6f}  ({R_frac*100:.4f}%)\n")
            f.write(f"(R = 1 − T − A, energy conservation)\n")
            f.write(f"{'='*50}\n")
        print(f"  [10b] Energy balance saved: {txt_path}")
    except Exception as e:
        print(f"  [10b] Power readout failed ({e}); check in GUI.")
        import traceback; traceback.print_exc()

    # ---- Save COMSOL model ----
    if save_mph:
        mph_path = os.path.join(case_dir, f"evanescent_{tag}.mph")
        model.save(mph_path)
        print(f"  Model saved: {mph_path}")
    else:
        print(f"  Skipped .mph save (--no-mph)")

    return model


# ====================================================================
# Main
# ====================================================================
def main():
    # Parse optional flags from argv
    seed = 42
    save_mph = True
    pol = 'TE'
    argv = sys.argv[1:]
    if '--seed' in argv:
        idx = argv.index('--seed')
        seed = int(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2:]
    if '--pol' in argv:
        idx = argv.index('--pol')
        pol = argv[idx + 1].upper()
        argv = argv[:idx] + argv[idx + 2:]
    if '--no-mph' in argv:
        argv.remove('--no-mph')
        save_mph = False

    if len(argv) < 3:
        print("Usage: python evanescent_sim.py [--seed N] [--pol TE|TM] [--no-mph] "
              "<theta_deg> <n_ext> <case> [case2 ...]")
        print("  --seed N    : random seed for particle generation (default 42)")
        print("  --pol TE/TM : polarization (default TE; backward compatible)")
        print("  --no-mph    : do not save .mph file")
        print("  theta_deg   : incidence angle in degrees (e.g. 75)")
        print("  n_ext       : external medium refractive index (air=1.0, water=1.33)")
        print("  case        : low, med, high, or all")
        print()
        print("Examples:")
        print("  python evanescent_sim.py 75 1.33 low                   # TE, water, 75°")
        print("  python evanescent_sim.py --pol TM 75 1.33 low          # TM run")
        print("  python evanescent_sim.py --seed 7 --pol TM 75 1.33 all # seed=7, TM")
        print("  python evanescent_sim.py 60 1.0  low                   # air, 60°")
        sys.exit(1)

    if pol not in ('TE', 'TM'):
        print(f"ERROR: --pol must be 'TE' or 'TM', got {pol!r}")
        sys.exit(1)

    theta_deg = float(argv[0])
    n_ext     = float(argv[1])
    targets   = argv[2:]

    # Check critical angle
    theta_c = np.degrees(np.arcsin(n_ext / N_Q))
    print(f"Settings: theta = {theta_deg}°,  n_ext = {n_ext},  "
          f"seed = {seed},  pol = {pol}")
    print(f"  Critical angle = {theta_c:.1f}°")
    if theta_deg <= theta_c:
        print(f"  WARNING: theta={theta_deg}° <= critical angle={theta_c:.1f}°")
        print(f"  No total internal reflection! Light will refract into medium.")
        print(f"  Consider using theta > {theta_c:.1f}°")

    if 'all' in targets or 'both' in targets:
        targets = ['low', 'med', 'high']

    print("Connecting to COMSOL ...")
    client = mph.start()
    print("Connected.\n")

    for name in targets:
        if name not in CASES:
            print(f"Unknown case '{name}'. Choose: low, med, high, or all.")
            continue
        try:
            build_model(name, client, theta_deg, n_ext,
                        seed=seed, save_mph=save_mph, pol=pol)
        except Exception as e:
            print(f"\n  *** ERROR in {name}: {e}")
            print("  The .mph file (if saved) can be opened in COMSOL GUI to debug.")
            import traceback; traceback.print_exc()

    client.disconnect()
    print("\nDone.")


if __name__ == '__main__':
    main()
