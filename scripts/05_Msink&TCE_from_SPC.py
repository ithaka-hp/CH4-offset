# 05_Msink_TCE_from_SPC.py
# Calculates:
#   I(H) = ∫_0^H f_SPC(t) · IRF(t) dt
#   TCE_SPC(H) = - M_sink · I(H)
#   M_sink_required(H) = TCE_CH4(100) / I(H)
# Uses analytic integral + numeric trapezoidal sanity check.
# Optional plot: kernel f_SPC(t)·IRF(t) and shaded area 0..H.

import numpy as np
import matplotlib.pyplot as plt
import argparse

# ==========================================================
# USER SETTINGS (easy to edit)
# ==========================================================
DEFAULT_H = 20                 # years (recommended 1..20)
DEFAULT_M_CH4 = 1.0            # tCH4
DEFAULT_GWP100_CH4 = 27.0      # biogenic default
DEFAULT_M_SINK = 1.0           # tCO2e (for example TCE_SPC output)
DEFAULT_N = 20_000             # trapezoids for numeric sanity check

# Plot defaults
DEFAULT_TMAX = 100             # years for x-axis
DEFAULT_DT = 0.1               # years

# ==========================================================
# Model parameters
# ==========================================================

# SPC retention (Eq. 11)
a1_spc, k1 = 0.1787, 0.5337
a2_spc, k2 = 0.8237, 0.00997

# IRF parameters (Jeltsch-Thömmes & Joos, 2019)
a0 = 0.008
a = np.array([0.044, 0.112, 0.224, 0.310, 0.297], dtype=float)
tau = np.array([68521.0, 5312.0, 362.0, 47.0, 6.0], dtype=float)

# Analytic IRF integral at 100 years (computed from the above parameters)
IRF_INT_100 = 50.480844011106

# ==========================================================
# Functions
# ==========================================================

def f_SPC(t):
    t = np.asarray(t, dtype=float)
    return a1_spc * np.exp(-k1 * t) + a2_spc * np.exp(-k2 * t)

def IRF(t):
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return a0 + np.sum(a * np.exp(-t / tau))
    return a0 + np.sum(a * np.exp(-t[:, None] / tau), axis=1)

def kernel(t):
    """Kernel for diminishing SPC sink: f_SPC(t) · IRF(t)"""
    return f_SPC(t) * IRF(t)

def trapz(yvals, xvals):
    """Robust trapezoidal integration (works across numpy versions)."""
    yvals = np.asarray(yvals, dtype=float)
    xvals = np.asarray(xvals, dtype=float)
    return np.sum((yvals[1:] + yvals[:-1]) * 0.5 * (xvals[1:] - xvals[:-1]))

def I_numeric(H, n=DEFAULT_N):
    t = np.linspace(0.0, float(H), n + 1)
    return trapz(kernel(t), t)

def I_analytic(H):
    """
    Analytic integral:
      I(H) = ∫_0^H (a1 e^-k1 t + a2 e^-k2 t) · (a0 + Σ ai e^-t/tau_i) dt
    """
    H = float(H)

    # Constant IRF term (a0)
    const_part = a0 * (
        a1_spc * (1.0 - np.exp(-k1 * H)) / k1 +
        a2_spc * (1.0 - np.exp(-k2 * H)) / k2
    )

    # Exponential IRF terms
    exp_part = 0.0
    for ai, taui in zip(a, tau):
        lam = 1.0 / taui
        exp_part += ai * (
            a1_spc * (1.0 - np.exp(-(k1 + lam) * H)) / (k1 + lam) +
            a2_spc * (1.0 - np.exp(-(k2 + lam) * H)) / (k2 + lam)
        )

    return const_part + exp_part

def TCE_CH4_100(M_CH4, GWP100_CH4):
    """TCE_CH4(100) = M_CH4 · GWP100(CH4) · IRF_int(100)"""
    return float(M_CH4) * float(GWP100_CH4) * IRF_INT_100

def TCE_SPC(H, M_sink):
    """TCE_SPC(H) = - M_sink · I(H)"""
    return -float(M_sink) * I_analytic(H)

def M_sink_required(H, M_CH4, GWP100_CH4):
    """M_sink_required(H) = TCE_CH4(100) / I(H)"""
    return TCE_CH4_100(M_CH4, GWP100_CH4) / I_analytic(H)

def plot_kernel_area(H, tmax, dt, out_png, show=False):
    t = np.arange(0.0, tmax + dt, dt)
    y = kernel(t)
    mask = (t <= H)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, y)
    plt.fill_between(t[mask], 0, y[mask], alpha=0.3)
    plt.xlabel("t (years)")
    plt.ylabel("f_SPC(t) · IRF(t)")
    plt.title(f"Kernel f_SPC(t)·IRF(t) with shaded integral area (0..H={H} yr)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Methane offsetting with diminishing SPC C-sinks (TCE framework)"
    )

    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Offset horizon in years (recommended 1..20)")
    parser.add_argument("--M_CH4", type=float, default=DEFAULT_M_CH4, help="Methane pulse size in tCH4")
    parser.add_argument("--GWP100_CH4", type=float, default=DEFAULT_GWP100_CH4, help="GWP100(CH4)")
    parser.add_argument("--M_sink", type=float, default=DEFAULT_M_SINK, help="Sink mass in tCO2e (for TCE_SPC example)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Trapezoids for numeric sanity check")

    parser.add_argument("--plot", action="store_true", help="Save kernel plot with shaded area 0..H")
    parser.add_argument("--tmax", type=int, default=DEFAULT_TMAX, help="Max time for plot x-axis (years)")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Time step for plot (years)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")

    args = parser.parse_args()

    H = args.H

    # Compute kernel integral
    I_a = I_analytic(H)
    I_n = I_numeric(H, n=args.n)

    # Methane and sink TCE values
    tce_ch4 = TCE_CH4_100(args.M_CH4, args.GWP100_CH4)
    M_req = M_sink_required(H, args.M_CH4, args.GWP100_CH4)
    tce_spc_req = -M_req * I_a

    # ------------------------------------------------------
    # Output (paper-friendly)
    # ------------------------------------------------------
    print(f"\n--- Methane offsetting with diminishing SPC sink (H = {H} yr) ---\n")

    print(f"TCE_CH4(100) for {args.M_CH4:.3f} tCH4 (GWP100 = {args.GWP100_CH4:.1f}) "
          f"= {tce_ch4:.6f} tCO2e·yr")

    print(f"I(H) = ∫_0^H f_SPC(t)·IRF(t) dt = {I_a:.6f} yr\n")

    print(f"M_sink_required({H}) = {M_req:.6f} tCO2e")
    print(f"TCE_SPC({H}) for M_sink = {M_req:.6f} tCO2e = {tce_spc_req:.6f} tCO2e·yr\n")

    print(f"Offset check:  TCE_CH4(100) + TCE_SPC({H}) = {tce_ch4 + tce_spc_req:.6f} tCO2e·yr")

    # Kernel sanity check
    print("\n--- Sanity check (numeric vs analytic integral) ---")
    print(f"Numeric trapezoid check (n={args.n}) = {I_n:.6f} yr")
    print(f"Absolute difference                  = {abs(I_n - I_a):.3e} yr")

    # Plot
    if args.plot:
        out_png = f"kernel_area_H{H}.png"
        plot_kernel_area(H=H, tmax=args.tmax, dt=args.dt, out_png=out_png, show=args.show)
        print(f"\nPlot saved as: {out_png}")
