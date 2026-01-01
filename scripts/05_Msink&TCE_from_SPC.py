# Msink_and_TCE_SPC.py
# Calculates:
# (1) I(H) = ∫_0^H f_SPC(t) * IRF(t) dt
# (2) TCE_SPC(H) = - M_sink * I(H)
# (3) M_sink_required(H) = TCE_CH4(100) / I(H)
# Includes analytic integral and numeric sanity check.
# Optional plot with shaded area.

import numpy as np
import matplotlib.pyplot as plt
import argparse

# ==========================================================
# USER SETTINGS (easy to edit)
# ==========================================================
# You can change the horizon either by editing DEFAULT_H in the following ling or by running python Msink_and_TCE_SPC.py --H 10.
DEFAULT_H = 20          # sink horizon in years (recommended 1..20)
DEFAULT_M_CH4 = 1.0     # methane pulse in tCH4
DEFAULT_GWP100_CH4 = 27.0
DEFAULT_M_SINK = 1.0    # initial sink mass (for TCE demonstration)

# ==========================================================
# Parameters
# ==========================================================
# SPC retention (Eq. 11)
a1_spc, k1 = 0.1787, 0.5337
a2_spc, k2 = 0.8237, 0.00997

# IRF(t) (Jeltsch-Thömmes & Joos, 2019)
a0 = 0.008
a = np.array([0.044, 0.112, 0.224, 0.310, 0.297], dtype=float)
tau = np.array([68521.0, 5312.0, 362.0, 47.0, 6.0], dtype=float)

# Methane reference (biogenic default)
GWP100_CH4_DEFAULT = 27.0
IRFint100 = 50.480844011106  # analytic value from the IRF parameterization

# ==========================================================
# Functions
# ==========================================================

def f_SPC(t):
    t = np.asarray(t, dtype=float)
    return a1_spc*np.exp(-k1*t) + a2_spc*np.exp(-k2*t)

def IRF(t):
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return a0 + np.sum(a*np.exp(-t/tau))
    return a0 + np.sum(a*np.exp(-t[:, None]/tau), axis=1)

def y(t):
    return f_SPC(t) * IRF(t)

def trapz(yvals, xvals):
    yvals = np.asarray(yvals, dtype=float)
    xvals = np.asarray(xvals, dtype=float)
    return np.sum((yvals[1:] + yvals[:-1]) * 0.5 * (xvals[1:] - xvals[:-1]))

def I_numeric(H, n=20_000):
    t = np.linspace(0.0, float(H), n+1)
    return trapz(y(t), t)

def I_analytic(H):
    """
    Analytic integral:
    ∫_0^H (a1 e^-k1 t + a2 e^-k2 t) * (a0 + Σ ai e^-t/tau_i ) dt
    """
    H = float(H)

    const_part = a0 * (
        a1_spc*(1.0 - np.exp(-k1*H))/k1 +
        a2_spc*(1.0 - np.exp(-k2*H))/k2
    )

    exp_part = 0.0
    for ai, taui in zip(a, tau):
        lam = 1.0/taui
        exp_part += ai * (
            a1_spc*(1.0 - np.exp(-(k1 + lam)*H))/(k1 + lam) +
            a2_spc*(1.0 - np.exp(-(k2 + lam)*H))/(k2 + lam)
        )

    return const_part + exp_part

def TCE_CH4_100(M_CH4=1.0, GWP100_CH4=GWP100_CH4_DEFAULT):
    """
    TCE_CH4(100) = GWP100(CH4) * IRF_int(100) * M_CH4
    Unit: tCO2e·yr
    """
    return float(M_CH4) * float(GWP100_CH4) * IRFint100

def TCE_SPC(H, M_sink):
    """
    TCE_SPC(H) = - M_sink * I(H)
    Unit: tCO2e·yr  (if M_sink is in tCO2e)
    """
    return -float(M_sink) * I_analytic(H)

def M_sink_required(H, M_CH4=1.0, GWP100_CH4=GWP100_CH4_DEFAULT):
    """
    M_sink_required(H) = TCE_CH4(100) / I(H)
    Unit: tCO2e
    """
    return TCE_CH4_100(M_CH4, GWP100_CH4) / I_analytic(H)

# ==========================================================
# Plot: y(t)=f_SPC(t)*IRF(t) and shaded area
# ==========================================================
def plot_kernel(H, tmax=100, dt=0.1, out_png="kernel_area.png", show=False):
    t = np.arange(0.0, tmax + dt, dt)
    yvals = y(t)
    mask = (t <= H)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, yvals)
    plt.fill_between(t[mask], 0, yvals[mask], alpha=0.3)
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
    parser = argparse.ArgumentParser(description="Compute TCE_SPC(H) and required M_sink for methane offsetting")
    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Sink horizon (years)")
    parser.add_argument("--M_sink", type=float, default=DEFAULT_M_SINK, help="Initial sink mass in tCO2e (for TCE_SPC)")
    parser.add_argument("--M_CH4", type=float, default=DEFAULT_M_CH4, help="Methane pulse size in tCH4")
    parser.add_argument("--GWP100_CH4", type=float, default=DEFAULT_GWP100_CH4, help="GWP100(CH4); default=27.0 (biogenic)")
    parser.add_argument("--n", type=int, default=20_000, help="Trapezoids for numeric sanity check")
    parser.add_argument("--plot", action="store_true", help="Generate kernel plot PNG with shaded area")
    parser.add_argument("--tmax", type=int, default=100, help="Max time for plot x-axis (years)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for plotting (years)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    H = args.H

    # Kernel integral
    I_a = I_analytic(H)
    I_n = I_numeric(H, n=args.n)

    # TCE of a given sink mass
    tce_sink = TCE_SPC(H, args.M_sink)

    # Methane TCE(100)
    tce_ch4 = TCE_CH4_100(args.M_CH4, args.GWP100_CH4)

    # Required sink mass
    M_req = M_sink_required(H, args.M_CH4, args.GWP100_CH4)

    print("\n--- Input settings ---")
    print(f"H = {H} yr")
    print(f"M_CH4 = {args.M_CH4} tCH4")
    print(f"GWP100_CH4 = {args.GWP100_CH4}")
    print(f"M_sink (for TCE_SPC) = {args.M_sink} tCO2e")

    print(f"I(H) = ∫_0^H f_SPC(t)·IRF(t) dt = {I_a:.12f} yr   - value to use")
    print(f"I(H) numeric sanity check      = {I_n:.12f} yr   (n={args.n})")
    print(f"abs diff                       = {abs(I_n - I_a):.3e} yr")

    print(f"\nTCE_SPC(H) for M_sink={args.M_sink} tCO2e:")
    print(f"TCE_SPC({H}) = {tce_sink:.6f} tCO2e·yr")

    print(f"\nTCE_CH4(100) for M_CH4={args.M_CH4} tCH4 and GWP100={args.GWP100_CH4}:")
    print(f"TCE_CH4(100) = {tce_ch4:.6f} tCO2e·yr")

    print(f"\nRequired initial SPC sink mass to offset methane:")
    print(f"M_sink_required({H}) = {M_req:.6f} tCO2e")

    if args.plot:
        out_png = f"SPC_IRF_kernel_area_H{H}.png"
        plot_kernel(H=H, tmax=args.tmax, dt=args.dt, out_png=out_png, show=args.show)
        print(f"\nPlot saved as: {out_png}")
