"""
06: Jig friction effect on three-point bending hysteresis
=========================================================

Physics:
  In a three-point bending test, the cable rests on two support rollers
  and is loaded at center by a third roller. As the cable deflects,
  it rotates at the support points. Roller friction creates a resisting
  torque that modifies the measured load.

  At each support roller (radius r_s, friction coeff mu):
    friction moment = mu * (P/2) * r_s  (opposes rotation)

  At loading roller (radius r_l):
    friction moment = mu * P * r_l  (opposes curvature change)

  Half-beam moment equilibrium:
    M_center = P*L/4 - mu*P*r_s/2 - mu*P*r_l/2

  Solving for P:
    Loading  (d increasing): P = 4*M / (L - 2*mu*(r_s + r_l))
    Unloading (d decreasing): P = 4*M / (L + 2*mu*(r_s + r_l))

  Key insight:
    - Jig friction creates UNIFORM (multiplicative) hysteresis
    - Internal strand friction creates PROGRESSIVE (strain-history) hysteresis
    - Real cables exhibit both: teardrop shape (internal) + widening (jig)

[<- README](../../README.md)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ============================================================
# Multi-layer friction model (from 05)
# ============================================================
@dataclass
class MultiLayerFriction:
    """N parallel friction elements + elastic backbone."""
    E_base: float
    k_virgin: np.ndarray
    k_degraded: np.ndarray
    f_y: np.ndarray
    slip: np.ndarray = field(default=None)
    slipped: np.ndarray = field(default=None)

    def __post_init__(self):
        n = len(self.k_virgin)
        if self.slip is None:
            self.slip = np.zeros(n)
        if self.slipped is None:
            self.slipped = np.zeros(n, dtype=bool)

    def stress(self, eps):
        sigma = self.E_base * eps
        for i in range(len(self.k_virgin)):
            k = self.k_degraded[i] if self.slipped[i] else self.k_virgin[i]
            trial = k * (eps - self.slip[i])
            if abs(trial) <= self.f_y[i]:
                sigma += trial
            else:
                self.slipped[i] = True
                k = self.k_degraded[i]
                trial = k * (eps - self.slip[i])
                if abs(trial) <= self.f_y[i]:
                    sigma += trial
                else:
                    s = np.sign(trial)
                    self.slip[i] += s * (abs(trial) - self.f_y[i]) / k
                    sigma += s * self.f_y[i]
        return sigma


class FiberSection:
    def __init__(self, diameter, n_fiber=60):
        self.diameter = diameter
        R = diameter / 2.0
        self.y = np.linspace(-R + R/n_fiber, R - R/n_fiber, n_fiber)
        dy = 2 * R / n_fiber
        self.A = np.array([self._area(y, dy, R) for y in self.y])
        self.fibers = []

    def setup(self, factory):
        self.fibers = [factory() for _ in self.y]

    @staticmethod
    def _area(y, dy, R):
        yl, yh = max(y - dy/2, -R), min(y + dy/2, R)
        if yh <= yl:
            return 0.0
        wl = 2 * np.sqrt(max(R**2 - yl**2, 0))
        wh = 2 * np.sqrt(max(R**2 - yh**2, 0))
        return (wl + wh) / 2 * (yh - yl)

    def moment(self, kappa):
        M = 0.0
        for y, A, f in zip(self.y, self.A, self.fibers):
            M += -f.stress(-kappa * y) * y * A
        return M


# ============================================================
# Bending cycle functions
# ============================================================
def bend_cycle(section, L, dmax, n=2000):
    """Load-unload cycle WITHOUT jig friction."""
    d_up = np.linspace(0, dmax, n + 1)
    P_up = np.array([4 * section.moment(12 * d / L**2) / L for d in d_up])

    d_dn, P_dn = [], []
    for d in np.linspace(dmax, 0, 3 * n + 1)[1:]:
        P = 4 * section.moment(12 * d / L**2) / L
        d_dn.append(d)
        P_dn.append(P)
        if P <= 0:
            if len(P_dn) >= 2 and P_dn[-2] > 0:
                t = P_dn[-2] / (P_dn[-2] - P)
                d_dn[-1] = d_dn[-2] + t * (d - d_dn[-2])
                P_dn[-1] = 0.0
            break
    return np.concatenate([d_up, d_dn]), np.concatenate([P_up, P_dn])


def bend_cycle_jig_friction(section, L, dmax, mu, r_s, r_l, n=2000):
    """Load-unload cycle WITH jig friction at support and loading rollers.

    Parameters
    ----------
    section : FiberSection
    L : float — span length [mm]
    dmax : float — maximum center deflection [mm]
    mu : float — friction coefficient between cable and jig surface
    r_s : float — support roller radius [mm]
    r_l : float — loading roller radius [mm]
    n : int — number of loading steps
    """
    correction = 2 * mu * (r_s + r_l)  # [mm]

    # Loading phase: P = 4M / (L - correction)
    d_up = np.linspace(0, dmax, n + 1)
    P_up = np.array([
        4 * section.moment(12 * d / L**2) / (L - correction)
        for d in d_up
    ])

    # Unloading phase: P = 4M / (L + correction)
    d_dn, P_dn = [], []
    for d in np.linspace(dmax, 0, 3 * n + 1)[1:]:
        M = section.moment(12 * d / L**2)
        P = 4 * M / (L + correction)
        d_dn.append(d)
        P_dn.append(P)
        if P <= 0:
            if len(P_dn) >= 2 and P_dn[-2] > 0:
                t = P_dn[-2] / (P_dn[-2] - P)
                d_dn[-1] = d_dn[-2] + t * (d - d_dn[-2])
                P_dn[-1] = 0.0
            break
    return np.concatenate([d_up, d_dn]), np.concatenate([P_up, P_dn])


# ============================================================
# Model factory
# ============================================================
def make_friction_model_factory(E_base, k_contact_total, N, degrade_ratio):
    """Create a factory for MultiLayerFriction models."""
    k_weights = np.linspace(0.5, 1.5, N)
    k_weights *= N / k_weights.sum()
    k_each = k_contact_total / N
    k_arr = k_each * k_weights
    eps_y_arr = np.logspace(np.log10(0.002), np.log10(0.30), N)
    f_y_arr = k_arr * eps_y_arr

    def factory():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_arr.copy(),
            k_degraded=(k_arr * degrade_ratio).copy(),
            f_y=f_y_arr.copy(),
        )
    return factory


def make_elastic_factory(E_total):
    """Create a factory for purely elastic (no friction) fibers."""
    def factory():
        return MultiLayerFriction(
            E_base=E_total,
            k_virgin=np.zeros(1),
            k_degraded=np.zeros(1),
            f_y=np.array([1e12]),
        )
    return factory


# ============================================================
# Plotting helpers
# ============================================================
def compute_slopes(d, P, dmax, d_ref=10.0):
    """Compute loading/unloading secant slopes and U/L ratio."""
    i_ref = min(np.searchsorted(d, d_ref), len(d) - 1)
    slope_L = (P[i_ref] - P[0]) / (d[i_ref] - d[0] + 1e-30)
    i_peak = np.argmax(d)
    i_unload = min(i_peak + np.searchsorted(-d[i_peak:], -(dmax - d_ref)), len(d) - 1)
    slope_U = (P[i_unload] - P[i_peak]) / (d[i_unload] - d[i_peak] + 1e-30) if i_unload > i_peak else 0
    ratio = abs(slope_U / slope_L) if slope_L else 0
    return slope_L, slope_U, ratio


# ============================================================
# Main
# ============================================================
def main():
    D, L, dmax = 17.0, 100.0, 30.0
    n_fiber = 60

    # Internal friction parameters (from 05)
    N_layers = 150
    E_base = 1500.0
    k_contact_total = 7500.0
    degrade_ratio = 0.25

    # Jig parameters
    r_s = 10.0   # support roller radius [mm]
    r_l = 10.0   # loading roller radius [mm]

    # ============================================================
    # (a) Jig friction only — elastic beam + jig friction
    # ============================================================
    print("=== (a) Jig friction only (elastic beam) ===")
    E_total = E_base + k_contact_total

    mu_values = [0.0, 0.1, 0.2, 0.3]
    jig_only_results = {}
    for mu in mu_values:
        sec = FiberSection(D, n_fiber=n_fiber)
        sec.setup(make_elastic_factory(E_total))
        if mu == 0.0:
            d, P = bend_cycle(sec, L, dmax, n=500)
        else:
            d, P = bend_cycle_jig_friction(sec, L, dmax, mu, r_s, r_l, n=500)
        jig_only_results[mu] = (d, P)
        corr = 2 * mu * (r_s + r_l)
        factor_L = L / (L - corr) if corr < L else float('inf')
        factor_U = L / (L + corr)
        print(f"  mu={mu:.1f}: correction={corr:.1f}mm, "
              f"P_load x{factor_L:.3f}, P_unload x{factor_U:.3f}")

    # ============================================================
    # (b) Internal friction only (no jig friction)
    # ============================================================
    print("\n=== (b) Internal friction only ===")
    sec_int = FiberSection(D, n_fiber=n_fiber)
    sec_int.setup(make_friction_model_factory(E_base, k_contact_total, N_layers, degrade_ratio))
    d_int, P_int = bend_cycle(sec_int, L, dmax, n=2000)
    sL, sU, ratio = compute_slopes(d_int, P_int, dmax)
    print(f"  Pmax={P_int.max()/1000:.1f}kN, d_res={d_int[-1]:.1f}mm, U/L={ratio:.2f}")

    # ============================================================
    # (c) Combined: internal friction + jig friction
    # ============================================================
    print("\n=== (c) Combined: internal + jig friction ===")
    combined_results = {}
    for mu in [0.0, 0.1, 0.2, 0.3]:
        sec = FiberSection(D, n_fiber=n_fiber)
        sec.setup(make_friction_model_factory(E_base, k_contact_total, N_layers, degrade_ratio))
        if mu == 0.0:
            d, P = bend_cycle(sec, L, dmax, n=2000)
        else:
            d, P = bend_cycle_jig_friction(sec, L, dmax, mu, r_s, r_l, n=2000)
        combined_results[mu] = (d, P)
        sL, sU, ratio = compute_slopes(d, P, dmax)
        print(f"  mu={mu:.1f}: Pmax={P.max()/1000:.1f}kN, d_res={d[-1]:.1f}mm, U/L={ratio:.2f}")

    # ============================================================
    # (d) Roller radius sensitivity (mu=0.2 fixed)
    # ============================================================
    print("\n=== (d) Roller radius sensitivity (mu=0.2) ===")
    mu_fixed = 0.2
    r_values = [5.0, 10.0, 20.0, 30.0]
    radius_results = {}
    for r in r_values:
        sec = FiberSection(D, n_fiber=n_fiber)
        sec.setup(make_friction_model_factory(E_base, k_contact_total, N_layers, degrade_ratio))
        d, P = bend_cycle_jig_friction(sec, L, dmax, mu_fixed, r, r, n=2000)
        radius_results[r] = (d, P)
        sL, sU, ratio = compute_slopes(d, P, dmax)
        print(f"  r={r:.0f}mm: Pmax={P.max()/1000:.1f}kN, d_res={d[-1]:.1f}mm, U/L={ratio:.2f}")

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- (a) Jig friction only (elastic beam) ---
    ax = axes[0, 0]
    colors_a = {0.0: 'gray', 0.1: 'blue', 0.2: 'orange', 0.3: 'red'}
    for mu in mu_values:
        d, P = jig_only_results[mu]
        ls = ':' if mu == 0.0 else '-'
        ax.plot(d, P / 1000, ls, color=colors_a[mu], lw=2,
                label=f'mu={mu:.1f}' + (' (elastic)' if mu == 0.0 else ''))
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(a) Jig friction only (elastic beam)\n'
                 f'r_s=r_l={r_s:.0f}mm — uniform load scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)

    # --- (b) Internal friction only vs combined ---
    ax = axes[0, 1]
    d0, P0 = combined_results[0.0]
    d2, P2 = combined_results[0.2]
    ax.plot(d0, P0 / 1000, 'b-', lw=2, label='Internal only (mu_jig=0)')
    ax.plot(d2, P2 / 1000, 'r-', lw=2, label='Internal + jig (mu=0.2)')
    ax.fill_between(d0, 0, P0 / 1000, alpha=0.04, color='blue')
    ax.fill_between(d2, 0, P2 / 1000, alpha=0.04, color='red')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(b) Internal friction vs combined\n'
                 'Jig friction widens loop uniformly')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)

    # --- (c) Combined with varying mu ---
    ax = axes[1, 0]
    colors_c = {0.0: 'gray', 0.1: 'blue', 0.2: 'orange', 0.3: 'red'}
    for mu in [0.0, 0.1, 0.2, 0.3]:
        d, P = combined_results[mu]
        ls = ':' if mu == 0.0 else '-'
        ax.plot(d, P / 1000, ls, color=colors_c[mu], lw=2, label=f'mu_jig={mu:.1f}')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(c) Combined: internal + jig friction\n'
                 f'r_s=r_l={r_s:.0f}mm, N={N_layers} layers, degrade={degrade_ratio}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)

    # --- (d) Roller radius sensitivity ---
    ax = axes[1, 1]
    colors_d = {5.0: 'green', 10.0: 'blue', 20.0: 'orange', 30.0: 'red'}
    for r in r_values:
        d, P = radius_results[r]
        ax.plot(d, P / 1000, '-', color=colors_d[r], lw=2,
                label=f'r={r:.0f}mm (corr={2*mu_fixed*2*r:.0f}mm)')
    # Add no-friction baseline
    d0, P0 = combined_results[0.0]
    ax.plot(d0, P0 / 1000, 'k:', lw=1.5, label='No jig friction')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(d) Roller radius sensitivity (mu={mu_fixed})\n'
                 'Larger rollers => wider loop')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = '/home/user/xkep-cae/work/beam_hysteresis/06_jig_friction.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out}")

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Jig friction effect on hysteresis")
    print("=" * 60)
    print(f"{'Model':<30} {'Pmax[kN]':>9} {'d_res[mm]':>9} {'U/L':>6}")
    print("-" * 56)
    for mu in [0.0, 0.1, 0.2, 0.3]:
        d, P = combined_results[mu]
        sL, sU, ratio = compute_slopes(d, P, dmax)
        label = f"internal + jig mu={mu:.1f}"
        print(f"  {label:<28} {P.max()/1000:>8.1f} {d[-1]:>9.1f} {ratio:>6.2f}")
    print("-" * 56)
    print(f"\nJig: r_s=r_l={r_s:.0f}mm")
    print(f"Internal: N={N_layers}, E_base={E_base}, k_contact={k_contact_total}, "
          f"degrade={degrade_ratio}")
    print(f"\nJig friction creates UNIFORM (multiplicative) hysteresis:")
    print(f"  P_load  = 4M / (L - 2*mu*(r_s+r_l))")
    print(f"  P_unload = 4M / (L + 2*mu*(r_s+r_l))")
    print(f"  => Load-proportional widening, independent of strain history")


if __name__ == '__main__':
    main()
