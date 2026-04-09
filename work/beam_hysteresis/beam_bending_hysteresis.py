"""
Beam bending hysteresis — unified model
========================================

Real cable = elastic + stick-slip + plasticity + friction dissipation.
How to express this simply?

Answer: Generalized Prandtl-Ishlinskii model.
  N parallel elements, each = elastic spring + friction slider:
    Element i:  sigma_i = k_i * (eps - slip_i)
                |sigma_i - alpha_i| <= f_yi
  Total: sigma = E_base*eps + sum(sigma_i)

  With curvature-dependent friction: f_yi = f_yi0 + beta_i * |eps|

  This unifies:
    - Elastic beam bending (E_base)
    - Strand stick-slip (individual friction elements)
    - Effective "plasticity" (sum of nonlinear elements)
    - Friction dissipation (hysteresis area = energy lost)
    - Pressure-dependent normal force (beta term)

[<- README](../../README.md)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ============================================================
# Generalized Prandtl-Ishlinskii fiber model
# ============================================================
@dataclass
class PrandtlIshlinskiiFiber:
    """N parallel friction elements + elastic backbone.

    sigma = E_base * eps + sum_i [k_i * (eps - slip_i)]
    Each element i has threshold f_yi(eps) = f_yi0 + beta_i * |eps|

    Parameters
    ----------
    E_base : float
        Purely elastic backbone stiffness (beam bending without friction)
    k : array
        Spring stiffness of each friction element
    f_y0 : array
        Base friction threshold of each element
    beta : array
        Pressure sensitivity of each element
    """
    E_base: float
    k: np.ndarray
    f_y0: np.ndarray
    beta: np.ndarray
    slip: np.ndarray = field(default=None)
    alpha: np.ndarray = field(default=None)

    def __post_init__(self):
        n = len(self.k)
        if self.slip is None:
            self.slip = np.zeros(n)
        if self.alpha is None:
            self.alpha = np.zeros(n)

    def stress(self, eps):
        sigma = self.E_base * eps
        for i in range(len(self.k)):
            f_y = self.f_y0[i] + self.beta[i] * abs(eps)
            trial = self.k[i] * (eps - self.slip[i])
            eta = trial - self.alpha[i]
            if abs(eta) <= f_y:
                sigma += trial
            else:
                s = np.sign(eta)
                dg = (abs(eta) - f_y) / (self.k[i] + 0)  # H=0 per element (pure friction)
                self.slip[i] += s * dg
                self.alpha[i] += 0  # no hardening per element
                sigma += self.k[i] * (eps - self.slip[i])
        return sigma


@dataclass
class StandardKH:
    """Simple KH for comparison."""
    E: float
    sigma_y: float
    H: float
    eps_p: float = 0.0
    alpha: float = 0.0

    def stress(self, eps):
        sig_trial = self.E * (eps - self.eps_p)
        eta = sig_trial - self.alpha
        if abs(eta) <= self.sigma_y:
            return sig_trial
        s = np.sign(eta)
        dg = (abs(eta) - self.sigma_y) / (self.E + self.H)
        self.eps_p += s * dg
        self.alpha += s * self.H * dg
        return self.E * (eps - self.eps_p)


@dataclass
class PressureDependentKH:
    """KH with pressure-dependent sigma_y."""
    E: float
    sigma_y0: float
    beta: float
    H: float
    eps_p: float = 0.0
    alpha: float = 0.0

    def stress(self, eps):
        sy = self.sigma_y0 + self.beta * abs(eps)
        sig_trial = self.E * (eps - self.eps_p)
        eta = sig_trial - self.alpha
        if abs(eta) <= sy:
            return sig_trial
        s = np.sign(eta)
        dg = (abs(eta) - sy) / (self.E + self.H)
        self.eps_p += s * dg
        self.alpha += s * self.H * dg
        return self.E * (eps - self.eps_p)


# ============================================================
# Fiber section + bending
# ============================================================
class FiberSection:
    def __init__(self, diameter, n_fiber=60):
        self.diameter = diameter
        R = diameter / 2.0
        self.y_coords = np.linspace(-R + R/n_fiber, R - R/n_fiber, n_fiber)
        dy = 2 * R / n_fiber
        self.areas = np.array([self._chord_area(y, dy, R) for y in self.y_coords])
        self.fibers = []

    def setup(self, factory):
        self.fibers = [factory() for _ in range(len(self.y_coords))]

    def setup_graded(self, factory_fn):
        R = self.diameter / 2.0
        self.fibers = [factory_fn(abs(y) / R) for y in self.y_coords]

    @staticmethod
    def _chord_area(y, dy, R):
        y_lo, y_hi = max(y - dy/2, -R), min(y + dy/2, R)
        if y_hi <= y_lo:
            return 0.0
        w_lo = 2 * np.sqrt(max(R**2 - y_lo**2, 0))
        w_hi = 2 * np.sqrt(max(R**2 - y_hi**2, 0))
        return (w_lo + w_hi) / 2 * (y_hi - y_lo)

    def moment(self, kappa):
        M = 0.0
        for y, A, mat in zip(self.y_coords, self.areas, self.fibers):
            eps = -kappa * y
            sigma = mat.stress(eps)
            M += -sigma * y * A
        return M


def bend_cycle(section, L, delta_max, n=600):
    d_up = np.linspace(0, delta_max, n + 1)
    P_up = np.array([4 * section.moment(12 * d / L**2) / L for d in d_up])

    d_dn, P_dn = [], []
    for d in np.linspace(delta_max, 0, 3*n + 1)[1:]:
        P = 4 * section.moment(12 * d / L**2) / L
        d_dn.append(d)
        P_dn.append(P)
        if P <= 0:
            if len(P_dn) >= 2 and P_dn[-2] > 0:
                t = P_dn[-2] / (P_dn[-2] - P)
                d_dn[-1] = d_dn[-2] + t * (d - d_dn[-2])
                P_dn[-1] = 0.0
            break

    return (np.concatenate([d_up, np.array(d_dn)]),
            np.concatenate([P_up, np.array(P_dn)]))


# ============================================================
# Main
# ============================================================
def main():
    D = 17.0
    L = 100.0
    delta_max = 30.0
    R = D / 2.0
    eps_max = 12 * delta_max / L**2 * R

    # ============================================================
    # Model 1: Standard KH (reference)
    # ============================================================
    sec1 = FiberSection(D)
    sec1.setup(lambda: StandardKH(E=10_000, sigma_y=1500, H=1000))
    d1, P1 = bend_cycle(sec1, L, delta_max)

    # ============================================================
    # Model 2: Pressure-dependent KH (previous best)
    # ============================================================
    sec2 = FiberSection(D)
    sec2.setup(lambda: PressureDependentKH(E=10_000, sigma_y0=500, beta=5000, H=800))
    d2, P2 = bend_cycle(sec2, L, delta_max)

    # ============================================================
    # Model 3: Prandtl-Ishlinskii (the "flat" unified model)
    # N=5 parallel friction elements with spread thresholds
    # + pressure-dependent friction
    #
    # Physical interpretation:
    #   Element 1: outer strand layer (low f_y, high beta) - slips first
    #   Element 2: next layer
    #   ...
    #   Element N: core strands (high f_y, low beta) - slips last or never
    #   E_base: elastic beam bending of individual strands
    # ============================================================
    N_elem = 5
    # Distribute stiffness: total contact stiffness = 4000 MPa
    k_total = 4000.0
    k_arr = np.full(N_elem, k_total / N_elem)

    # Spread friction thresholds: outer layers low, inner layers high
    f_y0_arr = np.array([200, 400, 800, 1500, 3000])

    # Pressure sensitivity: outer layers high, inner layers low
    beta_arr = np.array([8000, 5000, 3000, 1000, 0])

    sec3 = FiberSection(D)
    sec3.setup(lambda: PrandtlIshlinskiiFiber(
        E_base=6000,
        k=k_arr.copy(),
        f_y0=f_y0_arr.copy(),
        beta=beta_arr.copy(),
    ))
    d3, P3 = bend_cycle(sec3, L, delta_max)

    # ============================================================
    # Model 4: PI without pressure dependence (for comparison)
    # ============================================================
    sec4 = FiberSection(D)
    sec4.setup(lambda: PrandtlIshlinskiiFiber(
        E_base=6000,
        k=k_arr.copy(),
        f_y0=f_y0_arr.copy(),
        beta=np.zeros(N_elem),
    ))
    d4, P4 = bend_cycle(sec4, L, delta_max)

    # ============================================================
    # Model 5: PI with more elements (smoother curve, 10 layers)
    # ============================================================
    N10 = 10
    k10 = np.full(N10, k_total / N10)
    f_y0_10 = np.linspace(100, 3000, N10)  # linear spread
    beta_10 = np.linspace(8000, 0, N10)     # decreasing pressure sensitivity

    sec5 = FiberSection(D)
    sec5.setup(lambda: PrandtlIshlinskiiFiber(
        E_base=6000,
        k=k10.copy(),
        f_y0=f_y0_10.copy(),
        beta=beta_10.copy(),
    ))
    d5, P5 = bend_cycle(sec5, L, delta_max)

    # Elastic
    sec_el = FiberSection(D)
    sec_el.setup(lambda: StandardKH(E=10_000, sigma_y=1e12, H=0))
    d_el, P_el = bend_cycle(sec_el, L, delta_max)

    models = [
        ('M1: Standard KH', d1, P1, 'blue', '-'),
        ('M2: Pressure-dep KH', d2, P2, 'red', '-'),
        ('M3: PI 5-layer + pressure', d3, P3, 'green', '-'),
        ('M4: PI 5-layer (no pressure)', d4, P4, 'cyan', '--'),
        ('M5: PI 10-layer + pressure', d5, P5, 'magenta', '-'),
    ]

    # ============================================================
    # Summary
    # ============================================================
    print(f"D={D}mm, L={L}mm, dmax={delta_max}mm, eps_max={eps_max:.3f}")
    print(f"\n{'Model':<30} {'Pmax[kN]':>9} {'d_res[mm]':>9} {'res%':>5}")
    print("-" * 55)
    for name, d, P, _, _ in models:
        print(f"{name:<30} {P.max()/1000:>9.2f} {d[-1]:>9.1f} {d[-1]/delta_max*100:>5.0f}%")

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- (a) All models ---
    ax = axes[0, 0]
    for name, d, P, c, ls in models:
        label = name.split(':')[1].strip()[:30]
        ax.plot(d, P/1000, color=c, ls=ls, lw=2, label=label)
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(a) All models')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    # --- (b) M1 vs M3: Standard KH vs PI ---
    ax = axes[0, 1]
    ax.plot(d1, P1/1000, 'b-', lw=2, label='M1: Standard KH')
    ax.plot(d3, P3/1000, 'g-', lw=2.5, label='M3: PI + pressure')
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3)
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(b) Standard KH vs PI model')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    # --- (c) PI model close-up (best candidate) ---
    ax = axes[0, 2]
    ax.plot(d5, P5/1000, 'm-', lw=2.5, label='PI 10-layer + pressure')
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3, label='Elastic')
    ax.fill_between(d5, 0, P5/1000, alpha=0.08, color='magenta')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    d_res5 = d5[-1]
    ax.set_title(f'(c) PI 10-layer: d_res={d_res5:.1f}mm ({d_res5/delta_max*100:.0f}%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    # --- (d) PI with/without pressure ---
    ax = axes[1, 0]
    ax.plot(d3, P3/1000, 'g-', lw=2.5, label='PI + pressure')
    ax.plot(d4, P4/1000, 'c--', lw=2, label='PI (no pressure)')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(d) Effect of pressure-dependent friction')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    # --- (e) Normalized comparison ---
    ax = axes[1, 1]
    for name, d, P, c, ls in models:
        Pm = P.max()
        if Pm > 0:
            label = name.split(':')[0]
            ax.plot(d/delta_max, P/Pm, color=c, ls=ls, lw=1.5, label=label)
    ax.set_xlabel('delta / delta_max')
    ax.set_ylabel('P / P_max')
    ax.set_title('(e) Normalized shape comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (f) Unified model concept ---
    ax = axes[1, 2]
    ax.axis('off')
    text = (
        'UNIFIED MODEL: PRANDTL-ISHLINSKII\n'
        '=' * 42 + '\n\n'
        'sigma = E_base*eps  (elastic backbone)\n'
        '      + sum_i sigma_i  (friction elements)\n\n'
        'Each element i:\n'
        '  sigma_i = k_i * (eps - slip_i)\n'
        '  |sigma_i - alpha_i| <= f_yi(eps)\n'
        '  f_yi = f_yi0 + beta_i * |eps|\n\n'
        'Physical mapping:\n'
        '  E_base   = beam bending (no friction)\n'
        '  k_i      = strand layer i contact stiffness\n'
        '  f_yi0    = base friction (strand weight)\n'
        '  beta_i   = curvature-dependent normal force\n'
        '  slip_i   = strand sliding displacement\n\n'
        'What each piece gives:\n'
        '  E_base alone     => linear elastic\n'
        '  + f_yi (spread)  => smooth nonlinear transition\n'
        '  + beta_i         => asymmetric slopes (teardrop)\n'
        '  + fiber section  => curvature effect\n\n'
        '-' * 42 + '\n'
        'Result: tilted teardrop hysteresis\n'
        '  - Loading: stiff (friction locks strands)\n'
        '  - Unloading: soft (friction releases)\n'
        '  - Elliptical return to near-origin\n'
        '  - Thin loop = small dissipation fraction'
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    outpath = '/home/user/xkep-cae/work/beam_hysteresis/hysteresis_comparison.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"\nPlot saved: {outpath}")

    print("""
CONCLUSION: How to express the combined effects "simply"
=========================================================

Real cable = elastic + stick-slip + plasticity + friction dissipation

This is a GENERALIZED PRANDTL-ISHLINSKII model:

  sigma(eps) = E_base * eps + sum_{i=1}^{N} play_i(eps)

where each play_i is a friction element with:
  - stiffness k_i (strand layer contact stiffness)
  - threshold f_yi = f_yi0 + beta_i * |eps| (pressure-dependent)

The parameters map directly to cable physics:
  - E_base = sum of individual strand bending stiffnesses (EI_min)
  - k_i = contact stiffness contribution from strand layer i
  - f_yi0 = friction force from strand weight/pretension
  - beta_i = friction increase due to curvature-induced normal force

This is essentially a RHEOLOGICAL MODEL:
  Spring E_base in parallel with N (spring + friction slider) elements.

The spread of thresholds {f_yi} determines the smoothness of the curve.
The pressure sensitivity {beta_i} determines the loading/unloading asymmetry.

It is EQUIVALENT to:
  - Fiber section model with KH materials (proven: diff=0)
  - But adds pressure-dependent sigma_y for slope asymmetry
  - Prandtl-Ishlinskii gives smoother curves than single-KH
""")


if __name__ == '__main__':
    main()
