"""
Beam bending hysteresis — smooth teardrop shape
================================================

Target shape (real cable):
  - Right-upper: fat, rounded
  - Left-lower: thin, pointed (teardrop tail at origin)
  - Smooth curves, no straight segments
  - Loading slope > unloading slope

Approach:
  Many (50+) friction elements with log-spaced thresholds
  + contact stiffness degradation
  => smooth progressive slip/re-engagement
  => rounded teardrop

[<- README](../../README.md)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ============================================================
# Multi-layer friction with degradation
# ============================================================
@dataclass
class MultiLayerFriction:
    """N parallel friction elements with stiffness degradation.

    sigma = E_base * eps + sum_i k_i(state) * (eps - slip_i)

    Each element: k_virgin -> k_degraded after first slip.
    Many elements with spread thresholds => smooth curve.
    """
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
                    excess = abs(trial) - self.f_y[i]
                    self.slip[i] += s * excess / k
                    sigma += s * self.f_y[i]
        return sigma


# ============================================================
# Fiber section
# ============================================================
class FiberSection:
    def __init__(self, diameter, n_fiber=50):
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
        yl, yh = max(y-dy/2, -R), min(y+dy/2, R)
        if yh <= yl: return 0.0
        wl = 2*np.sqrt(max(R**2-yl**2, 0))
        wh = 2*np.sqrt(max(R**2-yh**2, 0))
        return (wl+wh)/2 * (yh-yl)

    def moment(self, kappa):
        M = 0.0
        for y, A, f in zip(self.y, self.A, self.fibers):
            M += -f.stress(-kappa * y) * y * A
        return M


def bend_cycle(section, L, dmax, n=800):
    d_up = np.linspace(0, dmax, n+1)
    P_up = np.array([4 * section.moment(12*d/L**2) / L for d in d_up])

    d_dn, P_dn = [], []
    for d in np.linspace(dmax, 0, 3*n+1)[1:]:
        P = 4 * section.moment(12*d/L**2) / L
        d_dn.append(d); P_dn.append(P)
        if P <= 0:
            if len(P_dn) >= 2 and P_dn[-2] > 0:
                t = P_dn[-2] / (P_dn[-2] - P)
                d_dn[-1] = d_dn[-2] + t * (d - d_dn[-2])
                P_dn[-1] = 0.0
            break
    return np.concatenate([d_up, d_dn]), np.concatenate([P_up, P_dn])


# ============================================================
# Main
# ============================================================
def main():
    D, L, dmax = 17.0, 100.0, 30.0
    R = D / 2.0
    eps_max = 12 * dmax / L**2 * R  # 0.306

    # ============================================================
    # Build multi-layer friction elements
    # ============================================================
    # Key constraint: element i yields when |eps| > f_y_i / k_i
    # eps_max at surface = 0.306
    # Want lowest f_y element to slip at eps ~ 0.01 (very early)
    # Want highest f_y element to slip at eps ~ 0.25 (near peak)
    # => f_y range = k_each * [0.01, 0.25] * eps_max-ish

    N = 30
    E_base = 3000.0     # individual strand bending (always elastic)
    k_contact_total = 7000.0  # inter-strand friction stiffness
    k_each = k_contact_total / N  # ~233 per element

    # Yield thresholds: log-spaced so slip is progressive
    # eps_yield_i = f_y_i / k_each
    # Want eps_yield from 0.005 to 0.25
    eps_y_min = 0.005
    eps_y_max = 0.25
    f_y_arr = k_each * np.logspace(np.log10(eps_y_min), np.log10(eps_y_max), N)

    degrade_ratio = 0.25  # contact stiffness drops to 25% after slip

    k_v = np.full(N, k_each)
    k_d = np.full(N, k_each * degrade_ratio)

    def make_model():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_v.copy(),
            k_degraded=k_d.copy(),
            f_y=f_y_arr.copy(),
        )

    # --- A: With degradation (target model) ---
    secA = FiberSection(D, n_fiber=40)
    secA.setup(make_model)
    dA, PA = bend_cycle(secA, L, dmax)

    # --- B: Without degradation (for comparison) ---
    def make_model_nodeg():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_v.copy(),
            k_degraded=k_v.copy(),  # no degradation
            f_y=f_y_arr.copy(),
        )
    secB = FiberSection(D, n_fiber=40)
    secB.setup(make_model_nodeg)
    dB, PB = bend_cycle(secB, L, dmax)

    # --- C: Steeper degradation (0.1x) ---
    k_d2 = np.full(N, k_each * 0.1)
    def make_model_steep():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_v.copy(),
            k_degraded=k_d2.copy(),
            f_y=f_y_arr.copy(),
        )
    secC = FiberSection(D, n_fiber=40)
    secC.setup(make_model_steep)
    dC, PC = bend_cycle(secC, L, dmax)

    # --- Elastic ---
    secE = FiberSection(D, n_fiber=40)
    secE.setup(lambda: MultiLayerFriction(
        E_base=10000.0,
        k_virgin=np.zeros(1), k_degraded=np.zeros(1), f_y=np.array([1e12]),
    ))
    dE, PE = bend_cycle(secE, L, dmax)

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Main result: smooth teardrop
    ax = axes[0, 0]
    ax.plot(dA, PA/1000, 'r-', lw=2.5, label=f'50-layer + degradation (x{degrade_ratio})')
    ax.plot(dE, PE/1000, 'k:', lw=1, alpha=0.3, label='Elastic')
    ax.fill_between(dA, 0, PA/1000, alpha=0.06, color='red')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(a) 50-layer friction + degradation\n'
                 f'Residual: {dA[-1]:.1f}mm ({dA[-1]/dmax*100:.0f}%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (b) With vs without degradation
    ax = axes[0, 1]
    ax.plot(dB, PB/1000, 'b-', lw=2, label='No degradation (same slopes)')
    ax.plot(dA, PA/1000, 'r-', lw=2, label=f'Degradation x{degrade_ratio} (asym. slopes)')
    ax.plot(dC, PC/1000, 'g-', lw=2, label='Degradation x0.1 (strong)')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(b) Effect of contact stiffness degradation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (c) Normalized shape
    ax = axes[1, 0]
    for label, d, P, c in [
        ('No degrade', dB, PB, 'blue'),
        (f'Degrade x{degrade_ratio}', dA, PA, 'red'),
        ('Degrade x0.1', dC, PC, 'green'),
    ]:
        Pm = P.max()
        if Pm > 0:
            ax.plot(d/dmax, P/Pm, color=c, lw=2, label=label)
    ax.set_xlabel('delta / delta_max')
    ax.set_ylabel('P / P_max')
    ax.set_title('(c) Normalized shape\n(teardrop: fat right-top, thin left-bottom)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Summary
    ax = axes[1, 1]
    ax.axis('off')

    # Compute slopes
    slopes = {}
    for name, d, P in [('No degrade', dB, PB), ('Degrade 0.3', dA, PA), ('Degrade 0.1', dC, PC)]:
        i10 = np.searchsorted(d, 10.0)
        sL = (P[i10] - P[0]) / (d[i10] - d[0] + 1e-30)
        ip = np.argmax(d)
        ip10 = ip + np.searchsorted(-d[ip:], -(dmax - 10.0))
        ip10 = min(ip10, len(d)-1)
        sU = (P[ip10] - P[ip]) / (d[ip10] - d[ip] + 1e-30)
        slopes[name] = (sL, sU, abs(sU/sL) if sL else 0, d[-1])

    text = (
        'SMOOTH TEARDROP MODEL\n'
        '=' * 42 + '\n\n'
        f'50 friction elements, log-spaced thresholds\n'
        f'f_y: {f_y_arr[0]:.1f} to {f_y_arr[-1]:.1f} MPa\n'
        f'E_base: {E_base:.0f} MPa\n'
        f'k_contact: {k_contact_total:.0f} MPa total\n\n'
        'Slope analysis (secant 0-10mm):\n'
        f'{"Model":<15} {"S_load":>7} {"S_unld":>7} {"U/L":>5} {"d_res":>5}\n'
        f'{"-"*40}\n'
    )
    for name, (sL, sU, ratio, dres) in slopes.items():
        text += f'{name:<15} {sL:>7.0f} {sU:>7.0f} {ratio:>5.2f} {dres:>5.1f}\n'
    text += (
        f'\n{"="*42}\n'
        'Smooth teardrop comes from:\n'
        '  1. Many friction layers (N=50)\n'
        '     => smooth progressive slip\n'
        '     => rounded curves (no kinks)\n\n'
        '  2. Log-spaced thresholds\n'
        '     => outer layers slip at low kappa\n'
        '     => core layers slip at high kappa\n'
        '     => gradual stiffness reduction\n\n'
        '  3. Contact stiffness degradation\n'
        '     => loading uses virgin k (stiff)\n'
        '     => unloading uses degraded k (soft)\n'
        '     => asymmetric slopes\n\n'
        '  4. Fiber section (curvature effect)\n'
        '     => outer fibers see more strain\n'
        '     => fat loop at high delta\n'
        '     => thin tail near origin'
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = '/home/user/xkep-cae/work/beam_hysteresis/hysteresis_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved: {out}")

    for name, (sL, sU, ratio, dres) in slopes.items():
        print(f"  {name:<15} SlopeL={sL:.0f} SlopeU={sU:.0f} U/L={ratio:.2f} d_res={dres:.1f}mm")


if __name__ == '__main__':
    main()
