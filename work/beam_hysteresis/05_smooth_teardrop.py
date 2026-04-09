"""
05: Smooth teardrop — high-resolution hysteresis
=================================================

Goal: truly smooth, rounded teardrop shape.
  - Upper-right: fat, rounded (elliptical)
  - Lower-left: thin, pointed tail near origin
  - No angular/linear segments

Improvements over 04:
  1. N=150 friction elements (was 40)
  2. Higher resolution (n=2000 loading points)
  3. Graduated stiffness: k_i varies across elements
     (outer contact layers softer, inner layers stiffer)
  4. EI ratio = 6x for wider loop

[<- README](../../README.md)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


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


def bend_cycle(section, L, dmax, n=2000):
    """Load to dmax, then unload until P<=0."""
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


def main():
    D, L, dmax = 17.0, 100.0, 30.0
    R = D / 2.0
    eps_max = 12 * dmax / L**2 * R  # 0.306

    # ============================================================
    # Model parameters
    # ============================================================
    # EI_max/EI_min ratio = 6x
    # E_base represents individual strand bending (EI_min)
    # k_contact represents inter-strand friction lock (EI_max - EI_min)
    N = 150  # many elements => smooth progressive slip
    E_base = 1500.0
    k_contact_total = 7500.0  # Total: 1500+7500=9000, ratio=6x

    # Graduated stiffness: elements have different k values
    # Outer contact layers (low threshold) are softer
    # Inner contact layers (high threshold) are stiffer
    # This creates more curvature in the transition region
    k_weights = np.linspace(0.5, 1.5, N)  # outer soft, inner stiff
    k_weights *= N / k_weights.sum()  # normalize so sum = N
    k_each = k_contact_total / N
    k_arr = k_each * k_weights

    # Log-spaced yield strains: 0.002 to 0.30
    # Very fine spacing ensures smooth progressive slip
    eps_y_arr = np.logspace(np.log10(0.002), np.log10(0.30), N)
    f_y_arr = k_arr * eps_y_arr

    # Degradation ratio
    degrade_ratio = 0.25

    def make_model():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_arr.copy(),
            k_degraded=(k_arr * degrade_ratio).copy(),
            f_y=f_y_arr.copy(),
        )

    # --- Main model ---
    n_fiber = 60
    sec = FiberSection(D, n_fiber=n_fiber)
    sec.setup(make_model)
    d, P = bend_cycle(sec, L, dmax, n=2000)

    # --- Elastic reference ---
    sec_el = FiberSection(D, n_fiber=n_fiber)
    sec_el.setup(lambda: MultiLayerFriction(
        E_base=E_base + k_contact_total,
        k_virgin=np.zeros(1), k_degraded=np.zeros(1), f_y=np.array([1e12]),
    ))
    d_el, P_el = bend_cycle(sec_el, L, dmax, n=500)

    # --- Comparison: degrade=0.1 (stronger) ---
    def make_model_strong():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_arr.copy(),
            k_degraded=(k_arr * 0.10).copy(),
            f_y=f_y_arr.copy(),
        )
    sec2 = FiberSection(D, n_fiber=n_fiber)
    sec2.setup(make_model_strong)
    d2, P2 = bend_cycle(sec2, L, dmax, n=2000)

    # --- No degradation ---
    def make_model_nodeg():
        return MultiLayerFriction(
            E_base=E_base,
            k_virgin=k_arr.copy(),
            k_degraded=k_arr.copy(),
            f_y=f_y_arr.copy(),
        )
    sec3 = FiberSection(D, n_fiber=n_fiber)
    sec3.setup(make_model_nodeg)
    d3, P3 = bend_cycle(sec3, L, dmax, n=2000)

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Main result: smooth teardrop
    ax = axes[0, 0]
    ax.plot(d, P/1000, 'r-', lw=2.5, label=f'N={N}, degrade={degrade_ratio}')
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3, label='Elastic (all stuck)')
    ax.fill_between(d, 0, P/1000, alpha=0.06, color='red')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(a) Smooth teardrop (N={N}, EI ratio={int((E_base+k_contact_total)/E_base)}x)\n'
                 f'Residual: {d[-1]:.1f}mm ({d[-1]/dmax*100:.0f}%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (b) Comparison of degradation levels
    ax = axes[0, 1]
    ax.plot(d3, P3/1000, 'b-', lw=2, label='No degradation')
    ax.plot(d, P/1000, 'r-', lw=2, label=f'Degrade x{degrade_ratio}')
    ax.plot(d2, P2/1000, 'g-', lw=2, label='Degrade x0.1')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(b) Effect of degradation ratio')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (c) Normalized shape
    ax = axes[1, 0]
    for label, dd, PP, c in [
        ('No degrade', d3, P3, 'blue'),
        (f'Degrade x{degrade_ratio}', d, P, 'red'),
        ('Degrade x0.1', d2, P2, 'green'),
    ]:
        Pm = PP.max()
        if Pm > 0:
            ax.plot(dd/dmax, PP/Pm, color=c, lw=2, label=label)
    ax.set_xlabel('delta / delta_max')
    ax.set_ylabel('P / P_max')
    ax.set_title('(c) Normalized shape')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Summary table
    ax = axes[1, 1]
    ax.axis('off')

    lines = [
        'SMOOTH TEARDROP MODEL (Stage 05)',
        '=' * 44,
        f'D={D}mm, L={L}mm, dmax={dmax}mm',
        f'N={N} friction elements (graduated k)',
        f'n_fiber={n_fiber}',
        f'E_base={E_base:.0f} (EI_min part)',
        f'k_contact={k_contact_total:.0f} total',
        f'EI_max/EI_min = {(E_base+k_contact_total)/E_base:.0f}x',
        f'eps_y: {eps_y_arr[0]:.4f} to {eps_y_arr[-1]:.3f}',
        f'eps_max = {eps_max:.3f}',
        '',
    ]

    lines.append(f'{"Model":<18} {"Pmax":>6} {"d_res":>6} {"SlopeL":>7} {"SlopeU":>7} {"U/L":>5}')
    lines.append('-' * 52)

    for name, dd, PP in [
        ('No degrade', d3, P3),
        (f'Deg={degrade_ratio}', d, P),
        ('Deg=0.1', d2, P2),
    ]:
        i10 = min(np.searchsorted(dd, 10.0), len(dd)-1)
        sL = (PP[i10]-PP[0]) / (dd[i10]-dd[0]+1e-30)
        ip = np.argmax(dd)
        ip10 = min(ip + np.searchsorted(-dd[ip:], -(dmax-10)), len(dd)-1)
        sU = (PP[ip10]-PP[ip]) / (dd[ip10]-dd[ip]+1e-30) if ip10>ip else 0
        ratio = abs(sU/sL) if sL else 0
        lines.append(f'{name:<18} {PP.max()/1000:>5.1f}k {dd[-1]:>6.1f} {sL:>7.0f} {sU:>7.0f} {ratio:>5.2f}')

    lines.extend([
        '',
        '=' * 44,
        'Key improvements for smoothness:',
        f'  1. N={N} elements (was 40)',
        '     => very smooth progressive slip',
        '  2. Graduated k (outer soft, inner stiff)',
        '     => rounded transition curves',
        '  3. eps_y 0.002-0.30 (wide spread)',
        '     => early onset + late completion',
        '  4. 60 fibers x 2000 load steps',
        '     => high resolution',
    ])

    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = '/home/user/xkep-cae/work/beam_hysteresis/05_smooth_teardrop.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved: {out}")

    # Console summary
    for name, dd, PP in [
        ('No degrade', d3, P3),
        (f'Deg={degrade_ratio}', d, P),
        ('Deg=0.1', d2, P2),
    ]:
        i10 = min(np.searchsorted(dd, 10.0), len(dd)-1)
        sL = (PP[i10]-PP[0]) / (dd[i10]-dd[0]+1e-30)
        ip = np.argmax(dd)
        ip10 = min(ip + np.searchsorted(-dd[ip:], -(dmax-10)), len(dd)-1)
        sU = (PP[ip10]-PP[ip]) / (dd[ip10]-dd[ip]+1e-30) if ip10>ip else 0
        ratio = abs(sU/sL) if sL else 0
        print(f"  {name:<18} Pmax={PP.max()/1000:.1f}kN d_res={dd[-1]:.1f}mm U/L={ratio:.2f}")


if __name__ == '__main__':
    main()
