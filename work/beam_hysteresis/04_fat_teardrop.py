"""
04: Fat teardrop — visible hysteresis loop
==========================================

Previous models had thin loops because E_base dominated.
Fix: make friction contribution LARGE relative to elastic backbone.

Physical picture:
  Cable EI_max / EI_min ratio can be 3-10x.
  EI_max = all strands stuck (composite action)
  EI_min = all strands slipping (individual bending only)

  So E_base (= EI_min part) should be SMALL compared to
  total E (= EI_max), and the friction elements carry most
  of the stiffness when stuck.

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
    def __init__(self, diameter, n_fiber=40):
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


def main():
    D, L, dmax = 17.0, 100.0, 30.0
    R = D / 2.0
    eps_max = 12 * dmax / L**2 * R  # 0.306

    # ============================================================
    # Key: EI_max/EI_min ratio ~ 5x
    # E_base = 2000 (individual strand bending, = EI_min contribution)
    # k_contact_total = 8000 (friction lock contribution)
    # Total when stuck: 2000 + 8000 = 10000 (= EI_max)
    # When all slipped: ~ 2000 (= EI_min)
    # Ratio: 5x — realistic for stranded conductors
    # ============================================================
    N = 40
    E_base = 2000.0
    k_contact_total = 8000.0
    k_each = k_contact_total / N  # 200 per element

    # Log-spaced yield strains: 0.003 to 0.20
    # At delta=1mm: kappa=0.0012, eps_surface=0.010 -> elements with eps_y<0.01 slip
    # At delta=15mm: kappa=0.018, eps_surface=0.153 -> ~80% of elements have slipped
    # At delta=30mm: eps_surface=0.306 -> all elements have slipped
    eps_y_arr = np.logspace(np.log10(0.003), np.log10(0.20), N)
    f_y_arr = k_each * eps_y_arr

    degrade_ratios = [1.0, 0.3, 0.1]
    results = {}

    for dr in degrade_ratios:
        k_v = np.full(N, k_each)
        k_d = np.full(N, k_each * dr)

        sec = FiberSection(D, n_fiber=40)
        sec.setup(lambda dr=dr: MultiLayerFriction(
            E_base=E_base,
            k_virgin=np.full(N, k_each),
            k_degraded=np.full(N, k_each * dr),
            f_y=f_y_arr.copy(),
        ))
        d, P = bend_cycle(sec, L, dmax)
        results[dr] = (d, P)

    # Elastic reference
    sec_el = FiberSection(D, n_fiber=40)
    sec_el.setup(lambda: MultiLayerFriction(
        E_base=E_base + k_contact_total,
        k_virgin=np.zeros(1), k_degraded=np.zeros(1), f_y=np.array([1e12]),
    ))
    d_el, P_el = bend_cycle(sec_el, L, dmax)

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) All hysteresis loops
    ax = axes[0, 0]
    colors = {1.0: 'blue', 0.3: 'red', 0.1: 'green'}
    for dr, (d, P) in results.items():
        label = f'degrade={dr}' if dr < 1.0 else 'no degradation'
        ax.plot(d, P/1000, color=colors[dr], lw=2, label=label)
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3, label='Elastic (all stuck)')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(a) EI_max/EI_min = {(E_base+k_contact_total)/E_base:.0f}x')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (b) Best candidate close-up (degrade=0.3)
    d03, P03 = results[0.3]
    ax = axes[0, 1]
    ax.plot(d03, P03/1000, 'r-', lw=2.5)
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3, label='Elastic')
    ax.fill_between(d03, 0, P03/1000, alpha=0.08, color='red')
    d_res = d03[-1]
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(b) degrade=0.3, d_res={d_res:.1f}mm ({d_res/dmax*100:.0f}%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (c) Normalized
    ax = axes[1, 0]
    for dr, (d, P) in results.items():
        Pm = P.max()
        label = f'degrade={dr}' if dr < 1.0 else 'no degradation'
        ax.plot(d/dmax, P/Pm, color=colors[dr], lw=2, label=label)
    ax.set_xlabel('delta / delta_max')
    ax.set_ylabel('P / P_max')
    ax.set_title('(c) Normalized shape')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Summary
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        f'PARAMETERS',
        f'{"="*40}',
        f'D={D}mm, L={L}mm, dmax={dmax}mm',
        f'E_base={E_base:.0f} (EI_min part)',
        f'k_contact={k_contact_total:.0f} ({N} elements)',
        f'EI_max/EI_min = {(E_base+k_contact_total)/E_base:.0f}x',
        f'f_y: {f_y_arr[0]:.1f} to {f_y_arr[-1]:.1f} MPa',
        f'eps_y: {eps_y_arr[0]:.4f} to {eps_y_arr[-1]:.3f}',
        f'eps_max = {eps_max:.3f}',
        f'',
    ]
    lines.append(f'{"Model":<18} {"Pmax":>6} {"d_res":>5} {"SlopeL":>7} {"SlopeU":>7} {"U/L":>5}')
    lines.append('-' * 50)
    for dr, (d, P) in results.items():
        i10 = np.searchsorted(d, 10.0)
        sL = (P[i10]-P[0]) / (d[i10]-d[0]+1e-30)
        ip = np.argmax(d)
        ip10 = min(ip + np.searchsorted(-d[ip:], -(dmax-10)), len(d)-1)
        sU = (P[ip10]-P[ip]) / (d[ip10]-d[ip]+1e-30) if ip10>ip else 0
        ratio = abs(sU/sL) if sL else 0
        name = f'deg={dr}' if dr < 1.0 else 'no deg'
        lines.append(f'{name:<18} {P.max()/1000:>5.1f}k {d[-1]:>5.1f} {sL:>7.0f} {sU:>7.0f} {ratio:>5.2f}')

    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = '/home/user/xkep-cae/work/beam_hysteresis/04_fat_teardrop.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved: {out}")

    # Console
    for dr, (d, P) in results.items():
        i10 = np.searchsorted(d, 10.0)
        sL = (P[i10]-P[0]) / (d[i10]-d[0]+1e-30)
        ip = np.argmax(d)
        ip10 = min(ip + np.searchsorted(-d[ip:], -(dmax-10)), len(d)-1)
        sU = (P[ip10]-P[ip]) / (d[ip10]-d[ip]+1e-30) if ip10>ip else 0
        ratio = abs(sU/sL) if sL else 0
        print(f"  deg={dr}: Pmax={P.max()/1000:.1f}kN d_res={d[-1]:.1f}mm U/L={ratio:.2f}")


if __name__ == '__main__':
    main()
