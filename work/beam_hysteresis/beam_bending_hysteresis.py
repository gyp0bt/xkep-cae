"""
Beam bending hysteresis — asymmetric slope investigation
=========================================================

PROBLEM: All previous fiber+KH models give the same loading/unloading
slope because E (elastic stiffness) is the same in both directions.

Real cable observation:
  - Loading slope (initial tangent) > Unloading slope
  - Loading: elastic-like start then flattens
  - Unloading: gentler slope, elliptical return to near-origin

NEW APPROACH: Direct M-kappa model + EI decomposition.

Physical picture:
  EI_total = EI_individual + EI_contact
  - EI_individual: sum of individual strand bending stiffnesses (always present)
  - EI_contact: additional stiffness from inter-strand friction lock

  Loading (from zero):
    All strands stuck -> EI = EI_individual + EI_contact = EI_max
    As curvature increases, outer strands slip -> EI decreases

  Unloading (from peak):
    At peak, strands have slipped to new positions.
    On reversal, relative strand motion reverses.
    BUT: the normal force N has decreased (less curvature)
    -> friction capacity is lower
    -> strands re-lock less firmly
    -> EI_contact is reduced
    -> unloading stiffness < initial loading stiffness

[<- README](../../README.md)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ============================================================
# Model 1: Standard fiber + KH (baseline, symmetric slopes)
# ============================================================
@dataclass
class StandardKH:
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


# ============================================================
# Model 2: Two-spring fiber (E_beam + E_contact)
# E_contact degrades after slip.
#
# sigma = E_beam * eps + sigma_contact
# sigma_contact: KH with E_contact that DEGRADES after yielding.
# On reversal, the elastic slope in the contact spring is E_degraded < E_contact.
# ============================================================
@dataclass
class TwoSpringDegradingContact:
    """Elastic beam + contact spring that degrades after slip.

    Physical: once strands slip past each other, the contact surface
    has changed. On re-engagement, the contact stiffness is lower.
    This directly reduces the unloading slope.

    Loading elastic slope = E_beam + E_contact_virgin
    Unloading elastic slope = E_beam + E_contact_degraded
    """
    E_beam: float           # always-present elastic stiffness
    E_contact_virgin: float # contact stiffness before any slip
    E_contact_degraded: float  # contact stiffness after slip
    sigma_y: float          # friction threshold
    H: float                # hardening

    # Contact spring state
    eps_p_c: float = 0.0
    alpha_c: float = 0.0
    ever_slipped: bool = False

    def stress(self, eps):
        # Elastic backbone (always linear)
        sig_beam = self.E_beam * eps

        # Contact spring (KH with degrading E)
        E_c = self.E_contact_degraded if self.ever_slipped else self.E_contact_virgin
        sig_trial_c = E_c * (eps - self.eps_p_c)
        eta = sig_trial_c - self.alpha_c

        if abs(eta) <= self.sigma_y:
            return sig_beam + sig_trial_c
        else:
            self.ever_slipped = True
            E_c = self.E_contact_degraded  # degrade
            # Recompute trial with degraded stiffness
            sig_trial_c = E_c * (eps - self.eps_p_c)
            eta = sig_trial_c - self.alpha_c
            if abs(eta) <= self.sigma_y:
                return sig_beam + sig_trial_c
            s = np.sign(eta)
            dg = (abs(eta) - self.sigma_y) / (E_c + self.H)
            self.eps_p_c += s * dg
            self.alpha_c += s * self.H * dg
            return sig_beam + E_c * (eps - self.eps_p_c)


# ============================================================
# Model 3: Multi-layer Prandtl-Ishlinskii with degradation
#
# Each friction element loses stiffness after first slip.
# Elements with low threshold slip first, degrade first.
# Net effect: initial loading uses high k, unloading uses degraded k.
# ============================================================
@dataclass
class PIWithDegradation:
    """Prandtl-Ishlinskii with contact stiffness degradation.

    N parallel elements: each has virgin and degraded stiffness.
    """
    E_base: float
    k_virgin: np.ndarray
    k_degraded: np.ndarray
    f_y: np.ndarray
    slip: np.ndarray = field(default=None)
    alpha: np.ndarray = field(default=None)
    has_slipped: np.ndarray = field(default=None)

    def __post_init__(self):
        n = len(self.k_virgin)
        if self.slip is None:
            self.slip = np.zeros(n)
        if self.alpha is None:
            self.alpha = np.zeros(n)
        if self.has_slipped is None:
            self.has_slipped = np.zeros(n, dtype=bool)

    def stress(self, eps):
        sigma = self.E_base * eps
        for i in range(len(self.k_virgin)):
            k = self.k_degraded[i] if self.has_slipped[i] else self.k_virgin[i]
            trial = k * (eps - self.slip[i])
            eta = trial - self.alpha[i]
            if abs(eta) <= self.f_y[i]:
                sigma += trial
            else:
                self.has_slipped[i] = True
                k = self.k_degraded[i]
                # Recompute
                trial = k * (eps - self.slip[i])
                eta = trial - self.alpha[i]
                if abs(eta) <= self.f_y[i]:
                    sigma += trial
                else:
                    s = np.sign(eta)
                    dg = (abs(eta) - self.f_y[i]) / k
                    self.slip[i] += s * dg
                    sigma += k * (eps - self.slip[i])
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
            eps = -kappa * y
            sig = f.stress(eps)
            M += -sig * y * A
        return M


# ============================================================
# Bending cycle
# ============================================================
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


def tangent_stiffness(d, P):
    """Compute dP/d(delta) along the curve."""
    dd = np.diff(d)
    dP = np.diff(P)
    stiffness = np.where(np.abs(dd) > 1e-10, dP / dd, 0)
    return 0.5*(d[:-1]+d[1:]), stiffness


# ============================================================
# Main
# ============================================================
def main():
    D, L, dmax = 17.0, 100.0, 30.0
    R = D / 2.0

    # eps_max at surface: kappa_max * R = 12*30/100^2 * 8.5 = 0.306
    eps_max = 12 * dmax / L**2 * R
    print(f"eps_max = {eps_max:.3f}")

    # =========================================
    # M1: Standard KH (baseline, same slopes)
    # sigma_y=1500, eps_y=0.15, eps_max/eps_y=2.0
    # =========================================
    sec1 = FiberSection(D)
    sec1.setup(lambda: StandardKH(E=10_000, sigma_y=1500, H=1000))
    d1, P1 = bend_cycle(sec1, L, dmax)

    # =========================================
    # M2: Two-spring with degradation
    # Contact spring yield: sigma_y=300, eps_y_c=300/4000=0.075
    # eps_max/eps_y_c = 4.1 => outer 75% of fibers yield in contact spring
    # Loading: E_beam + E_contact_virgin = 6000 + 4000 = 10000
    # After slip: E_beam + E_contact_degraded = 6000 + 1500 = 7500
    # =========================================
    sec2 = FiberSection(D)
    sec2.setup(lambda: TwoSpringDegradingContact(
        E_beam=6000, E_contact_virgin=4000, E_contact_degraded=1500,
        sigma_y=300, H=500,
    ))
    d2, P2 = bend_cycle(sec2, L, dmax)

    # =========================================
    # M3: Two-spring, stronger degradation
    # Contact yield: sigma_y=200, eps_y_c=200/4000=0.05
    # eps_max/eps_y_c = 6.1 => most fibers yield
    # After slip: E_degraded=500 => total=6500 (65% of virgin)
    # =========================================
    sec3 = FiberSection(D)
    sec3.setup(lambda: TwoSpringDegradingContact(
        E_beam=6000, E_contact_virgin=4000, E_contact_degraded=500,
        sigma_y=200, H=300,
    ))
    d3, P3 = bend_cycle(sec3, L, dmax)

    # =========================================
    # M4: PI with degradation (5 layers)
    # Lower thresholds so yield actually occurs
    # =========================================
    N = 5
    k_v = np.array([1000, 800, 600, 400, 200], dtype=float)
    k_d = np.array([300, 250, 200, 150, 100], dtype=float)
    fy  = np.array([50, 100, 200, 400, 800], dtype=float)  # much lower

    sec4 = FiberSection(D)
    sec4.setup(lambda: PIWithDegradation(
        E_base=6000, k_virgin=k_v.copy(), k_degraded=k_d.copy(), f_y=fy.copy(),
    ))
    d4, P4 = bend_cycle(sec4, L, dmax)

    # =========================================
    # Elastic
    # =========================================
    sec_el = FiberSection(D)
    sec_el.setup(lambda: StandardKH(E=10_000, sigma_y=1e12, H=0))
    d_el, P_el = bend_cycle(sec_el, L, dmax)

    models = [
        ('M1: Standard KH (same slope)', d1, P1, 'blue', '-'),
        ('M2: Degraded contact (mild)', d2, P2, 'red', '-'),
        ('M3: Degraded contact (strong)', d3, P3, 'green', '-'),
        ('M4: PI + degradation', d4, P4, 'magenta', '-'),
    ]

    # ============================================================
    # Measure slopes: loading 0-5mm vs unloading first 5mm
    # ============================================================
    print(f"D={D}mm, L={L}mm, dmax={dmax}mm\n")
    print(f"{'Model':<35} {'Pmax':>7} {'d_res':>6} {'SlopeL':>8} {'SlopeU':>8} {'U/L':>6}")
    print("-" * 72)
    for name, d, P, _, _ in models:
        # Loading slope: secant from 0 to 10mm
        i10 = np.searchsorted(d, 10.0)
        sL = (P[i10] - P[0]) / (d[i10] - d[0] + 1e-30)

        # Unloading slope: secant from peak to peak-10mm
        ip = np.argmax(d)
        ip10 = ip + np.searchsorted(-d[ip:], -(dmax - 10.0))
        ip10 = min(ip10, len(d)-1)
        sU = (P[ip10] - P[ip]) / (d[ip10] - d[ip] + 1e-30) if ip10 > ip else 0

        ratio = abs(sU / sL) if sL != 0 else 0
        print(f"{name:<35} {P.max()/1000:>6.1f}k {d[-1]:>5.1f}m {sL:>8.0f} {sU:>8.0f} {ratio:>6.2f}")

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) All hysteresis
    ax = axes[0, 0]
    for name, d, P, c, ls in models:
        ax.plot(d, P/1000, color=c, ls=ls, lw=2, label=name.split(':')[1].strip()[:28])
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(a) All models')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (b) M1 (baseline) + tangent stiffness
    ax = axes[0, 1]
    ax.plot(d1, P1/1000, 'b-', lw=2.5, label='M1: Standard KH')
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3)
    ax.set_title('(b) M1: Standard KH\n(loading slope = unloading slope)')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (c) M3 (strong degradation) — TARGET SHAPE
    ax = axes[0, 2]
    ax.plot(d3, P3/1000, 'g-', lw=2.5, label='M3: Strong degradation')
    ax.plot(d_el, P_el/1000, 'k:', lw=1, alpha=0.3)
    ax.set_title('(c) M3: Contact stiffness degrades after slip\n'
                 '(loading slope > unloading slope)')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (d) Tangent stiffness comparison
    ax = axes[1, 0]
    for name, d, P, c, _ in models:
        dm, stiff = tangent_stiffness(d, P)
        ax.plot(dm, stiff/1000, color=c, lw=1.5, alpha=0.8,
                label=name.split(':')[0])
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Tangent stiffness dP/d(delta) [kN/mm]')
    ax.set_title('(d) Tangent stiffness along path')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)

    # (e) M1 vs M3 direct
    ax = axes[1, 1]
    ax.plot(d1, P1/1000, 'b-', lw=2, label='M1: Standard KH')
    ax.plot(d3, P3/1000, 'g-', lw=2, label='M3: Degraded contact')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(e) Direct comparison: symmetric vs asymmetric')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0); ax.set_ylim(bottom=0)

    # (f) Explanation
    ax = axes[1, 2]
    ax.axis('off')
    text = (
        'WHY UNLOADING IS SOFTER\n'
        '=' * 38 + '\n\n'
        'Cable EI = EI_beam + EI_contact\n\n'
        'LOADING (virgin contact):\n'
        '  Strands stuck together\n'
        '  EI_contact = EI_contact_virgin (high)\n'
        '  => Steep initial slope\n\n'
        'FIRST SLIP:\n'
        '  Contact surfaces slide\n'
        '  Interface geometry changes\n'
        '  EI_contact DEGRADES permanently\n\n'
        'UNLOADING (degraded contact):\n'
        '  Even in stick mode, the contact\n'
        '  stiffness is now LOWER:\n'
        '  EI_contact = EI_contact_degraded\n'
        '  => Gentler unloading slope\n\n'
        '-' * 38 + '\n'
        'Model: sigma = E_beam*eps + sig_c\n'
        '  sig_c uses E_c_virgin before slip\n'
        '  sig_c uses E_c_degraded after slip\n\n'
        'This is contact DAMAGE, not plasticity.\n'
        'The elastic stiffness itself changes.\n'
        '(Similar to continuum damage mechanics)'
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = '/home/user/xkep-cae/work/beam_hysteresis/hysteresis_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out}")


if __name__ == '__main__':
    main()
