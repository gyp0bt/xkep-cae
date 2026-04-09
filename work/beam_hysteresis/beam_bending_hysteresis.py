"""
Beam three-point bending hysteresis — concept verification
==========================================================

Realistic cable hysteresis:
  - Loading: starts linear, then curve flattens (like geometric NL)
  - Unloading: draws an elliptical path back to near-origin
  - Residual displacement is SMALL (teardrop tail near origin)
  - The loop is a THIN, tilted teardrop (not fat parallelogram)

Key: the cable is mostly elastic. Friction dissipation is a small
perturbation on top of elastic behavior. sigma_y is high relative
to the strains involved.

[<- README](../../README.md)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ============================================================
# Material models
# ============================================================
@dataclass
class KinematicHardening1D:
    E: float
    sigma_y: float
    H: float
    eps_p: float = 0.0
    alpha: float = 0.0

    def stress(self, eps: float) -> float:
        sigma_trial = self.E * (eps - self.eps_p)
        eta = sigma_trial - self.alpha
        if abs(eta) <= self.sigma_y:
            return sigma_trial
        sign = np.sign(eta)
        dgamma = (abs(eta) - self.sigma_y) / (self.E + self.H)
        self.eps_p += sign * dgamma
        self.alpha += sign * self.H * dgamma
        return self.E * (eps - self.eps_p)


@dataclass
class StrandFriction1D:
    k_strand: float
    f_y: float
    k_slip: float
    u_slip: float = 0.0
    f_locked: float = 0.0

    def stress(self, eps: float) -> float:
        f_trial = self.k_strand * (eps - self.u_slip)
        eta = f_trial - self.f_locked
        if abs(eta) <= self.f_y:
            return f_trial
        sign = np.sign(eta)
        dslip = (abs(eta) - self.f_y) / (self.k_strand + self.k_slip)
        self.u_slip += sign * dslip
        self.f_locked += sign * self.k_slip * dslip
        return self.k_strand * (eps - self.u_slip)


# ============================================================
# Fiber section
# ============================================================
@dataclass
class FiberSection:
    diameter: float
    n_fiber: int = 50
    fibers: list = field(default_factory=list)
    y_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    areas: np.ndarray = field(default_factory=lambda: np.array([]))

    def setup_uniform(self, material_factory):
        self._init_geometry()
        self.fibers = [material_factory() for _ in range(self.n_fiber)]

    def setup_graded(self, material_factory_fn):
        self._init_geometry()
        R = self.diameter / 2.0
        self.fibers = [material_factory_fn(abs(y) / R) for y in self.y_coords]

    def _init_geometry(self):
        R = self.diameter / 2.0
        self.y_coords = np.linspace(-R + R/self.n_fiber, R - R/self.n_fiber, self.n_fiber)
        dy = 2 * R / self.n_fiber
        self.areas = np.array([self._chord_area(y, dy, R) for y in self.y_coords])

    @staticmethod
    def _chord_area(y_center, dy, R):
        y_lo = max(y_center - dy/2, -R)
        y_hi = min(y_center + dy/2, R)
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


# ============================================================
# Three-point bending
# ============================================================
def three_point_bend_cycle(section, L, delta_max, n_load=500):
    """Load to delta_max then unload to P=0."""
    # Loading
    deltas_up = np.linspace(0, delta_max, n_load + 1)
    forces_up = np.zeros(n_load + 1)
    for i, d in enumerate(deltas_up):
        kappa = 12.0 * d / L**2
        M = section.moment(kappa)
        forces_up[i] = 4.0 * M / L

    # Unloading: go to delta=0, stop at P<=0
    n_unload = n_load * 3
    deltas_dn_all = np.linspace(delta_max, 0, n_unload + 1)[1:]
    deltas_dn, forces_dn = [], []

    for d in deltas_dn_all:
        kappa = 12.0 * d / L**2
        M = section.moment(kappa)
        P = 4.0 * M / L
        deltas_dn.append(d)
        forces_dn.append(P)
        if P <= 0:
            if len(forces_dn) >= 2:
                P_prev, d_prev = forces_dn[-2], deltas_dn[-2]
                if P_prev != P:
                    t = P_prev / (P_prev - P)
                    deltas_dn[-1] = d_prev + t * (d - d_prev)
                    forces_dn[-1] = 0.0
            break

    return (np.concatenate([deltas_up, np.array(deltas_dn)]),
            np.concatenate([forces_up, np.array(forces_dn)]))


def find_residual(deltas, forces):
    n = len(deltas) // 2
    for j in range(n, len(forces) - 1):
        if forces[j] * forces[j+1] <= 0 and forces[j] != forces[j+1]:
            t = forces[j] / (forces[j] - forces[j+1])
            return deltas[j] + t * (deltas[j+1] - deltas[j])
    return deltas[-1]


# ============================================================
# Main
# ============================================================
def main():
    D = 17.0          # mm
    L = 100.0         # mm
    delta_max = 30.0  # mm
    R = D / 2.0

    eps_max = 12 * delta_max / L**2 * R  # = 0.306

    # ============================================================
    # Realistic cable parameters
    # ============================================================
    # The key insight: in a real cable, most of the section is elastic
    # at the bending strains involved. Only outer fibers enter the
    # friction-slip regime. This means:
    #   - High sigma_y relative to E*eps_max (only partial yield)
    #   - OR: use graded sigma_y where inner fibers never yield
    #
    # For eps_max = 0.306:
    #   If sigma_y = 0.9 * E * eps_max, only the outermost 10% yields
    #   => thin loop, small residual, teardrop with tail at origin

    # Parameter sweep to find the right shape
    param_sets = {
        'A: Mostly elastic (outer 20% yields)': dict(
            E=10_000, sigma_y=2500, H=500,
            # eps_y = 0.25, eps_max/eps_y = 1.22 => only outermost fibers yield
        ),
        'B: Moderate (outer 50% yields)': dict(
            E=10_000, sigma_y=1500, H=500,
            # eps_y = 0.15, eps_max/eps_y = 2.04
        ),
        'C: Deep yield (80% yields)': dict(
            E=10_000, sigma_y=600, H=500,
            # eps_y = 0.06, eps_max/eps_y = 5.1
        ),
        'D: Graded (outer=0.3x, inner=1x)': dict(
            E=10_000, sigma_y=2500, H=500, graded=True,
            # outer fibers have lower threshold
        ),
    }

    results = {}
    for name, p in param_sets.items():
        sec = FiberSection(diameter=D, n_fiber=60)
        E, sy, H = p['E'], p['sigma_y'], p['H']
        if p.get('graded'):
            sec.setup_graded(
                lambda r, E=E, sy=sy, H=H: KinematicHardening1D(
                    E=E, sigma_y=sy * (1.0 - 0.7 * r), H=H
                )
            )
        else:
            sec.setup_uniform(
                lambda E=E, sy=sy, H=H: KinematicHardening1D(E=E, sigma_y=sy, H=H)
            )
        d, P = three_point_bend_cycle(sec, L, delta_max)
        d_res = find_residual(d, P)
        results[name] = dict(d=d, P=P, d_res=d_res, **p)

    # Elastic reference
    sec_el = FiberSection(diameter=D, n_fiber=60)
    sec_el.setup_uniform(lambda: KinematicHardening1D(E=10_000, sigma_y=1e12, H=0))
    d_el, P_el = three_point_bend_cycle(sec_el, L, delta_max)

    # KH vs Friction equivalence (case A)
    pA = param_sets['A: Mostly elastic (outer 20% yields)']
    sec_kh = FiberSection(diameter=D, n_fiber=60)
    sec_kh.setup_uniform(lambda: KinematicHardening1D(E=pA['E'], sigma_y=pA['sigma_y'], H=pA['H']))
    d_kh, P_kh = three_point_bend_cycle(sec_kh, L, delta_max)

    sec_sf = FiberSection(diameter=D, n_fiber=60)
    sec_sf.setup_uniform(lambda: StrandFriction1D(k_strand=pA['E'], f_y=pA['sigma_y'], k_slip=pA['H']))
    d_sf, P_sf = three_point_bend_cycle(sec_sf, L, delta_max)

    n_c = min(len(P_kh), len(P_sf))
    diff_max = np.max(np.abs(P_kh[:n_c] - P_sf[:n_c]))

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'magenta'}

    # --- (a) All hysteresis loops ---
    ax = axes[0, 0]
    for name, r in results.items():
        key = name[0]
        ax.plot(r['d'], r['P'] / 1000, '-', color=colors[key], lw=2,
                label=f"{key} (d_res={r['d_res']:.1f}mm)")
    ax.plot(d_el, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(a) Hysteresis loops (different yield levels)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- (b) Case A close-up (most realistic) ---
    rA = results['A: Mostly elastic (outer 20% yields)']
    ax = axes[0, 1]
    ax.plot(rA['d'], rA['P'] / 1000, 'b-', lw=2.5, label='KH (mostly elastic)')
    ax.plot(d_el, P_el / 1000, 'k:', lw=1.5, alpha=0.4, label='Elastic')
    ax.fill_between(rA['d'][:len(rA['P'])], 0, rA['P'] / 1000, alpha=0.1, color='blue')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(b) Best match: sy={rA["sigma_y"]}MPa\n'
                 f'd_res={rA["d_res"]:.1f}mm ({rA["d_res"]/delta_max*100:.0f}%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # Arrow showing teardrop tail
    ax.annotate('Teardrop tail\n(small residual)',
                xy=(rA['d_res'], 0), xytext=(rA['d_res'] + 5, rA['P'].max()/1000 * 0.15),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')

    # --- (c) KH vs Strand friction ---
    ax = axes[0, 2]
    ax.plot(d_kh, P_kh / 1000, 'b-', lw=2.5, label='Kinematic hardening')
    ax.plot(d_sf, P_sf / 1000, 'r--', lw=2, label='Strand friction')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(c) KH vs Friction: diff={diff_max:.1e} N')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.text(0.5, 0.15, 'IDENTICAL', transform=ax.transAxes,
            ha='center', fontsize=18, color='green', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8))

    # --- (d) Normalized shape ---
    ax = axes[1, 0]
    for name, r in results.items():
        key = name[0]
        Pm = np.max(r['P'])
        if Pm > 0:
            ey = r['sigma_y'] / r['E']
            ax.plot(r['d'] / delta_max, r['P'] / Pm, '-', color=colors[key], lw=1.5,
                    label=f"{key} (eps/ey={eps_max/ey:.1f}x, res={r['d_res']/delta_max*100:.0f}%)")
    ax.set_xlabel('delta / delta_max')
    ax.set_ylabel('P / P_max')
    ax.set_title('(d) Normalized: shape depends on eps_max/eps_y')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)

    # --- (e) Graded vs Uniform (same yield level) ---
    rD = results['D: Graded (outer=0.3x, inner=1x)']
    ax = axes[1, 1]
    ax.plot(rA['d'], rA['P'] / 1000, 'b-', lw=2, label='A: Uniform sy')
    ax.plot(rD['d'], rD['P'] / 1000, 'm-', lw=2, label='D: Graded sy (outer low)')
    ax.plot(d_el, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(e) Graded: outer strands slip earlier\n=> wider loop, more dissipation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- (f) Summary ---
    ax = axes[1, 2]
    ax.axis('off')
    text = (
        "REALISTIC CABLE HYSTERESIS\n"
        "=" * 42 + "\n\n"
        "Loading: linear start -> curve flattens\n"
        "  (outer fibers/strands start to slip)\n\n"
        "Unloading: elliptical path back to\n"
        "  near-origin (small residual defl.)\n\n"
        "Teardrop tail at ORIGIN side.\n\n"
        "-" * 42 + "\n"
        "Key parameter: eps_max / eps_y\n"
        f"  eps_max = {eps_max:.3f} (at surface, delta={delta_max}mm)\n\n"
        "  eps/ey ~ 1.2: thin loop, small residual\n"
        "            (MOST REALISTIC for cables)\n"
        "  eps/ey ~ 2:   moderate loop\n"
        "  eps/ey ~ 5:   fat loop, large residual\n"
        "  eps/ey >> 10: parallelogram (all yielded)\n\n"
        "-" * 42 + "\n"
        "KH == Strand friction (exact)\n"
        "  sigma_y <-> f_y = mu*N_contact\n"
        "  H <-> k_slip (post-slip constraint)\n"
        "  Fiber yield = strand slip\n\n"
        "Cable is MOSTLY ELASTIC.\n"
        "Friction = small perturbation\n"
        "  => thin teardrop, tail near origin."
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    outpath = '/home/user/xkep-cae/work/beam_hysteresis/hysteresis_comparison.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Plot saved: {outpath}")

    # Console
    print(f"\nD={D}mm, L={L}mm, delta_max={delta_max}mm")
    print(f"eps_max = {eps_max:.4f} ({eps_max*100:.1f}%)")
    print(f"\n{'Case':<35} {'sy':>5} {'eps/ey':>7} {'Pmax[kN]':>9} {'d_res[mm]':>9} {'res%':>5}")
    print("-" * 75)
    for name, r in results.items():
        key = name.split(':')[0]
        ey = r['sigma_y'] / r['E']
        print(f"{key:<35} {r['sigma_y']:>5.0f} {eps_max/ey:>7.1f} "
              f"{np.max(r['P'])/1000:>9.2f} {r['d_res']:>9.1f} "
              f"{r['d_res']/delta_max*100:>5.0f}%")
    print(f"\nKH vs Friction: {diff_max:.1e} N")


if __name__ == '__main__':
    main()
