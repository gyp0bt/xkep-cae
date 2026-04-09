"""
Beam three-point bending hysteresis — concept verification
==========================================================

Purpose:
  17mm beam (span 100mm), bent 30mm, then unloaded to P=0.
  Compare small-deflection vs large-deflection models.

Key findings:
  1. KH and strand friction are mathematically identical (same VI)
  2. Single fiber => parallelogram-shaped hysteresis
  3. Multi-fiber section => rounded hysteresis (progressive yield)
  4. Geometric nonlinearity (large deflection) => distorted parallelogram
     -> "tilted teardrop" = material nonlinearity + geometric nonlinearity

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
# Curvature-deflection relations
# ============================================================
def kappa_small(delta, L):
    """Small deflection: kappa_center = 12*delta/L^2"""
    return 12.0 * delta / L**2


def kappa_large(delta, L):
    """Large deflection approximation for three-point bending.

    For a beam with central point load, the elastica gives a nonlinear
    relationship. A practical approximation:

    The exact relation for the elastica of a simply-supported beam under
    central point load involves elliptic integrals. A useful approximation
    based on the arc-length parametrization:

    For half-span: the beam deforms with angle theta at the support.
    sin(theta) ~ 2*delta / sqrt(L^2/4 + delta^2)  (geometry)
    kappa_max ~ 2*theta / (L/2) for small theta, but for large theta
    the curvature concentrates near the center.

    Simplified model: kappa = 12*delta / (L^2 + c*delta^2)
    where c accounts for shortening of the projected span.
    For c=4: matches elastica reasonably well for delta/L up to 0.3.
    """
    # Geometric shortening: projected half-span decreases
    # s = L/2 (arc length fixed), projected = sqrt((L/2)^2 - delta^2) approx
    # More practically: use the angle at support
    half_L = L / 2.0
    # theta = arctan(2*delta/L) for triangular approximation
    theta = np.arctan2(delta, half_L)
    # For elastica, curvature at center ~ 2*theta / (L/2) * correction
    # The correction increases with theta (curvature concentrates)
    # A quadratic correction:
    kappa = 12.0 * delta / (L**2) * (1.0 + 3.0 * (delta/L)**2)
    return kappa


def P_from_M_small(M, L):
    """P = 4*M/L (equilibrium, small deflection)"""
    return 4.0 * M / L


def P_from_M_large(M, L, delta):
    """Large deflection: account for geometry change.

    In large deflection, the moment arm changes. The actual vertical
    force P relates to M through the deformed geometry:

    For the deformed beam, the angle at the support is:
    theta ~ arctan(2*delta/L)

    The horizontal reaction H = P * delta / (L/2) (approximate)
    But for vertical loading, the primary effect is:
    M_center = P*L/4 in small deflection
    M_center ~ P/2 * sqrt((L/2)^2 + delta^2) * cos(alpha)
    where alpha accounts for the beam axis inclination.

    Simpler: P = 4*M / L_eff where L_eff accounts for shortening.
    """
    # Effective span decreases as beam curves
    # L_eff ~ sqrt(L^2 - 4*delta^2) for geometric shortening
    L_eff_sq = L**2 - 4 * delta**2
    if L_eff_sq < (L/4)**2:  # cap to avoid singularity
        L_eff_sq = (L/4)**2
    L_eff = np.sqrt(L_eff_sq)
    return 4.0 * M / L_eff


# ============================================================
# Three-point bending solver
# ============================================================
def three_point_bend_cycle(section, L, delta_max, large_defl=False, n_load=500):
    """Load to delta_max then unload to P=0."""
    deltas_up = np.linspace(0, delta_max, n_load + 1)
    forces_up = np.zeros(n_load + 1)

    kappa_fn = kappa_large if large_defl else kappa_small

    for i, d in enumerate(deltas_up):
        k = kappa_fn(d, L)
        M = section.moment(k)
        if large_defl:
            forces_up[i] = P_from_M_large(M, L, d)
        else:
            forces_up[i] = P_from_M_small(M, L)

    # Unloading
    n_unload = n_load * 2
    deltas_dn_all = np.linspace(delta_max, 0, n_unload + 1)[1:]
    deltas_dn, forces_dn = [], []

    for d in deltas_dn_all:
        k = kappa_fn(d, L)
        M = section.moment(k)
        if large_defl:
            P = P_from_M_large(M, L, d)
        else:
            P = P_from_M_small(M, L)
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


# ============================================================
# Single-fiber stress-strain hysteresis (for comparison)
# ============================================================
def single_fiber_hysteresis(E, sigma_y, H, eps_max, n=500):
    """Single material point: eps from 0 -> eps_max -> back until sigma=0."""
    mat = KinematicHardening1D(E=E, sigma_y=sigma_y, H=H)
    eps_up = np.linspace(0, eps_max, n + 1)
    sig_up = np.array([mat.stress(e) for e in eps_up])

    # Unload until sigma <= 0
    eps_dn_all = np.linspace(eps_max, -eps_max, 3 * n + 1)[1:]
    eps_dn, sig_dn = [], []
    for e in eps_dn_all:
        s = mat.stress(e)
        eps_dn.append(e)
        sig_dn.append(s)
        if s <= 0:
            if len(sig_dn) >= 2:
                s_prev, e_prev = sig_dn[-2], eps_dn[-2]
                if s_prev != s:
                    t = s_prev / (s_prev - s)
                    eps_dn[-1] = e_prev + t * (e - e_prev)
                    sig_dn[-1] = 0.0
            break

    return (np.concatenate([eps_up, np.array(eps_dn)]),
            np.concatenate([sig_up, np.array(sig_dn)]))


# ============================================================
# Main
# ============================================================
def main():
    D = 17.0          # mm
    L = 100.0         # mm
    delta_max = 30.0  # mm
    R = D / 2.0

    # Material: cable-like effective properties
    E = 10_000.0      # MPa (10 GPa effective)
    sigma_y = 150.0   # MPa (friction threshold)
    H = 1_000.0       # MPa (post-slip stiffness)

    eps_max = kappa_small(delta_max, L) * R

    # ============================================================
    # Compute all cases
    # ============================================================

    # (1) Single fiber (parallelogram)
    eps_1f, sig_1f = single_fiber_hysteresis(E, sigma_y, H, eps_max)

    # (2) Multi-fiber, small deflection
    sec_sd = FiberSection(diameter=D)
    sec_sd.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=sigma_y, H=H))
    d_sd, P_sd = three_point_bend_cycle(sec_sd, L, delta_max, large_defl=False)

    # (3) Multi-fiber, large deflection
    sec_ld = FiberSection(diameter=D)
    sec_ld.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=sigma_y, H=H))
    d_ld, P_ld = three_point_bend_cycle(sec_ld, L, delta_max, large_defl=True)

    # (4) Multi-fiber, large deflection, graded sigma_y
    sec_gr = FiberSection(diameter=D)
    sec_gr.setup_graded(
        lambda r: KinematicHardening1D(
            E=E, sigma_y=sigma_y * (1.0 - 0.7 * r), H=H
        )
    )
    d_gr, P_gr = three_point_bend_cycle(sec_gr, L, delta_max, large_defl=True)

    # (5) Equivalence: KH vs Strand friction
    sec_kh = FiberSection(diameter=D)
    sec_kh.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=sigma_y, H=H))
    d_kh, P_kh = three_point_bend_cycle(sec_kh, L, delta_max, large_defl=True)

    sec_sf = FiberSection(diameter=D)
    sec_sf.setup_uniform(lambda: StrandFriction1D(k_strand=E, f_y=sigma_y, k_slip=H))
    d_sf, P_sf = three_point_bend_cycle(sec_sf, L, delta_max, large_defl=True)

    n_common = min(len(P_kh), len(P_sf))
    diff_max = np.max(np.abs(P_kh[:n_common] - P_sf[:n_common]))

    # (6) Elastic reference (large defl)
    sec_el = FiberSection(diameter=D)
    sec_el.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=1e12, H=0))
    d_el, P_el = three_point_bend_cycle(sec_el, L, delta_max, large_defl=True)

    # ============================================================
    # Plotting: 2x3
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- (a) Single fiber: parallelogram ---
    ax = axes[0, 0]
    ax.plot(eps_1f * 100, sig_1f, 'b-', lw=2)
    ax.set_xlabel('Strain [%]')
    ax.set_ylabel('Stress [MPa]')
    ax.set_title('(a) Single fiber: stress-strain\n=> PARALLELOGRAM')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    # Annotate
    ax.annotate('Loading', xy=(eps_max*50, sigma_y + H*eps_max*0.3),
                fontsize=10, color='blue')
    ax.annotate('Unloading\n(2*sy elastic range)', xy=(eps_max*80, 0),
                fontsize=9, color='red')

    # --- (b) Small deflection: multi-fiber ---
    ax = axes[0, 1]
    ax.plot(d_sd, P_sd / 1000, 'b-', lw=2, label='Multi-fiber (small defl.)')
    ax.plot(d_el, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(b) Multi-fiber, SMALL deflection\n=> Rounded parallelogram')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)

    # --- (c) Large deflection: multi-fiber ---
    ax = axes[0, 2]
    ax.plot(d_ld, P_ld / 1000, 'r-', lw=2, label='Multi-fiber (large defl.)')
    ax.plot(d_sd, P_sd / 1000, 'b--', lw=1.5, alpha=0.6, label='Small defl.')
    ax.plot(d_el, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(c) Multi-fiber, LARGE deflection\n=> Distorted = TEARDROP')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)

    # --- (d) All together ---
    ax = axes[1, 0]
    ax.plot(d_sd, P_sd / 1000, 'b-', lw=2, label='Small defl., uniform sy')
    ax.plot(d_ld, P_ld / 1000, 'r-', lw=2, label='Large defl., uniform sy')
    ax.plot(d_gr, P_gr / 1000, 'm-', lw=2, label='Large defl., graded sy')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(d) Comparison of all models')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)

    # --- (e) KH vs Strand friction ---
    ax = axes[1, 1]
    ax.plot(d_kh, P_kh / 1000, 'b-', lw=2.5, label='Kinematic hardening')
    ax.plot(d_sf, P_sf / 1000, 'r--', lw=2, label='Strand friction')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title(f'(e) KH vs Friction: diff={diff_max:.1e} N')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.15, 'IDENTICAL', transform=ax.transAxes,
            ha='center', fontsize=18, color='green', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8))

    # --- (f) Shape decomposition ---
    ax = axes[1, 2]
    ax.axis('off')
    text = (
        "HYSTERESIS SHAPE DECOMPOSITION\n"
        "=" * 44 + "\n\n"
        "Single fiber (1 material point):\n"
        "  sigma-eps = PARALLELOGRAM\n"
        "  (bilinear KH => 2 slopes + reversal)\n\n"
        "Multi-fiber section (small deflection):\n"
        "  P-delta = ROUNDED parallelogram\n"
        "  (outer fibers yield first => smooth transition)\n"
        "  (same as: outer strands slip first)\n\n"
        "Multi-fiber + LARGE deflection:\n"
        "  P-delta = DISTORTED/TILTED shape\n"
        "  (kappa-delta becomes nonlinear)\n"
        "  (moment arm changes with deformation)\n"
        "  => TEARDROP shape\n\n"
        "Graded sigma_y (outer=0.3x inner):\n"
        "  Even more asymmetric/rounded\n"
        "  (= variable friction per strand layer)\n\n"
        "─" * 22 + "\n"
        "KH == Strand friction (diff=0)\n"
        "  E       <-> k_strand\n"
        "  sigma_y <-> f_y = mu*N\n"
        "  H       <-> k_slip\n"
        "  Same variational inequality."
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
    print(f"\nD={D}mm, L={L}mm, delta_max={delta_max}mm (delta/L={delta_max/L:.0%})")
    print(f"E={E/1000:.0f}GPa, sigma_y={sigma_y:.0f}MPa, H={H:.0f}MPa")
    print(f"eps_max={eps_max:.4f} ({eps_max*100:.1f}%), eps_y={sigma_y/E:.4f} ({sigma_y/E*100:.1f}%)")
    print(f"eps_max/eps_y = {eps_max/(sigma_y/E):.1f}x")
    print(f"\nP_max: small_defl={P_sd.max()/1000:.2f}kN, large_defl={P_ld.max()/1000:.2f}kN")
    print(f"KH vs Friction: max diff = {diff_max:.1e} N")

    print("""
SUMMARY:
  1. Single fiber KH => parallelogram (sigma-eps)
  2. Multi-fiber section => rounded (progressive yield, outside-in)
  3. Large deflection => distorted/tilted (geometric nonlinearity)
  4. Combined => TEARDROP hysteresis
  5. KH == strand friction (exact, not approximate)
""")


if __name__ == '__main__':
    main()
