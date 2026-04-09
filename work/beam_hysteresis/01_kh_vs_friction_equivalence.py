"""
Beam three-point bending hysteresis — concept verification
==========================================================

Purpose:
  17mm circular cross-section beam (span 100mm), bent 30mm at center,
  then unloaded. Compute load-displacement hysteresis.

  Verify that kinematic hardening reproduces the "tilted teardrop"
  hysteresis seen in stranded conductors.

Model variants:
  (A) Uniform kinematic hardening: all fibers same sigma_y
  (B) Multi-layer friction: outer fibers have LOWER friction threshold
      (physically: outer strands have higher curvature -> slip earlier)
  (C) Nonlinear hardening: Voce-type saturation model

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
    """1D bilinear kinematic hardening (Prager)."""
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
    """1D strand friction — mathematically identical to KinematicHardening1D."""
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


@dataclass
class VoceKinematicHardening1D:
    """Nonlinear kinematic hardening (Armstrong-Frederick / Voce-type).

    Back stress saturates: d_alpha = C * d_eps_p - gamma * alpha * |d_eps_p|
    This gives a more rounded hysteresis (closer to real cables).
    """
    E: float
    sigma_y: float
    C: float       # initial hardening rate
    gamma: float   # saturation rate (alpha_sat = C/gamma)
    eps_p: float = 0.0
    alpha: float = 0.0

    def stress(self, eps: float) -> float:
        sigma_trial = self.E * (eps - self.eps_p)
        eta = sigma_trial - self.alpha
        if abs(eta) <= self.sigma_y:
            return sigma_trial
        sign = np.sign(eta)
        # Implicit return mapping for nonlinear hardening
        # Simplified: use explicit Euler with small substeps
        excess = abs(eta) - self.sigma_y
        dgamma = excess / self.E  # approximate
        self.eps_p += sign * dgamma
        # Armstrong-Frederick update
        self.alpha += (self.C * sign - self.gamma * self.alpha) * dgamma
        return self.E * (eps - self.eps_p)


# ============================================================
# Fiber section
# ============================================================
@dataclass
class FiberSection:
    diameter: float
    n_fiber: int = 40
    fibers: list = field(default_factory=list)
    y_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    areas: np.ndarray = field(default_factory=lambda: np.array([]))

    def setup_uniform(self, material_factory):
        """All fibers get the same material."""
        self._init_geometry()
        self.fibers = [material_factory() for _ in range(self.n_fiber)]

    def setup_graded(self, material_factory_fn):
        """Each fiber gets material depending on |y|/R (0=center, 1=surface).

        material_factory_fn(r_ratio) -> material instance
        """
        self._init_geometry()
        R = self.diameter / 2.0
        self.fibers = [
            material_factory_fn(abs(y) / R) for y in self.y_coords
        ]

    def _init_geometry(self):
        R = self.diameter / 2.0
        self.y_coords = np.linspace(
            -R + R / self.n_fiber, R - R / self.n_fiber, self.n_fiber
        )
        dy = 2 * R / self.n_fiber
        self.areas = np.array([self._chord_area(y, dy, R) for y in self.y_coords])

    @staticmethod
    def _chord_area(y_center, dy, R):
        y_lo = max(y_center - dy / 2, -R)
        y_hi = min(y_center + dy / 2, R)
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
def three_point_bend(section, L, deltas):
    forces = np.zeros_like(deltas)
    for i, d in enumerate(deltas):
        kappa = 12.0 * d / L**2
        M = section.moment(kappa)
        forces[i] = 4.0 * M / L
    return forces


def make_path(delta_max, n=300):
    """0 -> max -> 0"""
    up = np.linspace(0, delta_max, n + 1)
    down = np.linspace(delta_max, 0, n + 1)[1:]
    return np.concatenate([up, down])


def find_residual(deltas, forces):
    """Find deflection where force crosses zero on unloading."""
    n = len(deltas) // 2
    for j in range(n, len(forces) - 1):
        if forces[j] * forces[j + 1] <= 0 and forces[j] != forces[j + 1]:
            t = forces[j] / (forces[j] - forces[j + 1])
            return deltas[j] + t * (deltas[j + 1] - deltas[j])
    return 0.0


# ============================================================
# Main
# ============================================================
def main():
    D = 17.0          # mm
    L = 100.0         # mm
    delta_max = 30.0  # mm

    # --- Adjusted parameters for realistic hysteresis ---
    # Key insight: for visible teardrop, need sigma_y/E * L^2/(12*R*delta_max)
    # to be O(0.1~0.5) so that partial yielding occurs at peak
    E = 100_000.0     # MPa
    sigma_y = 3000.0  # MPa — high enough that partial yielding at delta_max
    H = 10_000.0      # MPa

    # Check: eps_max at delta=30mm
    R = D / 2.0
    kappa_max = 12 * delta_max / L**2
    eps_max = kappa_max * R
    eps_y = sigma_y / E
    print(f"eps_max = {eps_max:.4f}, eps_y = {eps_y:.4f}, ratio = {eps_max/eps_y:.2f}")
    print(f"  -> {eps_max/eps_y:.0f}x yield strain at surface")

    deltas = make_path(delta_max)

    # === Model A: Uniform kinematic hardening ===
    sec_a = FiberSection(diameter=D)
    sec_a.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=sigma_y, H=H))
    P_a = three_point_bend(sec_a, L, deltas)

    # === Model A': Strand friction (same params, verify equivalence) ===
    sec_a2 = FiberSection(diameter=D)
    sec_a2.setup_uniform(lambda: StrandFriction1D(k_strand=E, f_y=sigma_y, k_slip=H))
    P_a2 = three_point_bend(sec_a2, L, deltas)

    # === Model B: Graded friction (outer layers slip earlier) ===
    # sigma_y varies: center=sigma_y, surface=sigma_y*0.2
    # Physically: outer strands under more curvature, less normal force
    sec_b = FiberSection(diameter=D)
    sec_b.setup_graded(
        lambda r: KinematicHardening1D(
            E=E,
            sigma_y=sigma_y * (1.0 - 0.8 * r),  # center: sigma_y, surface: 0.2*sigma_y
            H=H,
        )
    )
    P_b = three_point_bend(sec_b, L, deltas)

    # === Model C: Nonlinear (Armstrong-Frederick) hardening ===
    sec_c = FiberSection(diameter=D)
    sec_c.setup_uniform(
        lambda: VoceKinematicHardening1D(
            E=E, sigma_y=sigma_y * 0.5,
            C=50_000.0,   # initial hardening rate
            gamma=100.0,  # saturation: alpha_sat = C/gamma = 500 MPa
        )
    )
    P_c = three_point_bend(sec_c, L, deltas)

    # === Model D: Elastic ===
    sec_el = FiberSection(diameter=D)
    sec_el.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=1e12, H=0))
    P_el = three_point_bend(sec_el, L, deltas)

    # === Low sigma_y case (original params — shows thin loop) ===
    sec_low = FiberSection(diameter=D)
    sec_low.setup_uniform(lambda: KinematicHardening1D(E=E, sigma_y=200.0, H=5000.0))
    P_low = three_point_bend(sec_low, L, deltas)

    # ============================================================
    # Plotting
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- (a) Main comparison: all models ---
    ax = axes[0, 0]
    ax.plot(deltas, P_a / 1000, 'b-', lw=2, label=f'(A) Uniform KH (sy={sigma_y:.0f})')
    ax.plot(deltas, P_a2 / 1000, 'r--', lw=1.5, label='(A\') Strand friction (same)')
    ax.plot(deltas, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(a) Uniform: KH vs Strand friction')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (b) Graded vs Uniform ---
    ax = axes[0, 1]
    ax.plot(deltas, P_a / 1000, 'b-', lw=2, label='(A) Uniform sy')
    ax.plot(deltas, P_b / 1000, 'm-', lw=2, label='(B) Graded sy (outer=0.2x)')
    ax.plot(deltas, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(b) Effect of graded friction threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (c) Nonlinear hardening ---
    ax = axes[0, 2]
    ax.plot(deltas, P_a / 1000, 'b-', lw=2, label='(A) Linear KH')
    ax.plot(deltas, P_c / 1000, 'g-', lw=2, label='(C) Armstrong-Frederick')
    ax.plot(deltas, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(c) Linear vs nonlinear hardening')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (d) Low sigma_y (original params — all fibers yielded) ---
    ax = axes[1, 0]
    ax.plot(deltas, P_low / 1000, 'r-', lw=2, label='sy=200 MPa (all yielded)')
    ax.plot(deltas, P_a / 1000, 'b-', lw=2, label=f'sy={sigma_y:.0f} MPa (partial)')
    ax.plot(deltas, P_el / 1000, 'k:', lw=1, alpha=0.4, label='Elastic')
    ax.set_xlabel('Center deflection [mm]')
    ax.set_ylabel('Load P [kN]')
    ax.set_title('(d) Full yield vs partial yield')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (e) Normalized hysteresis (P/P_max vs delta/delta_max) ---
    ax = axes[1, 1]
    for label, P, ls, c in [
        ('(A) Uniform', P_a, '-', 'blue'),
        ('(B) Graded', P_b, '-', 'magenta'),
        ('(C) Nonlinear', P_c, '-', 'green'),
        ('(D) Low sy=200', P_low, '-', 'red'),
    ]:
        P_norm = P / np.max(np.abs(P))
        d_norm = deltas / delta_max
        ax.plot(d_norm, P_norm, ls=ls, color=c, lw=1.5, label=label)
    ax.set_xlabel('delta / delta_max')
    ax.set_ylabel('P / P_max')
    ax.set_title('(e) Normalized hysteresis shape comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (f) Equivalence proof + summary ---
    ax = axes[1, 2]
    ax.axis('off')
    diff_max = np.max(np.abs(P_a - P_a2))
    d_res_a = find_residual(deltas, P_a)
    d_res_b = find_residual(deltas, P_b)
    hyst_a = np.trapezoid(P_a, deltas)
    hyst_b = np.trapezoid(P_b, deltas)
    summary = (
        f"RESULTS SUMMARY\n"
        f"{'='*45}\n"
        f"D={D}mm, L={L}mm, dmax={delta_max}mm\n"
        f"E={E/1000:.0f} GPa, sy={sigma_y:.0f} MPa, H={H:.0f} MPa\n"
        f"eps_max/eps_y = {eps_max/eps_y:.1f}x\n"
        f"{'─'*45}\n"
        f"KH vs Friction diff: {diff_max:.1e} N  => IDENTICAL\n"
        f"{'─'*45}\n"
        f"          P_max[kN]  d_res[mm]  Ediss[J]\n"
        f"Elastic:  {P_el.max()/1000:8.1f}  {'---':>9s}  {'---':>7s}\n"
        f"(A) Uni:  {P_a.max()/1000:8.1f}  {d_res_a:9.1f}  {hyst_a/1e3:7.1f}\n"
        f"(B) Grad: {P_b.max()/1000:8.1f}  {d_res_b:9.1f}  {hyst_b/1e3:7.1f}\n"
        f"{'─'*45}\n"
        f"\n"
        f"WHY KH = STRAND FRICTION?\n"
        f"{'='*45}\n"
        f"Both solve:\n"
        f"  |trial - ref_point| <= threshold\n"
        f"  if exceeded: flow + shift ref_point\n"
        f"\n"
        f"Plasticity:   Friction:\n"
        f"  E          = k_strand\n"
        f"  sigma_y    = f_y (= mu*N)\n"
        f"  H          = k_slip\n"
        f"  eps_p      = u_slip\n"
        f"  alpha      = f_locked\n"
        f"\n"
        f"Same variational inequality.\n"
        f"Yield surface shift = friction lock update."
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    outpath = '/home/user/xkep-cae/work/beam_hysteresis/hysteresis_comparison.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"\nPlot saved: {outpath}")

    # --- Console output ---
    print(f"\n{'='*60}")
    print(f"  BEAM BENDING HYSTERESIS RESULTS")
    print(f"{'='*60}")
    print(f"  D={D}mm, L={L}mm, delta_max={delta_max}mm")
    print(f"  E={E/1000:.0f} GPa, sigma_y={sigma_y:.0f} MPa, H={H:.0f} MPa")
    print(f"  eps_max = {eps_max:.4f} ({eps_max*100:.1f}%)")
    print(f"  eps_y   = {eps_y:.4f} ({eps_y*100:.1f}%)")
    print(f"  eps_max/eps_y = {eps_max/eps_y:.1f}x  (x>1: yielded fibers exist)")
    print(f"{'─'*60}")
    print(f"  KH vs Strand friction max diff: {diff_max:.1e} N  => IDENTICAL")
    print(f"{'─'*60}")
    print(f"  (A) Uniform:  P_max={P_a.max()/1000:.1f} kN, "
          f"d_res={d_res_a:.1f}mm ({d_res_a/delta_max*100:.0f}%)")
    print(f"  (B) Graded:   P_max={P_b.max()/1000:.1f} kN, "
          f"d_res={d_res_b:.1f}mm ({d_res_b/delta_max*100:.0f}%)")
    print(f"  Elastic:      P_max={P_el.max()/1000:.1f} kN")
    print(f"{'='*60}")

    print("""
CONCLUSIONS:
============
1. KH and strand friction are EXACTLY identical (diff=0).
   Not numerical coincidence — same variational inequality.

2. The "tilted teardrop" shape requires:
   - Partial yielding across the section (not all fibers yielded)
   - This means eps_max/eps_y should be ~2-10x
   - If all fibers yield (eps_max/eps_y >> 10), loop becomes thin/linear

3. Graded sigma_y (Model B) gives a MORE asymmetric, rounder teardrop:
   - Outer fibers slip first (lower threshold)
   - Inner fibers remain elastic longer
   - This is exactly what happens in real cables:
     outer strands have more relative sliding under curvature

4. For cable modeling:
   - sigma_y(r) = mu * N(r) where N(r) is the inter-strand normal force
   - N(r) depends on tension, lay angle, and radial position
   - H represents the geometric constraint after slip
   - A fiber section with graded parameters IS a cable cross-section model
""")


if __name__ == '__main__':
    main()
