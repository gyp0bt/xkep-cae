"""Phase 4.1 弾塑性構成則の検証図生成.

テストとは独立のスクリプト。手動で実行:
  python tests/generate_verification_plots.py

出力先: docs/verification/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "verification"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def plot_stress_strain_isotropic():
    """等方硬化の応力-歪み曲線（解析解と数値解の比較）."""
    plt = _setup_matplotlib()
    from xkep_cae.core.state import PlasticState1D
    from xkep_cae.materials.plasticity_1d import IsotropicHardening, Plasticity1D

    E = 200_000.0
    sigma_y0 = 250.0
    H_iso = 1000.0
    plas = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y0, H_iso=H_iso))
    E_t = E * H_iso / (E + H_iso)
    eps_y = sigma_y0 / E

    # 数値解: 増分載荷
    eps_list = np.linspace(0, 5 * eps_y, 200)
    sigma_num = []
    state = PlasticState1D()
    for eps in eps_list:
        r = plas.return_mapping(eps, state)
        sigma_num.append(r.stress)
        state = r.state_new

    # 解析解
    sigma_ana = np.where(
        eps_list <= eps_y,
        E * eps_list,
        sigma_y0 + E_t * (eps_list - eps_y),
    )

    fig, ax = plt.subplots()
    ax.plot(eps_list * 100, sigma_ana, "k-", linewidth=2, label="Analytical (bilinear)")
    ax.plot(eps_list * 100, sigma_num, "r--", linewidth=1.5, label="Return mapping")
    ax.set_xlabel("Strain [%]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title(
        f"Isotropic Hardening: E={E:.0f}, "
        f"$\\sigma_{{y0}}$={sigma_y0:.0f}, $H_{{iso}}$={H_iso:.0f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stress_strain_isotropic.png")
    plt.close(fig)
    print("  -> stress_strain_isotropic.png")


def plot_hysteresis_loop():
    """繰返し荷重のヒステリシスループ."""
    plt = _setup_matplotlib()
    from xkep_cae.core.state import PlasticState1D
    from xkep_cae.materials.plasticity_1d import (
        IsotropicHardening,
        KinematicHardening,
        Plasticity1D,
    )

    E = 200_000.0
    sigma_y0 = 250.0
    H_iso = 500.0
    C_kin = 3000.0

    plas = Plasticity1D(
        E=E,
        iso=IsotropicHardening(sigma_y0=sigma_y0, H_iso=H_iso),
        kin=KinematicHardening(C_kin=C_kin),
    )

    eps_y = sigma_y0 / E
    eps_max = 4.0 * eps_y

    # 繰返し歪み履歴: +max → -max → +max → -max → +max
    n_pts = 100
    eps_history = np.concatenate([
        np.linspace(0, eps_max, n_pts),
        np.linspace(eps_max, -eps_max, 2 * n_pts),
        np.linspace(-eps_max, eps_max, 2 * n_pts),
        np.linspace(eps_max, -eps_max, 2 * n_pts),
        np.linspace(-eps_max, eps_max, 2 * n_pts),
    ])

    sigma_history = []
    state = PlasticState1D()
    for eps in eps_history:
        r = plas.return_mapping(eps, state)
        sigma_history.append(r.stress)
        state = r.state_new

    fig, ax = plt.subplots()
    ax.plot(np.array(eps_history) * 100, sigma_history, "b-", linewidth=1)
    ax.set_xlabel("Strain [%]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title(
        f"Hysteresis Loop: "
        f"$\\sigma_{{y0}}$={sigma_y0:.0f}, "
        f"$H_{{iso}}$={H_iso:.0f}, $C_{{kin}}$={C_kin:.0f}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hysteresis_loop.png")
    plt.close(fig)
    print("  -> hysteresis_loop.png")


def plot_bauschinger_comparison():
    """バウシンガー効果: 等方硬化 vs 移動硬化."""
    plt = _setup_matplotlib()
    from xkep_cae.core.state import PlasticState1D
    from xkep_cae.materials.plasticity_1d import (
        IsotropicHardening,
        KinematicHardening,
        Plasticity1D,
    )

    E = 200_000.0
    sigma_y0 = 250.0
    H = 2000.0

    plas_iso = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y0, H_iso=H))
    plas_kin = Plasticity1D(
        E=E,
        iso=IsotropicHardening(sigma_y0=sigma_y0, H_iso=0.0),
        kin=KinematicHardening(C_kin=H),
    )

    eps_y = sigma_y0 / E
    n_pts = 100
    eps_history = np.concatenate([
        np.linspace(0, 3 * eps_y, n_pts),
        np.linspace(3 * eps_y, -3 * eps_y, 3 * n_pts),
    ])

    fig, ax = plt.subplots()
    for plas, label, color in [
        (plas_iso, f"Isotropic ($H_{{iso}}$={H})", "blue"),
        (plas_kin, f"Kinematic ($C_{{kin}}$={H})", "red"),
    ]:
        sigma_list = []
        state = PlasticState1D()
        for eps in eps_history:
            r = plas.return_mapping(eps, state)
            sigma_list.append(r.stress)
            state = r.state_new
        ax.plot(
            np.array(eps_history) * 100, sigma_list,
            color=color, linewidth=1.5, label=label,
        )

    ax.axhline(y=sigma_y0, color="gray", linestyle=":", alpha=0.5, label=f"$\\sigma_{{y0}}$={sigma_y0}")
    ax.axhline(y=-sigma_y0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Strain [%]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title("Bauschinger Effect: Isotropic vs Kinematic Hardening")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bauschinger_comparison.png")
    plt.close(fig)
    print("  -> bauschinger_comparison.png")


def plot_load_displacement_bar():
    """弾塑性棒の荷重-変位曲線（NR 結果 vs 解析解）."""
    plt = _setup_matplotlib()
    from xkep_cae.core.state import CosseratPlasticState
    from xkep_cae.elements.beam_cosserat import CosseratRod, assemble_cosserat_beam_plastic
    from xkep_cae.materials.beam_elastic import BeamElastic1D
    from xkep_cae.materials.plasticity_1d import IsotropicHardening, Plasticity1D
    from xkep_cae.sections.beam import BeamSection
    from xkep_cae.solver import newton_raphson

    E = 200_000.0
    nu = 0.3
    sigma_y0 = 250.0
    H_iso = 1000.0
    E_t = E * H_iso / (E + H_iso)

    sec = BeamSection.rectangle(10.0, 20.0)
    mat = BeamElastic1D(E=E, nu=nu)
    rod = CosseratRod(section=sec, integration_scheme="uniform", n_gauss=1)
    plas = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y0, H_iso=H_iso))

    n_elems = 4
    L = 100.0
    n_nodes = n_elems + 1
    total_dof = n_nodes * 6
    fixed_dofs = np.arange(6)

    P_y = sigma_y0 * sec.A
    P_max = 2.0 * P_y

    # NR 解析（載荷 → 除荷）
    n_load = 10
    n_unload = 5
    load_factors = np.concatenate([
        np.linspace(0, 1, n_load + 1)[1:],
        np.linspace(1, 0, n_unload + 1)[1:],
    ])
    P_history = load_factors * P_max

    states = [CosseratPlasticState() for _ in range(n_elems)]
    u = np.zeros(total_dof)
    u_tip_hist = [0.0]
    P_tip_hist = [0.0]
    states_trial = None

    for P in P_history:
        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P

        def _fint(u_, _st=states):
            nonlocal states_trial
            _, f, st = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_, _st, plas,
                stiffness=False, internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_, _st=states):
            K, _, _ = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_, _st, plas,
                stiffness=True, internal_force=False,
            )
            return sp.csr_matrix(K)

        result = newton_raphson(
            f_ext, fixed_dofs, _Kt, _fint,
            n_load_steps=1, u0=u, show_progress=False,
        )
        u = result.u
        states = [s.copy() for s in states_trial]
        u_tip_hist.append(u[6 * n_elems])
        P_tip_hist.append(P)

    # 解析解
    u_y = sigma_y0 / E * L
    P_ana = np.linspace(0, P_max, 100)
    u_ana_load = np.where(
        P_ana <= P_y,
        P_ana * L / (E * sec.A),
        u_y + (P_ana / sec.A - sigma_y0) / E_t * L,
    )
    # 除荷線（P_max からの弾性除荷）
    u_at_Pmax = u_y + (P_max / sec.A - sigma_y0) / E_t * L
    P_unload = np.linspace(P_max, 0, 50)
    u_unload = u_at_Pmax + (P_unload - P_max) * L / (E * sec.A)

    fig, ax = plt.subplots()
    ax.plot(u_ana_load, P_ana / 1000, "k-", linewidth=2, label="Analytical (loading)")
    ax.plot(u_unload, P_unload / 1000, "k--", linewidth=2, label="Analytical (unloading)")
    ax.plot(u_tip_hist, np.array(P_tip_hist) / 1000, "ro-", markersize=4, label="NR result")
    ax.axhline(y=P_y / 1000, color="gray", linestyle=":", alpha=0.5, label=f"$P_y$={P_y / 1000:.1f} kN")
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Applied load [kN]")
    ax.set_title(
        f"Elastoplastic Bar: L={L}, A={sec.A}, "
        f"E={E:.0f}, $\\sigma_y$={sigma_y0:.0f}, $H$={H_iso:.0f}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "load_displacement_bar.png")
    plt.close(fig)
    print("  -> load_displacement_bar.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating verification plots to {OUTPUT_DIR}/")
    plot_stress_strain_isotropic()
    plot_hysteresis_loop()
    plot_bauschinger_comparison()
    plot_load_displacement_bar()
    print("Done.")


if __name__ == "__main__":
    main()
