"""バリデーション検証図の一括生成.

全 Phase のバリデーションテストに対応する検証図を生成する。
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

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
    return plt


# =====================================================================
# Phase 2.1-2.2: 梁要素 — Euler-Bernoulli vs Timoshenko
# =====================================================================


def plot_cantilever_eb_timo():
    """片持ち梁のたわみ分布: EB vs Timoshenko vs 解析解."""
    plt = _setup_matplotlib()
    from xkep_cae.bc import apply_dirichlet
    from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_global
    from xkep_cae.elements.beam_timo2d import timo_beam2d_ke_global
    from xkep_cae.solver import solve_displacement

    E = 200e3
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    b, h = 10.0, 10.0
    A = b * h
    I_val = b * h**3 / 12.0
    L_total = 100.0
    P = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (n_elems, label_suffix) in enumerate([(10, "10 elems"), (40, "40 elems")]):
        n_nodes = n_elems + 1
        s = np.linspace(0, L_total, n_nodes)
        nodes = np.column_stack([s, np.zeros(n_nodes)])
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
        ndof = 3 * n_nodes

        results = {}
        for name, ke_func in [
            ("EB", lambda c: eb_beam2d_ke_global(c, E, A, I_val)),
            ("Timoshenko", lambda c: timo_beam2d_ke_global(c, E, A, I_val, kappa, G)),
        ]:
            K = np.zeros((ndof, ndof))
            for e in conn:
                n1, n2 = e
                coords = nodes[[n1, n2]]
                Ke = ke_func(coords)
                edofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])
                for ii in range(6):
                    for jj in range(6):
                        K[edofs[ii], edofs[jj]] += Ke[ii, jj]

            f = np.zeros(ndof)
            f[3 * n_elems + 1] = P
            K_sp = sp.csr_matrix(K)
            Kbc, fbc = apply_dirichlet(K_sp, f, np.array([0, 1, 2]))
            u, _ = solve_displacement(Kbc, fbc, show_progress=False)
            uy = np.array([u[3 * i + 1] for i in range(n_nodes)])
            results[name] = uy

        # Analytical solution
        x = np.linspace(0, L_total, 200)
        delta_eb = P * x**2 / (6 * E * I_val) * (3 * L_total - x)
        delta_timo = delta_eb + P * x / (kappa * G * A)

        ax = axes[ax_idx]
        ax.plot(x, delta_eb, "k-", linewidth=2, label="Analytical (EB)")
        ax.plot(x, delta_timo, "b-", linewidth=2, label="Analytical (Timoshenko)")
        ax.plot(s, results["EB"], "rs", markersize=5, label="FEM (EB)")
        ax.plot(s, results["Timoshenko"], "b^", markersize=5, label="FEM (Timoshenko)")
        ax.set_xlabel("Position x [mm]")
        ax.set_ylabel("Deflection [mm]")
        ax.set_title(f"Cantilever Deflection ({label_suffix})")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Cantilever Beam: P={P} N, L={L_total} mm, b*h={b}*{h} mm, E={E:.0f} MPa, nu={nu}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cantilever_eb_timo.png")
    plt.close(fig)
    print("  -> cantilever_eb_timo.png")


# =====================================================================
# Phase 2.6: 数値試験フレームワーク — 解析解との精度比較
# =====================================================================


def plot_numerical_tests_accuracy():
    """数値試験フレームワークの全試験の解析解との相対誤差."""
    plt = _setup_matplotlib()
    from xkep_cae.numerical_tests.core import NumericalTestConfig
    from xkep_cae.numerical_tests.runner import run_test

    E = 200e3
    nu = 0.3
    L = 100.0
    n_elems = 20
    P = 1000.0
    T = 500.0
    rect = {"b": 10.0, "h": 20.0}
    circ = {"d": 10.0}

    configs = [
        (
            "3pt-Bend\n(EB2D)",
            NumericalTestConfig(
                name="bend3p",
                beam_type="eb2d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
            ),
        ),
        (
            "3pt-Bend\n(Timo2D)",
            NumericalTestConfig(
                name="bend3p",
                beam_type="timo2d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
            ),
        ),
        (
            "3pt-Bend\n(Timo3D)",
            NumericalTestConfig(
                name="bend3p",
                beam_type="timo3d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
            ),
        ),
        (
            "3pt-Bend\n(Cosserat)",
            NumericalTestConfig(
                name="bend3p",
                beam_type="cosserat",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
            ),
        ),
        (
            "4pt-Bend\n(Timo2D)",
            NumericalTestConfig(
                name="bend4p",
                beam_type="timo2d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
                load_span=25.0,
            ),
        ),
        (
            "4pt-Bend\n(Timo3D)",
            NumericalTestConfig(
                name="bend4p",
                beam_type="timo3d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
                load_span=25.0,
            ),
        ),
        (
            "Tensile\n(Timo2D)",
            NumericalTestConfig(
                name="tensile",
                beam_type="timo2d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
            ),
        ),
        (
            "Tensile\n(Cosserat)",
            NumericalTestConfig(
                name="tensile",
                beam_type="cosserat",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=P,
                section_shape="rectangle",
                section_params=rect,
            ),
        ),
        (
            "Torsion\n(Timo3D)",
            NumericalTestConfig(
                name="torsion",
                beam_type="timo3d",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=T,
                section_shape="circle",
                section_params=circ,
            ),
        ),
        (
            "Torsion\n(Cosserat)",
            NumericalTestConfig(
                name="torsion",
                beam_type="cosserat",
                E=E,
                nu=nu,
                length=L,
                n_elems=n_elems,
                load_value=T,
                section_shape="circle",
                section_params=circ,
            ),
        ),
    ]

    labels = []
    errors = []
    for label, cfg in configs:
        result = run_test(cfg)
        labels.append(label)
        errors.append(result.relative_error if result.relative_error is not None else 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(labels))
    colors = ["#2196F3" if e < 1e-6 else "#4CAF50" if e < 1e-3 else "#FF9800" for e in errors]
    bars = ax.bar(x_pos, errors, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1.0)
    ax.axhline(y=1e-10, color="green", linestyle="--", alpha=0.5, label="Machine precision")
    ax.axhline(y=1e-3, color="orange", linestyle="--", alpha=0.5, label="0.1%")
    ax.legend(fontsize=8)
    ax.set_title("Numerical Test Framework: Relative Error vs Analytical (20 elems)")

    # バーの上に値を表示
    for bar, err in zip(bars, errors, strict=True):
        if err > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                err * 1.5,
                f"{err:.1e}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numerical_tests_accuracy.png")
    plt.close(fig)
    print("  -> numerical_tests_accuracy.png")


# =====================================================================
# Phase 3: Euler Elastica — 端モーメントによる変形形状
# =====================================================================


def plot_euler_elastica_moment():
    """Euler elastica: 端モーメントの変形形状と解析解."""
    plt = _setup_matplotlib()
    from xkep_cae.elements.beam_cosserat import CosseratRod, assemble_cosserat_beam
    from xkep_cae.materials.beam_elastic import BeamElastic1D
    from xkep_cae.sections.beam import BeamSection
    from xkep_cae.solver import newton_raphson

    E = 1.0e6
    nu = 0.0
    L = 10.0
    _b, _h = 0.1, 0.1
    sec = BeamSection.rectangle(_b, _h)
    EI = E * sec.Iz
    mat = BeamElastic1D(E=E, nu=nu)

    fig, ax = plt.subplots(figsize=(10, 8))

    theta_cases = [
        (np.pi / 4, "π/4", 20, 5),
        (np.pi / 2, "π/2", 20, 10),
        (np.pi, "π", 20, 20),
        (3 * np.pi / 2, "3π/2", 30, 30),
        (2 * np.pi, "2π", 40, 40),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (theta, label, n_elems, n_steps), color in zip(theta_cases, colors, strict=True):
        M = theta * EI / L
        rod = CosseratRod(section=sec, nonlinear=True)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _Kt(u, _rod=rod, _n=n_elems):
            K, _ = assemble_cosserat_beam(_n, L, _rod, mat, u, stiffness=True, internal_force=False)
            return sp.csr_matrix(K)

        def _fint(u, _rod=rod, _n=n_elems):
            _, f = assemble_cosserat_beam(_n, L, _rod, mat, u, stiffness=False, internal_force=True)
            return f

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 5] = M
        fixed_dofs = np.arange(6)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _Kt,
            _fint,
            n_load_steps=n_steps,
            max_iter=100,
            show_progress=False,
        )
        u = result.u

        # 変形形状の抽出
        x_def = np.zeros(n_nodes)
        y_def = np.zeros(n_nodes)
        Le = L / n_elems
        for i in range(n_nodes):
            x_def[i] = i * Le + u[6 * i + 0]
            y_def[i] = u[6 * i + 1]

        # 解析解（円弧）
        R_curv = EI / M
        s_arc = np.linspace(0, L, 200)
        x_ana = R_curv * np.sin(s_arc / R_curv)
        y_ana = R_curv * (1.0 - np.cos(s_arc / R_curv))

        ax.plot(x_ana, y_ana, "-", color=color, linewidth=1.5, alpha=0.7)
        ax.plot(
            x_def,
            y_def,
            "o-",
            color=color,
            markersize=3,
            linewidth=0.8,
            label=f"theta={label} (FEM {n_elems} elems)",
        )

    # θ=2πの解析解円を追加（参考）
    R = EI / (2 * np.pi * EI / L)
    theta_arr = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        R * np.sin(theta_arr),
        R * (1 - np.cos(theta_arr)),
        "k--",
        linewidth=1,
        alpha=0.3,
        label="Analytical (arc)",
    )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Euler Elastica: End Moment Deformed Shapes (solid=analytical, marker=FEM)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "euler_elastica_moment.png")
    plt.close(fig)
    print("  -> euler_elastica_moment.png")


# =====================================================================
# Phase 3: Euler Elastica — 先端荷重
# =====================================================================


def plot_euler_elastica_tip_load():
    """Euler elastica: 先端荷重の変位 vs Mattiasson参照値."""
    plt = _setup_matplotlib()
    from xkep_cae.elements.beam_cosserat import CosseratRod, assemble_cosserat_beam
    from xkep_cae.materials.beam_elastic import BeamElastic1D
    from xkep_cae.sections.beam import BeamSection
    from xkep_cae.solver import newton_raphson

    E = 1.0e6
    nu = 0.0
    L = 10.0
    _b, _h = 0.1, 0.1
    sec = BeamSection.rectangle(_b, _h)
    EI = E * sec.Iz
    mat = BeamElastic1D(E=E, nu=nu)

    # Mattiasson参照値: (alpha, dx/L, dy/L)
    reference = {
        1: (0.05634, 0.30174),
        2: (0.16056, 0.49349),
        5: (0.38756, 0.71384),
        10: (0.55494, 0.81066),
    }

    alpha_list = [1, 2, 5, 10]
    dx_num_list = []
    dy_num_list = []
    dx_ref_list = []
    dy_ref_list = []

    for alpha in alpha_list:
        P = alpha * EI / L**2
        n_elems = 30 if alpha <= 5 else 40
        n_steps = 20 if alpha <= 2 else 30 if alpha <= 5 else 40
        rod = CosseratRod(section=sec, nonlinear=True)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _Kt(u, _rod=rod, _n=n_elems):
            K, _ = assemble_cosserat_beam(_n, L, _rod, mat, u, stiffness=True, internal_force=False)
            return sp.csr_matrix(K)

        def _fint(u, _rod=rod, _n=n_elems):
            _, f = assemble_cosserat_beam(_n, L, _rod, mat, u, stiffness=False, internal_force=True)
            return f

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 1] = P
        fixed_dofs = np.arange(6)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _Kt,
            _fint,
            n_load_steps=n_steps,
            max_iter=100,
            show_progress=False,
        )
        u = result.u
        dx = -u[6 * n_elems + 0] / L
        dy = u[6 * n_elems + 1] / L
        dx_num_list.append(dx)
        dy_num_list.append(dy)
        dx_ref, dy_ref = reference[alpha]
        dx_ref_list.append(dx_ref)
        dy_ref_list.append(dy_ref)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: δy/L
    ax = axes[0]
    ax.plot(alpha_list, dy_ref_list, "ks-", markersize=8, linewidth=2, label="Mattiasson exact")
    ax.plot(
        alpha_list, dy_num_list, "ro--", markersize=8, linewidth=1.5, label="FEM (Cosserat rod)"
    )
    ax.set_xlabel("alpha = PL^2/EI")
    ax.set_ylabel("dy / L")
    ax.set_title("Tip Deflection")
    ax.legend()

    # Right: dx/L (shortening)
    ax = axes[1]
    ax.plot(alpha_list, dx_ref_list, "ks-", markersize=8, linewidth=2, label="Mattiasson exact")
    ax.plot(
        alpha_list, dx_num_list, "ro--", markersize=8, linewidth=1.5, label="FEM (Cosserat rod)"
    )
    ax.set_xlabel("alpha = PL^2/EI")
    ax.set_ylabel("dx / L (shortening)")
    ax.set_title("Tip Shortening")
    ax.legend()

    fig.suptitle("Euler Elastica: Tip Load -- FEM vs Mattiasson Exact", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "euler_elastica_tip_load.png")
    plt.close(fig)
    print("  -> euler_elastica_tip_load.png")


# =====================================================================
# Phase 2.3: 3D梁 — ねじり・曲げ解析解比較
# =====================================================================


def plot_beam3d_torsion_bending():
    """3D Timoshenko梁: ねじり角と曲げたわみの解析解比較."""
    plt = _setup_matplotlib()
    from xkep_cae.bc import apply_dirichlet
    from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
    from xkep_cae.sections.beam import BeamSection
    from xkep_cae.solver import solve_displacement

    E = 200e3
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    L_total = 100.0
    P = 1.0
    T = 100.0

    sec = BeamSection.rectangle(10.0, 20.0)
    nu = 0.3
    kappa = sec.cowper_kappa_z(nu)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: bending deflection ---
    ax = axes[0]
    n_elems_list = [4, 8, 16, 32]
    delta_fem_list = []
    delta_ana = P * L_total**3 / (3.0 * E * sec.Iz) + P * L_total / (kappa * G * sec.A)

    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        s = np.linspace(0, L_total, n_nodes)
        nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
        ndof = 6 * n_nodes
        K = np.zeros((ndof, ndof))
        for e in conn:
            n1, n2 = e
            coords = nodes[[n1, n2]]
            ky = sec.cowper_kappa_y(nu)
            kz = sec.cowper_kappa_z(nu)
            Ke = timo_beam3d_ke_global(coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, ky, kz)
            edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
            for ii in range(12):
                for jj in range(12):
                    K[edofs[ii], edofs[jj]] += Ke[ii, jj]

        f = np.zeros(ndof)
        f[6 * n_elems + 2] = P  # z方向荷重 → Iz曲げ
        K_sp = sp.csr_matrix(K)
        Kbc, fbc = apply_dirichlet(K_sp, f, np.arange(6))
        u, _ = solve_displacement(Kbc, fbc, show_progress=False)
        delta_fem_list.append(u[6 * n_elems + 2])

    ax.axhline(
        y=delta_ana, color="k", linestyle="-", linewidth=2, label=f"Analytical d={delta_ana:.6f}"
    )
    ax.plot(n_elems_list, delta_fem_list, "ro-", markersize=6, label="FEM (Timoshenko 3D)")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Tip deflection [mm]")
    ax.set_title("3D Cantilever: Mesh convergence")
    ax.legend(fontsize=8)

    # --- 右: ねじり角 ---
    ax = axes[1]
    theta_fem_list = []
    theta_ana = T * L_total / (G * sec.J)

    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        s = np.linspace(0, L_total, n_nodes)
        nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
        ndof = 6 * n_nodes
        K = np.zeros((ndof, ndof))
        for e in conn:
            n1, n2 = e
            coords = nodes[[n1, n2]]
            ky = sec.cowper_kappa_y(nu)
            kz = sec.cowper_kappa_z(nu)
            Ke = timo_beam3d_ke_global(coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, ky, kz)
            edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
            for ii in range(12):
                for jj in range(12):
                    K[edofs[ii], edofs[jj]] += Ke[ii, jj]

        f = np.zeros(ndof)
        f[6 * n_elems + 3] = T  # ねじりモーメント
        K_sp = sp.csr_matrix(K)
        Kbc, fbc = apply_dirichlet(K_sp, f, np.arange(6))
        u, _ = solve_displacement(Kbc, fbc, show_progress=False)
        theta_fem_list.append(u[6 * n_elems + 3])

    ax.axhline(
        y=theta_ana,
        color="k",
        linestyle="-",
        linewidth=2,
        label=f"Analytical theta={theta_ana:.6f}",
    )
    ax.plot(n_elems_list, theta_fem_list, "bs-", markersize=6, label="FEM (Timoshenko 3D)")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Tip twist angle [rad]")
    ax.set_title("3D Torsion: Mesh convergence")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"3D Timoshenko Beam: b*h=10*20, E={E:.0f}, nu={nu}, L={L_total}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beam3d_torsion_bending.png")
    plt.close(fig)
    print("  -> beam3d_torsion_bending.png")


# =====================================================================
# Phase 2.5: Cosserat rod — 線形テスト（軸力・曲げ収束）
# =====================================================================


def plot_cosserat_convergence():
    """Cosserat rod: メッシュ収束テスト（軸力、ねじり、曲げ）."""
    plt = _setup_matplotlib()
    from scipy.sparse.linalg import spsolve

    from xkep_cae.elements.beam_cosserat import CosseratRod, assemble_cosserat_beam
    from xkep_cae.materials.beam_elastic import BeamElastic1D
    from xkep_cae.sections.beam import BeamSection

    E = 200_000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    L = 100.0
    sec = BeamSection.rectangle(10.0, 20.0)
    mat = BeamElastic1D(E=E, nu=nu)

    n_elems_list = [2, 4, 8, 16, 32]

    # 解析解
    P = 1000.0
    T = 500.0
    delta_axial_ana = P * L / (E * sec.A)
    theta_torsion_ana = T * L / (G * sec.J)
    kz = sec.cowper_kappa_z(nu)
    delta_bend_ana = P * L**3 / (3 * E * sec.Iz) + P * L / (kz * G * sec.A)

    tests = [
        ("Axial tension", 0, P, delta_axial_ana, "d [mm]"),
        ("Torsion", 3, T, theta_torsion_ana, "theta [rad]"),
        ("Bending (z-dir)", 2, P, delta_bend_ana, "d [mm]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (title, load_dof_offset, load_val, ana_val, _ylabel) in zip(axes, tests, strict=True):
        errors = []
        for n_elems in n_elems_list:
            rod = CosseratRod(section=sec)
            n_nodes = n_elems + 1
            total_dof = n_nodes * 6

            K, _ = assemble_cosserat_beam(n_elems, L, rod, mat, np.zeros(total_dof))
            f_ext = np.zeros(total_dof)
            f_ext[6 * n_elems + load_dof_offset] = load_val

            fixed = np.arange(6)
            free = np.array([d for d in range(total_dof) if d not in fixed])
            K_ff = K[np.ix_(free, free)]
            f_ff = f_ext[free]
            u_free = spsolve(sp.csr_matrix(K_ff), f_ff)
            u = np.zeros(total_dof)
            u[free] = u_free
            val = u[6 * n_elems + load_dof_offset]
            errors.append(abs(val - ana_val) / abs(ana_val))

        ax.loglog(n_elems_list, errors, "ro-", markersize=6, linewidth=1.5)
        ax.set_xlabel("Number of elements")
        ax.set_ylabel("Relative error")
        ax.set_title(title)
        # h^2 convergence reference line
        ref = errors[0] * (np.array(n_elems_list) / n_elems_list[0]) ** (-2.0)
        ax.loglog(n_elems_list, ref, "k--", alpha=0.3, label="O(h^2)")
        ax.legend(fontsize=8)

    fig.suptitle("Cosserat Rod: Mesh Convergence (relative error vs analytical)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cosserat_convergence.png")
    plt.close(fig)
    print("  -> cosserat_convergence.png")


# =====================================================================
# Phase 4.1: 弾塑性 — 既存プロット（変更なし）
# =====================================================================


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
        f"Isotropic Hardening: E={E:.0f}, $\\sigma_{{y0}}$={sigma_y0:.0f}, $H_{{iso}}$={H_iso:.0f}"
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
    eps_history = np.concatenate(
        [
            np.linspace(0, eps_max, n_pts),
            np.linspace(eps_max, -eps_max, 2 * n_pts),
            np.linspace(-eps_max, eps_max, 2 * n_pts),
            np.linspace(eps_max, -eps_max, 2 * n_pts),
            np.linspace(-eps_max, eps_max, 2 * n_pts),
        ]
    )

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
    eps_history = np.concatenate(
        [
            np.linspace(0, 3 * eps_y, n_pts),
            np.linspace(3 * eps_y, -3 * eps_y, 3 * n_pts),
        ]
    )

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
            np.array(eps_history) * 100,
            sigma_list,
            color=color,
            linewidth=1.5,
            label=label,
        )

    ax.axhline(
        y=sigma_y0, color="gray", linestyle=":", alpha=0.5, label=f"$\\sigma_{{y0}}$={sigma_y0}"
    )
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
    load_factors = np.concatenate(
        [
            np.linspace(0, 1, n_load + 1)[1:],
            np.linspace(1, 0, n_unload + 1)[1:],
        ]
    )
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
                n_elems,
                L,
                rod,
                mat,
                u_,
                _st,
                plas,
                stiffness=False,
                internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_, _st=states):
            K, _, _ = assemble_cosserat_beam_plastic(
                n_elems,
                L,
                rod,
                mat,
                u_,
                _st,
                plas,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _Kt,
            _fint,
            n_load_steps=1,
            u0=u,
            show_progress=False,
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
    ax.axhline(
        y=P_y / 1000, color="gray", linestyle=":", alpha=0.5, label=f"$P_y$={P_y / 1000:.1f} kN"
    )
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


# =====================================================================
# Phase 4.2: ファイバーモデル（曲げの塑性化）
# =====================================================================


def plot_fiber_moment_curvature():
    """ファイバーモデルのモーメント-曲率曲線.

    矩形断面 (b=10, h=20) に等方硬化弾塑性材料を適用し、
    弾性→弾塑性→全塑性の遷移をプロットする。

    解析解:
      M_y = sigma_y * b*h^2/6  (弾性限界)
      M_p = sigma_y * b*h^2/4  (全塑性)
      shape factor = 1.5 (矩形断面)
    """
    plt = _setup_matplotlib()

    from xkep_cae.core.state import CosseratFiberPlasticState
    from xkep_cae.elements.beam_cosserat import (
        _compute_generalized_stress_fiber,
        _cosserat_constitutive_matrix,
    )
    from xkep_cae.materials.plasticity_1d import (
        IsotropicHardening,
        Plasticity1D,
    )
    from xkep_cae.sections.fiber import FiberSection

    E = 200_000.0
    nu = 0.3
    sigma_y = 250.0
    b, h = 10.0, 20.0
    G = E / (2.0 * (1.0 + nu))

    # --- 完全弾塑性 (H=0) ---
    fs = FiberSection.rectangle(b, h, ny=4, nz=80)
    kappa_cowper = fs.cowper_kappa_y(nu)
    C_elastic = _cosserat_constitutive_matrix(
        E,
        G,
        fs.A,
        fs.Iy,
        fs.Iz,
        fs.J,
        kappa_cowper,
        kappa_cowper,
    )
    plas_pp = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y, H_iso=0.0))

    kappa_y_limit = sigma_y / (E * h / 2.0)
    kappas = np.linspace(0, 15 * kappa_y_limit, 200)
    moments_pp = []
    state = CosseratFiberPlasticState.create(fs.n_fibers)
    for kap in kappas:
        strain = np.array([0.0, 0.0, 0.0, 0.0, kap, 0.0])
        stress, _, state_new = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas_pp,
            state,
            fs,
        )
        moments_pp.append(stress[4])
        state = state_new

    # --- 等方硬化 (H=1000) ---
    plas_iso = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y, H_iso=1000.0))
    moments_iso = []
    state = CosseratFiberPlasticState.create(fs.n_fibers)
    for kap in kappas:
        strain = np.array([0.0, 0.0, 0.0, 0.0, kap, 0.0])
        stress, _, state_new = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas_iso,
            state,
            fs,
        )
        moments_iso.append(stress[4])
        state = state_new

    kappas = np.array(kappas)
    moments_pp = np.array(moments_pp)
    moments_iso = np.array(moments_iso)

    # 解析解
    W_el = b * h**2 / 6.0
    W_pl = b * h**2 / 4.0
    M_y = sigma_y * W_el
    M_p = sigma_y * W_pl
    M_elastic = E * fs.Iy * kappas

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        kappas / kappa_y_limit,
        M_elastic / M_y,
        "k--",
        lw=1,
        label="Elastic (EI*kappa)",
    )
    ax.plot(
        kappas / kappa_y_limit,
        moments_pp / M_y,
        "b-",
        lw=2,
        label="Fiber: perfectly plastic (H=0)",
    )
    ax.plot(
        kappas / kappa_y_limit,
        moments_iso / M_y,
        "r-",
        lw=2,
        label="Fiber: isotropic hardening (H=1000)",
    )
    ax.axhline(M_p / M_y, color="gray", ls=":", lw=1, label=f"M_p/M_y = {M_p / M_y:.2f}")
    ax.axhline(1.0, color="gray", ls="-.", lw=1, alpha=0.5)
    ax.set_xlabel("kappa / kappa_y")
    ax.set_ylabel("M / M_y")
    ax.set_title("Fiber Model: Moment-Curvature for Rectangle (b=10, h=20)")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fiber_moment_curvature.png")
    plt.close(fig)
    print("  -> fiber_moment_curvature.png")


def plot_fiber_cantilever_load_displacement():
    """ファイバーモデル片持ち梁の荷重-変位曲線.

    先端モーメントを増分載荷し、NR法で解いた結果をプロットする。
    弾性解析解と比較して塑性化の効果を確認する。
    """
    plt = _setup_matplotlib()

    from xkep_cae.core.state import CosseratFiberPlasticState
    from xkep_cae.elements.beam_cosserat import (
        CosseratRod,
        assemble_cosserat_beam_fiber,
    )
    from xkep_cae.materials.beam_elastic import BeamElastic1D
    from xkep_cae.materials.plasticity_1d import (
        IsotropicHardening,
        Plasticity1D,
    )
    from xkep_cae.sections.fiber import FiberSection
    from xkep_cae.solver import newton_raphson

    E = 200_000.0
    nu = 0.3
    sigma_y = 250.0
    H_iso = 1000.0
    b, h = 10.0, 20.0
    L = 100.0
    n_elems = 8

    fs = FiberSection.rectangle(b, h, ny=4, nz=40)
    rod = CosseratRod(section=fs, integration_scheme="uniform", n_gauss=1)
    mat = BeamElastic1D(E=E, nu=nu)
    plas = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y, H_iso=H_iso))

    n_nodes = n_elems + 1
    total_dof = n_nodes * 6
    fixed_dofs = np.arange(6)

    W_el = b * h**2 / 6.0
    M_y = sigma_y * W_el
    M_max = 2.0 * M_y
    n_steps = 20

    moments = []
    theta_tips = []
    n_gp = n_elems * rod.n_gauss
    states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]
    u = np.zeros(total_dof)

    for step in range(1, n_steps + 1):
        lam = step / n_steps
        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 4] = lam * M_max

        states_trial = [None] * n_gp

        def _fint(u_, _states=states):
            nonlocal states_trial
            _, f, st = assemble_cosserat_beam_fiber(
                n_elems,
                L,
                rod,
                mat,
                u_,
                _states,
                plas,
                fs,
                stiffness=False,
                internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_, _states=states):
            K, _, _ = assemble_cosserat_beam_fiber(
                n_elems,
                L,
                rod,
                mat,
                u_,
                _states,
                plas,
                fs,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _Kt,
            _fint,
            n_load_steps=1,
            u0=u,
            show_progress=False,
        )
        u = result.u
        states = [s.copy() for s in states_trial]

        theta_tips.append(u[6 * n_elems + 4])
        moments.append(lam * M_max)

    moments = np.array(moments)
    theta_tips = np.array(theta_tips)

    # 弾性解析解
    Iy_exact = b * h**3 / 12.0
    theta_elastic = moments * L / (E * Iy_exact)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        theta_elastic * 1000,
        moments / M_y,
        "k--",
        lw=1.5,
        label="Elastic (analytical)",
    )
    ax.plot(
        theta_tips * 1000,
        moments / M_y,
        "ro-",
        ms=4,
        lw=2,
        label="Fiber model (NR)",
    )
    ax.axhline(1.0, color="gray", ls="-.", lw=1, alpha=0.5, label="M_y (elastic limit)")
    ax.axhline(1.5, color="gray", ls=":", lw=1, alpha=0.5, label="M_p = 1.5*M_y")
    ax.set_xlabel("Tip rotation [mrad]")
    ax.set_ylabel("M / M_y")
    ax.set_title(
        f"Fiber Model: Cantilever Tip Moment ({n_elems} elements, sigma_y={sigma_y}, H={H_iso})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fiber_cantilever_moment.png")
    plt.close(fig)
    print("  -> fiber_cantilever_moment.png")


# =====================================================================
# Phase C: 梁–梁接触 — 交差梁の接触力・ギャップ収束・ペナルティ依存
# =====================================================================

_CONTACT_NDOF_PER_NODE = 6


def _make_contact_crossing_beams(
    k_spring=1e4,
    z_top=0.082,
    radii=0.04,
):
    """交差梁ばねモデル（検証図用）."""
    n_nodes = 4
    ndof_total = n_nodes * _CONTACT_NDOF_PER_NODE

    node_coords_ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -0.5, z_top],
            [0.5, 0.5, z_top],
        ]
    )
    connectivity = np.array([[0, 1], [2, 3]])

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for n0, n1 in connectivity:
            for d in range(3):
                d0 = n0 * _CONTACT_NDOF_PER_NODE + d
                d1 = n1 * _CONTACT_NDOF_PER_NODE + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for n0, n1 in connectivity:
            for d in range(3):
                d0 = n0 * _CONTACT_NDOF_PER_NODE + d
                d1 = n1 * _CONTACT_NDOF_PER_NODE + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_spring * delta
                f_int[d1] += k_spring * delta
        return f_int

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    )


def _contact_fixed_dofs():
    """交差梁の拘束DOF."""
    fixed = []
    for d in range(_CONTACT_NDOF_PER_NODE):
        fixed.append(d)
    for d in range(_CONTACT_NDOF_PER_NODE):
        if d != 0:
            fixed.append(1 * _CONTACT_NDOF_PER_NODE + d)
    for d in range(_CONTACT_NDOF_PER_NODE):
        fixed.append(2 * _CONTACT_NDOF_PER_NODE + d)
    for d in range(_CONTACT_NDOF_PER_NODE):
        if d not in (1, 2):
            fixed.append(3 * _CONTACT_NDOF_PER_NODE + d)
    return np.array(fixed, dtype=int)


def plot_contact_crossing_beam():
    """交差梁接触の荷重ステップ応答（ギャップ・接触力・変位）."""
    plt = _setup_matplotlib()
    from xkep_cae.contact.solver_hooks import newton_raphson_with_contact

    from xkep_cae.contact.pair import ContactConfig, ContactManager

    (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_contact_crossing_beams()

    f_ext = np.zeros(ndof_total)
    f_ext[1 * _CONTACT_NDOF_PER_NODE + 0] = 10.0
    f_ext[3 * _CONTACT_NDOF_PER_NODE + 1] = 5.0
    f_ext[3 * _CONTACT_NDOF_PER_NODE + 2] = -50.0

    fixed_dofs = _contact_fixed_dofs()
    n_steps = 20

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=1e5,
            k_t_ratio=0.1,
            mu=0.3,
            g_on=0.0,
            g_off=1e-4,
            n_outer_max=8,
            use_friction=False,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.01,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
        ),
    )

    result = newton_raphson_with_contact(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        node_coords_ref,
        connectivity,
        radii,
        n_load_steps=n_steps,
        max_iter=50,
        show_progress=False,
        broadphase_margin=0.05,
    )

    # 荷重ステップ
    steps = np.arange(1, len(result.load_history) + 1)
    loads = np.array(result.load_history)

    # ギャップ履歴の取得（最終ステップのペアから）
    final_gaps = []
    final_pns = []
    for pair in mgr.pairs:
        if pair.is_active():
            final_gaps.append(pair.state.gap)
            final_pns.append(pair.state.p_n)

    # 接触力履歴
    cf_hist = np.array(result.contact_force_history)

    # z変位履歴（node3）
    z_dof = 3 * _CONTACT_NDOF_PER_NODE + 2
    z_hist = [uh[z_dof] for uh in result.displacement_history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) z変位 vs 荷重ステップ
    ax = axes[0]
    ax.plot(
        steps,
        [z_hist[i] * 1000 for i in range(len(steps))],
        "b-o",
        ms=3,
        lw=1.5,
        label="Node 3 (beam B)",
    )
    ax.set_xlabel("Load step")
    ax.set_ylabel("z-displacement [mm]")
    ax.set_title("(a) Displacement history")
    ax.legend(fontsize=8)

    # (b) 接触力 vs 荷重ステップ
    ax = axes[1]
    ax.plot(steps, cf_hist[: len(steps)], "r-s", ms=3, lw=1.5)
    ax.set_xlabel("Load step")
    ax.set_ylabel("Contact force norm [N]")
    ax.set_title("(b) Contact force history")

    # (c) 荷重係数 vs ステップ
    ax = axes[2]
    ax.plot(steps, loads[: len(steps)], "g-^", ms=3, lw=1.5)
    ax.set_xlabel("Load step")
    ax.set_ylabel("Load factor")
    ax.set_title("(c) Load factor history")

    fig.suptitle(
        "Crossing Beam Contact (spring model, k_pen=1e5, f_z=50N)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "contact_crossing_beam.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> contact_crossing_beam.png")


def plot_contact_penetration_control():
    """ペナルティ剛性と貫入量の関係（適応的ペナルティ増大の効果）."""
    plt = _setup_matplotlib()
    from xkep_cae.contact.solver_hooks import newton_raphson_with_contact

    from xkep_cae.contact.pair import ContactConfig, ContactManager

    k_pen_values = [1e3, 1e4, 1e5, 1e6]
    pen_ratios_noadapt = []
    pen_ratios_adapt = []

    radii = 0.04
    search_radius = 2 * radii  # r_a + r_b

    for k_pen in k_pen_values:
        for adaptive, results_list in [
            (False, pen_ratios_noadapt),
            (True, pen_ratios_adapt),
        ]:
            (
                node_coords_ref,
                connectivity,
                radii_val,
                ndof_total,
                assemble_tangent,
                assemble_internal_force,
            ) = _make_contact_crossing_beams()

            f_ext = np.zeros(ndof_total)
            f_ext[1 * _CONTACT_NDOF_PER_NODE + 0] = 10.0
            f_ext[3 * _CONTACT_NDOF_PER_NODE + 1] = 5.0
            f_ext[3 * _CONTACT_NDOF_PER_NODE + 2] = -50.0

            fixed_dofs = _contact_fixed_dofs()

            mgr = ContactManager(
                config=ContactConfig(
                    k_pen_scale=k_pen,
                    k_t_ratio=0.1,
                    mu=0.3,
                    g_on=0.0,
                    g_off=1e-4,
                    n_outer_max=8,
                    use_friction=False,
                    use_line_search=True,
                    use_geometric_stiffness=True,
                    tol_penetration_ratio=0.01 if adaptive else 0.0,
                    penalty_growth_factor=2.0,
                    k_pen_max=1e12,
                ),
            )

            newton_raphson_with_contact(
                f_ext,
                fixed_dofs,
                assemble_tangent,
                assemble_internal_force,
                mgr,
                node_coords_ref,
                connectivity,
                radii_val,
                n_load_steps=20,
                max_iter=50,
                show_progress=False,
                broadphase_margin=0.05,
            )

            # 最終ギャップの最大貫入比
            max_pen = 0.0
            for pair in mgr.pairs:
                if pair.is_active() and pair.state.gap < 0:
                    pen = abs(pair.state.gap) / search_radius
                    max_pen = max(max_pen, pen)
            results_list.append(max_pen * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(k_pen_values))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        pen_ratios_noadapt,
        width,
        label="Without adaptive penalty",
        color="salmon",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        pen_ratios_adapt,
        width,
        label="With adaptive penalty",
        color="steelblue",
        edgecolor="black",
    )

    ax.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.7, label="1% target")
    ax.set_xlabel("Initial penalty stiffness k_pen")
    ax.set_ylabel("Max penetration / search_radius [%]")
    ax.set_title(
        "Penetration Control: Adaptive Penalty Augmentation\n(crossing beams, f_z=50N, radii=40mm)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"$10^{{{int(np.log10(k))}}}$" for k in k_pen_values])
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # 数値ラベル
    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.1,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.1,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "contact_penetration_control.png")
    plt.close(fig)
    print("  -> contact_penetration_control.png")


def plot_contact_friction_stick_slip():
    """摩擦 return mapping: stick→slip 遷移の検証."""
    plt = _setup_matplotlib()
    from xkep_cae.contact.solver_hooks import newton_raphson_with_contact

    from xkep_cae.contact.pair import ContactConfig, ContactManager

    # 接線荷重を変化させて stick→slip 遷移を観察
    f_tangential_values = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    mu = 0.3

    friction_forces = []
    normal_forces = []
    tangential_disps = []

    for f_t in f_tangential_values:
        (
            node_coords_ref,
            connectivity,
            radii_val,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_contact_crossing_beams()

        f_ext = np.zeros(ndof_total)
        f_ext[1 * _CONTACT_NDOF_PER_NODE + 0] = f_t  # 接線方向荷重
        f_ext[3 * _CONTACT_NDOF_PER_NODE + 1] = 5.0
        f_ext[3 * _CONTACT_NDOF_PER_NODE + 2] = -50.0  # 法線押し込み

        fixed_dofs = _contact_fixed_dofs()

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                k_t_ratio=0.5,
                mu=mu,
                g_on=0.0,
                g_off=1e-4,
                n_outer_max=8,
                use_friction=True,
                mu_ramp_steps=3,
                use_line_search=True,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.01,
                penalty_growth_factor=2.0,
                k_pen_max=1e12,
            ),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords_ref,
            connectivity,
            radii_val,
            n_load_steps=20,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        max_q = 0.0
        max_pn = 0.0
        for pair in mgr.pairs:
            if pair.is_active():
                q_norm = float(np.linalg.norm(pair.state.z_t))
                max_q = max(max_q, q_norm)
                max_pn = max(max_pn, pair.state.p_n)

        friction_forces.append(max_q)
        normal_forces.append(max_pn)
        # x変位（node1）
        x_dof = 1 * _CONTACT_NDOF_PER_NODE + 0
        tangential_disps.append(result.u[x_dof] * 1000)  # mm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) 摩擦力 vs 接線荷重
    ax = axes[0]
    ax.plot(
        f_tangential_values, friction_forces, "ro-", ms=5, lw=1.5, label=r"$|q_t|$ (friction force)"
    )
    # Coulomb limit
    mu_pn = [mu * pn for pn in normal_forces]
    ax.plot(
        f_tangential_values,
        mu_pn,
        "b--s",
        ms=4,
        lw=1,
        alpha=0.7,
        label=r"$\mu \cdot p_n$ (Coulomb limit)",
    )
    ax.set_xlabel("Tangential load [N]")
    ax.set_ylabel("Force [N]")
    ax.set_title(r"(a) Friction force vs Coulomb limit ($\mu$=0.3)")
    ax.legend(fontsize=9)

    # (b) 接線変位 vs 接線荷重
    ax = axes[1]
    ax.plot(f_tangential_values, tangential_disps, "g^-", ms=5, lw=1.5)
    ax.set_xlabel("Tangential load [N]")
    ax.set_ylabel("Tangential displacement [mm]")
    ax.set_title("(b) Tangential displacement response")

    fig.suptitle(
        r"Friction Stick-Slip Transition (crossing beams, $\mu$=0.3, f_z=50N)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "contact_friction_stick_slip.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> contact_friction_stick_slip.png")


# =====================================================================
# Phase S3: チューニングタスク検証プロット
# =====================================================================


def _run_tuning_scaling():
    """スケーリング分析データを生成（7本+19本）."""
    from xkep_cae.tuning.executor import run_scaling_analysis

    return run_scaling_analysis(
        strand_counts=[7, 19],
        auto_kpen=True,
        lambda_n_max_factor=0.1,
        al_relaxation=0.01,
        k_pen_scaling="sqrt",
        staged_activation=True,
        g_on=0.0005,
        g_off=0.001,
        preserve_inactive_lambda=True,
        no_deactivation_within_step=True,
        penalty_growth_factor=1.0,
        gap=0.0005,
        use_block_solver=True,
        adaptive_omega=True,
        omega_min=0.01,
        omega_max=0.3,
        omega_growth=2.0,
    )


def plot_tuning_scaling_analysis(tuning_result=None):
    """素線数スケーリング分析: DOF・計算時間・Newton反復数.

    素線数増加に対するスケーリング挙動を3つのサブプロットで表示。
    計算コストの支配項を特定するための基礎データ。
    """
    plt = _setup_matplotlib()

    if tuning_result is None:
        tuning_result = _run_tuning_scaling()

    runs = tuning_result.runs
    if not runs:
        print("  -> スキップ（実行データなし）")
        return tuning_result

    n_strands_list = [r.metadata["n_strands"] for r in runs]
    ndof_list = [r.metadata["ndof"] for r in runs]
    time_list = [r.metrics["total_time_s"] for r in runs]
    newton_list = [r.metrics["total_newton_iterations"] for r in runs]
    converged_list = [r.metrics["converged"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("S3 チューニングタスク: スケーリング分析", fontsize=13)

    # (a) DOF vs 素線数
    ax = axes[0]
    ax.plot(n_strands_list, ndof_list, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("素線数")
    ax.set_ylabel("自由度数 (DOF)")
    ax.set_title("(a) DOF スケーリング")
    for ns, nd in zip(n_strands_list, ndof_list, strict=True):
        ax.annotate(
            f"{nd}", (ns, nd), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8
        )

    # (b) 計算時間 vs 素線数
    ax = axes[1]
    colors = ["green" if c else "red" for c in converged_list]
    ax.bar(range(len(n_strands_list)), time_list, color=colors, alpha=0.7)
    ax.set_xticks(range(len(n_strands_list)))
    ax.set_xticklabels([str(n) for n in n_strands_list])
    ax.set_xlabel("素線数")
    ax.set_ylabel("計算時間 (s)")
    ax.set_title("(b) 計算時間 (緑=収束 / 赤=非収束)")

    # (c) Newton反復数 vs 素線数
    ax = axes[2]
    ax.plot(n_strands_list, newton_list, "s-", color="darkorange", linewidth=2)
    ax.set_xlabel("素線数")
    ax.set_ylabel("合計Newton反復数")
    ax.set_title("(c) Newton反復数")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_scaling_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> tuning_scaling_analysis.png")
    return tuning_result


def plot_tuning_contact_topology(tuning_result=None):
    """接触トポロジー進化: 活性ペア数・接触力・stick/slip比の時間推移.

    荷重ステップに沿った接触状態の変遷を可視化。
    接触の活性化パターンと安定性を評価する。
    """
    plt = _setup_matplotlib()

    if tuning_result is None:
        tuning_result = _run_tuning_scaling()

    runs = tuning_result.runs
    if not runs:
        print("  -> スキップ（実行データなし）")
        return tuning_result

    n_plots = len(runs)
    fig, axes = plt.subplots(n_plots, 3, figsize=(16, 5 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("S3 チューニングタスク: 接触トポロジー進化", fontsize=13, y=1.02)

    for row, run in enumerate(runs):
        ns = run.metadata["n_strands"]
        ts = run.time_series

        # (a) 活性ペア数
        ax = axes[row, 0]
        if "active_pairs" in ts:
            steps = list(range(len(ts["active_pairs"])))
            ax.step(steps, ts["active_pairs"], where="mid", color="steelblue", linewidth=2)
            ax.fill_between(steps, ts["active_pairs"], alpha=0.2, step="mid", color="steelblue")
        ax.set_xlabel("荷重ステップ")
        ax.set_ylabel("活性接触ペア数")
        ax.set_title(f"({chr(97 + row * 3)}) {ns}本: 活性ペア")

        # (b) 接触力
        ax = axes[row, 1]
        if "contact_force" in ts:
            steps = list(range(len(ts["contact_force"])))
            ax.plot(steps, ts["contact_force"], "o-", color="crimson", linewidth=1.5, markersize=4)
        if "total_normal_force" in ts:
            steps2 = list(range(len(ts["total_normal_force"])))
            ax.plot(
                steps2,
                ts["total_normal_force"],
                "^--",
                color="darkorange",
                linewidth=1.5,
                markersize=4,
                label="法線力合計",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("荷重ステップ")
        ax.set_ylabel("接触力")
        ax.set_title(f"({chr(98 + row * 3)}) {ns}本: 接触力推移")

        # (c) 荷重係数
        ax = axes[row, 2]
        if "load_factor" in ts:
            steps = list(range(len(ts["load_factor"])))
            ax.plot(steps, ts["load_factor"], "D-", color="forestgreen", linewidth=2, markersize=4)
        ax.set_xlabel("荷重ステップ")
        ax.set_ylabel("荷重係数")
        ax.set_title(f"({chr(99 + row * 3)}) {ns}本: 荷重係数")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_contact_topology.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> tuning_contact_topology.png")
    return tuning_result


def plot_tuning_timing_breakdown(tuning_result=None):
    """工程別タイミング内訳: 各素線数の処理時間をスタックバーで表示.

    ボトルネック工程の特定とスケーリング効率の分析に使用。
    """
    plt = _setup_matplotlib()

    if tuning_result is None:
        tuning_result = _run_tuning_scaling()

    runs = tuning_result.runs
    if not runs:
        print("  -> スキップ（実行データなし）")
        return tuning_result

    # 工程名とカラーマップ
    phase_keys = [
        "time_broadphase",
        "time_geometry_update",
        "time_contact_force",
        "time_contact_stiffness",
        "time_structural_internal_force",
        "time_structural_tangent",
        "time_bc_apply",
        "time_linear_solve",
        "time_line_search",
        "time_outer_convergence_check",
    ]
    phase_labels = [
        "Broadphase",
        "Geometry",
        "Contact F",
        "Contact K",
        "Structural F",
        "Structural K",
        "BC Apply",
        "Linear Solve",
        "Line Search",
        "Outer Check",
    ]
    colors = plt.cm.tab20(np.linspace(0, 1, len(phase_keys)))

    n_strands_list = [r.metadata["n_strands"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("S3 チューニングタスク: 工程別タイミング内訳", fontsize=13)

    # (a) 絶対時間スタックバー
    ax = axes[0]
    x = np.arange(len(runs))
    bottoms = np.zeros(len(runs))
    for pi, (pk, pl) in enumerate(zip(phase_keys, phase_labels, strict=True)):
        vals = [r.metrics.get(pk, 0.0) for r in runs]
        ax.bar(x, vals, bottom=bottoms, color=colors[pi], label=pl, width=0.6)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}本" for n in n_strands_list])
    ax.set_xlabel("素線数")
    ax.set_ylabel("処理時間 (s)")
    ax.set_title("(a) 絶対時間")
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    # (b) 割合スタックバー
    ax = axes[1]
    bottoms = np.zeros(len(runs))
    for pi, (pk, pl) in enumerate(zip(phase_keys, phase_labels, strict=True)):
        vals = [r.metrics.get(pk, 0.0) for r in runs]
        totals = [r.metrics.get("total_time_s", 1.0) for r in runs]
        ratios = [v / t * 100 if t > 0 else 0 for v, t in zip(vals, totals, strict=True)]
        ax.bar(x, ratios, bottom=bottoms, color=colors[pi], label=pl, width=0.6)
        bottoms += np.array(ratios)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}本" for n in n_strands_list])
    ax.set_xlabel("素線数")
    ax.set_ylabel("割合 (%)")
    ax.set_title("(b) 工程比率")
    ax.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_timing_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> tuning_timing_breakdown.png")
    return tuning_result


def plot_tuning_wire_cross_section(tuning_result=None):
    """ワイヤ断面2D投影: 接触ペアを線で結んだ断面図.

    撚線の幾何学的配置と接触パターンを直感的に可視化。
    CAE後処理の「AI目視検査」の基礎。
    """
    plt = _setup_matplotlib()
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

    if tuning_result is None:
        tuning_result = _run_tuning_scaling()

    runs = tuning_result.runs
    if not runs:
        print("  -> スキップ（実行データなし）")
        return tuning_result

    n_plots = len(runs)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("S3 チューニングタスク: 断面接触マップ", fontsize=13)

    wire_d = 0.002
    pitch = 0.040

    for idx, run in enumerate(runs):
        ax = axes[idx]
        ns = run.metadata["n_strands"]

        mesh = make_twisted_wire_mesh(
            ns,
            wire_d,
            pitch,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )

        # 各素線の中点（z=pitch/2 付近）の断面位置
        strand_centers = []
        for sid in range(mesh.n_strands):
            nodes = mesh.strand_nodes(sid)
            mid_node = nodes[len(nodes) // 2]
            pos = mesh.node_coords[mid_node]
            strand_centers.append(pos[:2])  # x, y
        strand_centers = np.array(strand_centers)

        # 素線円を描画
        layer_map = {}
        for info in mesh.strand_infos:
            layer_map[info.strand_id] = info.layer

        layer_colors = plt.cm.Set2(np.linspace(0, 0.8, max(layer_map.values()) + 1))
        for sid, (cx, cy) in enumerate(strand_centers):
            layer = layer_map.get(sid, 0)
            circle = plt.Circle(
                (cx, cy), wire_d / 2, fill=False, color=layer_colors[layer], linewidth=1.5
            )
            ax.add_patch(circle)
            ax.annotate(str(sid), (cx, cy), ha="center", va="center", fontsize=6)

        # 接触ペアを線で表示（素線間を赤線で接続）
        n_active = run.metrics.get("n_active_pairs", 0)
        if n_active > 0:
            # 素線間接触: 隣接層間の素線を接続
            for si in range(ns):
                for sj in range(si + 1, ns):
                    li = layer_map.get(si, 0)
                    lj = layer_map.get(sj, 0)
                    if abs(li - lj) <= 1 and li != lj:
                        # 隣接層間ペア → 接触線を描画
                        ci = strand_centers[si]
                        cj = strand_centers[sj]
                        dist = np.linalg.norm(ci - cj)
                        if dist < wire_d * 2.5:  # 近接ペアのみ
                            ax.plot([ci[0], cj[0]], [ci[1], cj[1]], "r-", alpha=0.4, linewidth=0.8)

        ax.set_aspect("equal")
        margin = wire_d * 2
        if len(strand_centers) > 0:
            ax.set_xlim(strand_centers[:, 0].min() - margin, strand_centers[:, 0].max() + margin)
            ax.set_ylim(strand_centers[:, 1].min() - margin, strand_centers[:, 1].max() + margin)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        conv_str = "収束" if run.metrics["converged"] else "非収束"
        ax.set_title(
            f"{ns}本撚り ({conv_str})\n"
            f"活性ペア={n_active}, 貫入比={run.metrics['max_penetration_ratio']:.3f}",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_wire_cross_section.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> tuning_wire_cross_section.png")
    return tuning_result


def plot_tuning_acceptance_summary(tuning_result=None):
    """合格判定サマリー: 各基準の達成状況をヒートマップで表示.

    TuningTask の AcceptanceCriterion に対する各実行の合否を
    一覧表示し、パラメータチューニングの進捗を俯瞰する。
    """
    plt = _setup_matplotlib()

    if tuning_result is None:
        tuning_result = _run_tuning_scaling()

    runs = tuning_result.runs
    task = tuning_result.task
    if not runs or not task.criteria:
        print("  -> スキップ（データ不足）")
        return tuning_result

    # 判定マトリクス: rows=runs, cols=criteria
    criteria_names = [c.metric for c in task.criteria]
    n_runs = len(runs)
    n_criteria = len(criteria_names)
    matrix = np.zeros((n_runs, n_criteria))

    for ri, run in enumerate(runs):
        verdicts = run.evaluate_criteria(task.criteria)
        for ci, cname in enumerate(criteria_names):
            matrix[ri, ci] = 1.0 if verdicts.get(cname, False) else 0.0

    fig, ax = plt.subplots(figsize=(max(8, n_criteria * 2), max(4, n_runs * 1.5)))
    fig.suptitle("S3 チューニングタスク: 合格判定サマリー", fontsize=13)

    # ヒートマップ
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#ff6b6b", "#51cf66"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # ラベル
    run_labels = [f"{r.metadata.get('n_strands', '?')}本" for r in runs]
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels(run_labels)
    ax.set_xticks(range(n_criteria))
    ax.set_xticklabels(criteria_names, rotation=45, ha="right")

    # セル内にPass/Fail表示
    for ri in range(n_runs):
        for ci in range(n_criteria):
            cname = criteria_names[ci]
            val = runs[ri].metrics.get(cname, "N/A")
            label = "Pass" if matrix[ri, ci] > 0.5 else "Fail"
            if isinstance(val, bool):
                val_str = str(val)
            elif isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            ax.text(
                ci,
                ri,
                f"{label}\n({val_str})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[ri, ci] < 0.5 else "black",
            )

    # 基準の詳細をフッターに表示
    footer_lines = []
    for c in task.criteria:
        footer_lines.append(f"  {c.metric}: {c.op} {c.target} — {c.description}")
    fig.text(
        0.02,
        -0.02,
        "\n".join(footer_lines),
        fontsize=7,
        verticalalignment="top",
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_acceptance_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> tuning_acceptance_summary.png")

    # JSON保存
    json_path = OUTPUT_DIR / "tuning_result.json"
    tuning_result.save_json(json_path)
    print(f"  -> {json_path.name}")
    return tuning_result


def plot_tuning_sensitivity_heatmap(tuning_result=None):
    """パラメータ感度分析: omega_max × al_relaxation のヒートマップ.

    2パラメータの組合せに対する収束性・Newton反復数・貫入比を
    ヒートマップで可視化し、パラメータ感度を直感的に把握する。
    """
    plt = _setup_matplotlib()
    from xkep_cae.tuning.executor import run_sensitivity_analysis

    if tuning_result is None:
        tuning_result = run_sensitivity_analysis(
            n_strands=7,
            param1_name="omega_max",
            param1_values=[0.1, 0.3, 0.5],
            param2_name="al_relaxation",
            param2_values=[0.005, 0.01, 0.05],
            auto_kpen=True,
            lambda_n_max_factor=0.1,
            k_pen_scaling="sqrt",
            staged_activation=True,
            g_on=0.0005,
            g_off=0.001,
            preserve_inactive_lambda=True,
            no_deactivation_within_step=True,
            penalty_growth_factor=1.0,
            gap=0.0005,
            use_block_solver=True,
            adaptive_omega=True,
            omega_min=0.01,
            omega_growth=2.0,
        )

    runs = tuning_result.runs
    if not runs:
        print("  -> スキップ（実行データなし）")
        return tuning_result

    # パラメータ値の抽出
    p1_vals = sorted({r.params.get("omega_max", 0.3) for r in runs})
    p2_vals = sorted({r.params.get("al_relaxation", 0.01) for r in runs})
    n1, n2 = len(p1_vals), len(p2_vals)

    if n1 < 2 or n2 < 2:
        print("  -> スキップ（グリッドが小さすぎる）")
        return tuning_result

    # ルックアップ用辞書
    run_map = {}
    for r in runs:
        key = (r.params.get("omega_max", 0.3), r.params.get("al_relaxation", 0.01))
        run_map[key] = r

    # メトリクスの3種ヒートマップ
    metrics_info = [
        ("converged", "Convergence", "RdYlGn", False),
        ("total_newton_iterations", "Newton Iterations", "YlOrRd", True),
        ("max_penetration_ratio", "Max Penetration Ratio", "YlOrRd", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Parameter Sensitivity: omega_max x al_relaxation (7-strand)",
        fontsize=13,
    )

    for ax, (metric, title, cmap_name, _) in zip(axes, metrics_info, strict=True):
        matrix = np.full((n2, n1), np.nan)
        for i1, v1 in enumerate(p1_vals):
            for i2, v2 in enumerate(p2_vals):
                r = run_map.get((v1, v2))
                if r is not None and metric in r.metrics:
                    val = r.metrics[metric]
                    matrix[i2, i1] = (
                        float(val) if not isinstance(val, bool) else (1.0 if val else 0.0)
                    )

        im = ax.imshow(matrix, cmap=cmap_name, aspect="auto", origin="lower")
        ax.set_xticks(range(n1))
        ax.set_xticklabels([f"{v:.2f}" for v in p1_vals])
        ax.set_yticks(range(n2))
        ax.set_yticklabels([f"{v:.3f}" for v in p2_vals])
        ax.set_xlabel("omega_max")
        ax.set_ylabel("al_relaxation")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

        # セル内に値を表示
        for i1 in range(n1):
            for i2 in range(n2):
                val = matrix[i2, i1]
                if np.isnan(val):
                    continue
                if metric == "converged":
                    txt = "Yes" if val > 0.5 else "No"
                elif metric == "max_penetration_ratio":
                    txt = f"{val:.3f}"
                else:
                    txt = f"{int(val)}"
                ax.text(i1, i2, txt, ha="center", va="center", fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_sensitivity_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> tuning_sensitivity_heatmap.png")
    return tuning_result


# =====================================================================
# Phase 3+5: 幾何学非線形 — CR大変形の応力・曲率コンター + 変形メッシュ
# =====================================================================


def plot_cr_stress_curvature_contour():
    """CR大変形カンチレバーの応力・曲率コンター＆変形メッシュの2D投影.

    3つのサブプロット:
      1. 変形メッシュ（初期形状との比較）
      2. 要素モーメント分布（コンター）
      3. 要素曲率分布（コンター）
    """
    plt = _setup_matplotlib()
    from xkep_cae.elements.beam_timo3d import (
        _beam3d_length_and_direction,
        _build_local_axes,
        _rotmat_to_rotvec,
        _rotvec_to_rotmat,
        assemble_cr_beam3d,
        timo_beam3d_ke_local,
    )
    from xkep_cae.sections.beam import BeamSection
    from xkep_cae.solver import newton_raphson

    # --- 問題設定 ---
    n_elems = 30
    L = 1.0
    E = 2.1e11
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 0.02
    sec = BeamSection.circle(d)
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    nodes = np.zeros((n_elems + 1, 3))
    nodes[:, 0] = np.linspace(0, L, n_elems + 1)
    conn = np.array([[i, i + 1] for i in range(n_elems)])
    n_nodes = n_elems + 1
    ndof = 6 * n_nodes
    fixed_dofs = np.arange(6)

    # --- 3段階の荷重レベル ---
    load_levels = [0.03, 0.10, 0.20]  # δ/L targets
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, delta_ratio in enumerate(load_levels):
        delta_target = delta_ratio * L
        P = 3.0 * E * sec.Iy * delta_target / L**3

        f_ext = np.zeros(ndof)
        f_ext[6 * n_elems + 1] = P

        def assemble_tangent(u, _E=E, _G=G):
            K, _ = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                _E,
                _G,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.J,
                kappa,
                kappa,
                stiffness=True,
                internal_force=False,
                sparse=True,
            )
            return K

        def assemble_fint(u, _E=E, _G=G):
            _, f = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                _E,
                _G,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.J,
                kappa,
                kappa,
                stiffness=False,
                internal_force=True,
                sparse=False,
            )
            return f

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            assemble_tangent,
            assemble_fint,
            n_load_steps=max(5, int(delta_ratio * 50)),
            tol_force=1e-5,
            show_progress=False,
        )
        if not result.converged:
            print(f"  WARNING: δ/L={delta_ratio} not converged, skipping")
            continue

        u = result.u

        # 変形節点座標
        deformed = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            deformed[i] = nodes[i] + u[6 * i : 6 * i + 3]

        # CR断面力抽出
        moments = np.zeros(n_elems)
        curvatures = np.zeros(n_elems)
        elem_centers = np.zeros(n_elems)
        for ei, (n1, n2) in enumerate(conn):
            coords = nodes[np.array([n1, n2])]
            edofs = np.array([6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)])
            u_elem = u[edofs]

            L_0, e_x_0 = _beam3d_length_and_direction(coords)
            R_0 = _build_local_axes(e_x_0, None)
            v_ref_stable = R_0[1, :]
            x1_def = coords[0] + u_elem[0:3]
            x2_def = coords[1] + u_elem[6:9]
            coords_def = np.array([x1_def, x2_def])
            L_def, e_x_def = _beam3d_length_and_direction(coords_def)
            R_cr = _build_local_axes(e_x_def, v_ref_stable)
            R_node1 = _rotvec_to_rotmat(u_elem[3:6])
            R_node2 = _rotvec_to_rotmat(u_elem[9:12])
            R_def1 = R_cr @ R_node1 @ R_0.T
            R_def2 = R_cr @ R_node2 @ R_0.T
            theta_def1 = _rotmat_to_rotvec(R_def1)
            theta_def2 = _rotmat_to_rotvec(R_def2)
            d_cr = np.zeros(12)
            d_cr[3:6] = theta_def1
            d_cr[6] = L_def - L_0
            d_cr[9:12] = theta_def2
            Ke_local = timo_beam3d_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L_0, kappa, kappa)
            f_cr = Ke_local @ d_cr

            moments[ei] = abs(f_cr[11])  # |Mz| at node 2
            curvatures[ei] = abs(f_cr[11]) / (E * sec.Iz) if E * sec.Iz > 0 else 0
            elem_centers[ei] = (deformed[n1, 0] + deformed[n2, 0]) / 2.0

        # --- サブプロット1: 変形メッシュ ---
        ax = axes[0, 0]
        ax.plot(nodes[:, 0], nodes[:, 1], "b--", linewidth=1, alpha=0.5, label="Initial")
        colors_load = ["#2196F3", "#FF9800", "#F44336"]
        ax.plot(
            deformed[:, 0],
            deformed[:, 1],
            "-o",
            color=colors_load[idx],
            markersize=2,
            label=f"δ/L={delta_ratio:.0%}",
        )

        # --- サブプロット2: モーメントコンター ---
        ax2 = axes[0, 1]
        ax2.plot(
            elem_centers,
            moments,
            "-o",
            color=colors_load[idx],
            markersize=3,
            label=f"δ/L={delta_ratio:.0%}",
        )

        # --- サブプロット3: 曲率コンター ---
        ax3 = axes[1, 0]
        ax3.plot(
            elem_centers,
            curvatures,
            "-o",
            color=colors_load[idx],
            markersize=3,
            label=f"δ/L={delta_ratio:.0%}",
        )

    # 解析解（線形）
    ax_ana = axes[1, 1]
    x_ana = np.linspace(0, L, 200)
    for idx, delta_ratio in enumerate(load_levels):
        P = 3.0 * E * sec.Iy * delta_ratio * L / L**3
        M_ana = P * (L - x_ana)
        kappa_ana = M_ana / (E * sec.Iz)
        ax_ana.plot(
            x_ana,
            kappa_ana,
            "--",
            color=colors_load[idx],
            linewidth=1.5,
            label=f"Linear δ/L={delta_ratio:.0%}",
        )
    ax_ana.set_xlabel("x [m]")
    ax_ana.set_ylabel("κ [1/m] (analytical)")
    ax_ana.set_title("Linear Analytical Curvature")
    ax_ana.legend(fontsize=8)

    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("Deformed Shape (CR Nonlinear)")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("|Mz| [N·m]")
    axes[0, 1].set_title("Bending Moment Distribution (CR)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel("x [m]")
    axes[1, 0].set_ylabel("κ [1/m]")
    axes[1, 0].set_title("Curvature Distribution (CR)")
    axes[1, 0].legend(fontsize=8)

    fig.suptitle(
        f"CR Nonlinear Cantilever: L={L}m, d={d * 1000:.0f}mm, E={E / 1e9:.0f}GPa",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cr_stress_curvature_contour.png")
    plt.close(fig)
    print("  -> cr_stress_curvature_contour.png")


# =====================================================================
# Phase 5: 動的解析 — エネルギー時刻歴 + 変位応答
# =====================================================================


def plot_dynamics_energy_history():
    """動的解析のエネルギー保存検証プロット.

    3つのサブプロット:
      1. 線形梁の自由振動エネルギー（K, U, Total）
      2. CR非線形梁のエネルギー
      3. HHT-αのエネルギー散逸
    """
    plt = _setup_matplotlib()
    from scipy.linalg import eigh as sp_eigh

    from xkep_cae.dynamics import NonlinearTransientConfig, solve_nonlinear_transient
    from xkep_cae.elements.beam_timo3d import (
        assemble_cr_beam3d,
        timo_beam3d_ke_global,
    )

    # --- 梁の構築 ---
    n_elems, L, E, nu, rho, r = 10, 0.5, 2.1e11, 0.3, 7800.0, 0.005
    G = E / (2.0 * (1.0 + nu))
    A = np.pi * r**2
    Iy = np.pi * r**4 / 4.0
    Iz = Iy
    J = 2.0 * Iy
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    nodes_arr = np.zeros((n_elems + 1, 3))
    nodes_arr[:, 0] = np.linspace(0, L, n_elems + 1)
    conn_arr = np.array([[i, i + 1] for i in range(n_elems)])
    n_nodes = n_elems + 1
    ndof = 6 * n_nodes

    K = np.zeros((ndof, ndof))
    for e in range(n_elems):
        n1, n2 = conn_arr[e]
        coords = nodes_arr[np.array([n1, n2])]
        ke = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, kappa, kappa)
        edofs = np.array([6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)])
        K[np.ix_(edofs, edofs)] += ke

    Le = L / n_elems
    M = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        m_i = rho * A * Le * (0.5 if (i == 0 or i == n_elems) else 1.0)
        for dd in range(3):
            M[6 * i + dd, 6 * i + dd] = m_i
        I_rot = m_i * r**2 / 2.0
        for dd in range(3, 6):
            M[6 * i + dd, 6 * i + dd] = I_rot

    fixed = np.arange(6)
    free = np.setdiff1d(np.arange(ndof), fixed)
    tip_dof = 6 * n_elems + 1

    # 固有周期
    eigvals, _ = sp_eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    omega1 = np.sqrt(max(eigvals[0], 0.0))
    T1 = 2.0 * np.pi / omega1

    # 初期変位
    F0 = np.zeros(ndof)
    F0[tip_dof] = -0.5
    u0 = np.zeros(ndof)
    u0[free] = np.linalg.solve(K[np.ix_(free, free)], F0[free])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. 線形自由振動 ---
    def f_int_lin(u):
        return K @ u

    def K_T_lin(u):
        return K

    dt = T1 / 40.0
    n_steps = int(5 * T1 / dt)
    cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
    res = solve_nonlinear_transient(
        M,
        np.zeros(ndof),
        u0,
        np.zeros(ndof),
        cfg,
        f_int_lin,
        K_T_lin,
        fixed_dofs=fixed,
        show_progress=False,
    )

    kinetic = np.array([0.5 * res.velocity[i] @ M @ res.velocity[i] for i in range(n_steps + 1)])
    strain = np.array(
        [0.5 * res.displacement[i] @ K @ res.displacement[i] for i in range(n_steps + 1)]
    )
    total = kinetic + strain
    t = res.time / T1

    ax = axes[0]
    ax.plot(t, kinetic, "r-", alpha=0.7, label="Kinetic")
    ax.plot(t, strain, "b-", alpha=0.7, label="Strain")
    ax.plot(t, total, "k-", linewidth=2, label="Total")
    ax.set_xlabel("t / T₁")
    ax.set_ylabel("Energy [J]")
    ax.set_title(f"Linear Free Vibration\nΔE/E₀ = {abs(total[-1] - total[0]) / total[0]:.2e}")
    ax.legend(fontsize=8)

    # --- 2. CR非線形自由振動 ---
    def f_int_cr(u):
        _, f = assemble_cr_beam3d(
            nodes_arr,
            conn_arr,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa,
            kappa,
            stiffness=False,
            internal_force=True,
            sparse=False,
        )
        return f

    def K_T_cr(u):
        Kt, _ = assemble_cr_beam3d(
            nodes_arr,
            conn_arr,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa,
            kappa,
            stiffness=True,
            internal_force=False,
            sparse=False,
        )
        return Kt

    n_steps_cr = int(3 * T1 / dt)
    cfg_cr = NonlinearTransientConfig(dt=dt, n_steps=n_steps_cr, tol_force=1e-6)
    res_cr = solve_nonlinear_transient(
        M,
        np.zeros(ndof),
        u0,
        np.zeros(ndof),
        cfg_cr,
        f_int_cr,
        K_T_cr,
        fixed_dofs=fixed,
        show_progress=False,
    )

    if res_cr.converged:
        kinetic_cr = np.array(
            [0.5 * res_cr.velocity[i] @ M @ res_cr.velocity[i] for i in range(n_steps_cr + 1)]
        )
        strain_cr = np.array(
            [
                0.5 * res_cr.displacement[i] @ f_int_cr(res_cr.displacement[i])
                for i in range(n_steps_cr + 1)
            ]
        )
        total_cr = kinetic_cr + strain_cr
        t_cr = res_cr.time / T1

        ax2 = axes[1]
        ax2.plot(t_cr, kinetic_cr, "r-", alpha=0.7, label="Kinetic")
        ax2.plot(t_cr, strain_cr, "b-", alpha=0.7, label="Strain")
        ax2.plot(t_cr, total_cr, "k-", linewidth=2, label="Total")
        ax2.set_xlabel("t / T₁")
        ax2.set_ylabel("Energy [J]")
        err_cr = np.max(np.abs(total_cr - total_cr[0])) / total_cr[0] if total_cr[0] > 0 else 0
        ax2.set_title(f"CR Nonlinear Free Vibration\nmax|ΔE/E₀| = {err_cr:.2e}")
        ax2.legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "CR not converged", transform=axes[1].transAxes, ha="center")

    # --- 3. HHT-α散逸 ---
    alpha_hht = -0.1
    gamma_hht = 0.5 * (1.0 - 2.0 * alpha_hht)
    beta_hht = 0.25 * (1.0 - alpha_hht) ** 2
    n_steps_hht = int(10 * T1 / dt)
    cfg_hht = NonlinearTransientConfig(
        dt=dt,
        n_steps=n_steps_hht,
        tol_force=1e-12,
        alpha_hht=alpha_hht,
        gamma=gamma_hht,
        beta=beta_hht,
    )
    res_hht = solve_nonlinear_transient(
        M,
        np.zeros(ndof),
        u0,
        np.zeros(ndof),
        cfg_hht,
        f_int_lin,
        K_T_lin,
        fixed_dofs=fixed,
        show_progress=False,
    )

    kinetic_h = np.array(
        [0.5 * res_hht.velocity[i] @ M @ res_hht.velocity[i] for i in range(n_steps_hht + 1)]
    )
    strain_h = np.array(
        [
            0.5 * res_hht.displacement[i] @ K @ res_hht.displacement[i]
            for i in range(n_steps_hht + 1)
        ]
    )
    total_h = kinetic_h + strain_h
    t_h = res_hht.time / T1

    ax3 = axes[2]
    ax3.plot(t_h, total_h / total_h[0], "k-", linewidth=2, label="Total / E₀")
    ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("t / T₁")
    ax3.set_ylabel("E(t) / E(0)")
    dissipation = (total_h[0] - total_h[-1]) / total_h[0]
    ax3.set_title(f"HHT-α (α={alpha_hht}) Dissipation\nΔE/E₀ = {dissipation:.2%}")
    ax3.legend(fontsize=8)

    fig.suptitle("Dynamic Analysis Energy Conservation Verification", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dynamics_energy_history.png")
    plt.close(fig)
    print("  -> dynamics_energy_history.png")


# =====================================================================
# Phase 3+5: 動的解析 — 変位応答時刻歴
# =====================================================================


def plot_dynamics_displacement_response():
    """動的解析の変位応答を3パネルで可視化.

    1. 自由振動の先端変位時刻歴
    2. ランプ荷重に対する変位応答（動的 vs 静的）
    3. 減衰ありの変位減衰
    """
    plt = _setup_matplotlib()
    from scipy.linalg import eigh as sp_eigh

    from xkep_cae.dynamics import NonlinearTransientConfig, solve_nonlinear_transient
    from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global

    n_elems, L, E, nu, rho, r = 10, 0.5, 2.1e11, 0.3, 7800.0, 0.005
    G = E / (2.0 * (1.0 + nu))
    A = np.pi * r**2
    Iy = np.pi * r**4 / 4.0
    Iz = Iy
    J = 2.0 * Iy
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    nodes_arr = np.zeros((n_elems + 1, 3))
    nodes_arr[:, 0] = np.linspace(0, L, n_elems + 1)
    conn_arr = np.array([[i, i + 1] for i in range(n_elems)])
    n_nodes = n_elems + 1
    ndof = 6 * n_nodes

    K = np.zeros((ndof, ndof))
    for e in range(n_elems):
        n1, n2 = conn_arr[e]
        coords = nodes_arr[np.array([n1, n2])]
        ke = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, kappa, kappa)
        edofs = np.array([6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)])
        K[np.ix_(edofs, edofs)] += ke

    Le = L / n_elems
    M_mat = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        m_i = rho * A * Le * (0.5 if (i == 0 or i == n_elems) else 1.0)
        for dd in range(3):
            M_mat[6 * i + dd, 6 * i + dd] = m_i
        I_rot = m_i * r**2 / 2.0
        for dd in range(3, 6):
            M_mat[6 * i + dd, 6 * i + dd] = I_rot

    fixed = np.arange(6)
    free = np.setdiff1d(np.arange(ndof), fixed)
    tip_dof = 6 * n_elems + 1

    eigvals, _ = sp_eigh(K[np.ix_(free, free)], M_mat[np.ix_(free, free)])
    omega1 = np.sqrt(max(eigvals[0], 0.0))
    T1 = 2.0 * np.pi / omega1

    def f_int_fn(u):
        return K @ u

    def K_T_fn(u):
        return K

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. 自由振動 ---
    F0 = np.zeros(ndof)
    F0[tip_dof] = -1.0
    u0 = np.zeros(ndof)
    u0[free] = np.linalg.solve(K[np.ix_(free, free)], F0[free])
    delta_0 = u0[tip_dof]

    dt = T1 / 40.0
    n_steps = int(5 * T1 / dt)
    cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
    res = solve_nonlinear_transient(
        M_mat,
        np.zeros(ndof),
        u0,
        np.zeros(ndof),
        cfg,
        f_int_fn,
        K_T_fn,
        fixed_dofs=fixed,
        show_progress=False,
    )
    t = res.time / T1
    tip = res.displacement[:, tip_dof] / abs(delta_0)

    ax = axes[0]
    ax.plot(t, tip, "b-", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("t / T₁")
    ax.set_ylabel("δ_tip / δ₀")
    ax.set_title("Free Vibration (Undamped)")
    ax.set_ylim(-1.5, 1.5)

    # --- 2. ランプ荷重 ---
    P = 1.0
    f_static = np.zeros(ndof)
    f_static[tip_dof] = P
    u_static = np.zeros(ndof)
    u_static[free] = np.linalg.solve(K[np.ix_(free, free)], f_static[free])
    delta_static = u_static[tip_dof]

    ramp_time = 5.0 * T1

    def get_ramp(t):
        return f_static * min(t / ramp_time, 1.0)

    n_steps_r = int(10 * T1 / dt)
    cfg_r = NonlinearTransientConfig(dt=dt, n_steps=n_steps_r, tol_force=1e-12)
    res_r = solve_nonlinear_transient(
        M_mat,
        get_ramp,
        np.zeros(ndof),
        np.zeros(ndof),
        cfg_r,
        f_int_fn,
        K_T_fn,
        fixed_dofs=fixed,
        show_progress=False,
    )
    t_r = res_r.time / T1
    tip_r = res_r.displacement[:, tip_dof] / abs(delta_static)

    ax2 = axes[1]
    ax2.plot(t_r, tip_r, "b-", linewidth=1, label="Dynamic")
    ax2.plot(
        t_r, np.minimum(t_r * T1 / ramp_time, 1.0), "r--", linewidth=1.5, label="Static (ramp)"
    )
    ax2.set_xlabel("t / T₁")
    ax2.set_ylabel("δ_tip / δ_static")
    ax2.set_title("Ramp Loading Response")
    ax2.legend(fontsize=8)

    # --- 3. 減衰自由振動 ---
    zeta = 0.05
    alpha_R = 2.0 * zeta * omega1
    C = alpha_R * M_mat

    n_steps_d = int(15 * T1 / dt)
    cfg_d = NonlinearTransientConfig(dt=dt, n_steps=n_steps_d, tol_force=1e-12)
    res_d = solve_nonlinear_transient(
        M_mat,
        np.zeros(ndof),
        u0,
        np.zeros(ndof),
        cfg_d,
        f_int_fn,
        K_T_fn,
        C=C,
        fixed_dofs=fixed,
        show_progress=False,
    )
    t_d = res_d.time / T1
    tip_d = res_d.displacement[:, tip_dof] / abs(delta_0)

    # 理論的エンベロープ
    envelope = np.exp(-zeta * omega1 * res_d.time)

    ax3 = axes[2]
    ax3.plot(t_d, tip_d, "b-", linewidth=1, label="Response")
    ax3.plot(t_d, envelope, "r--", linewidth=1.5, label=f"Envelope (ζ={zeta})")
    ax3.plot(t_d, -envelope, "r--", linewidth=1.5)
    ax3.set_xlabel("t / T₁")
    ax3.set_ylabel("δ_tip / δ₀")
    ax3.set_title("Damped Free Vibration (ζ=5%)")
    ax3.legend(fontsize=8)

    fig.suptitle("Dynamic Analysis Verification: Cantilever Beam", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dynamics_displacement_response.png")
    plt.close(fig)
    print("  -> dynamics_displacement_response.png")


# =====================================================================
# 撚線3D断面可視化 + 被膜/シース + 断面応力コンター
# =====================================================================


def plot_twisted_wire_3d_cross_section():
    """撚線の3D断面可視化: ワイヤ断面 + 被膜 + シースの2D投影.

    7本撚りメッシュの断面を z=0 と z=L/2 で切断し、
    各素線の円形断面・被膜環・シース外筒を描画する。
    さらに側面図（xz平面）でヘリカル中心線を描画。
    """
    plt = _setup_matplotlib()
    from xkep_cae.mesh.twisted_wire import (
        CoatingModel,
        SheathModel,
        compute_envelope_radius,
        make_twisted_wire_mesh,
        sheath_inner_radius,
    )

    # 7本撚りメッシュ
    wire_d = 2.0e-3
    pitch = 40.0e-3
    length = pitch
    n_elems = 32
    mesh = make_twisted_wire_mesh(
        n_strands=7,
        wire_diameter=wire_d,
        pitch=pitch,
        length=length,
        n_elems_per_strand=n_elems,
    )
    r_wire = wire_d / 2.0

    # 被膜・シース
    coating = CoatingModel(thickness=0.08e-3, E=3.0e9, nu=0.35, mu=0.2)
    sheath = SheathModel(thickness=0.3e-3, E=70.0e9, nu=0.33, mu=0.15)
    r_coat = r_wire + coating.thickness
    r_env = compute_envelope_radius(mesh, coating=coating)
    r_sheath_in = sheath_inner_radius(mesh, sheath, coating=coating)
    r_sheath_out = r_sheath_in + sheath.thickness

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    theta_circ = np.linspace(0, 2 * np.pi, 64)

    # --- 断面図 (z=0 と z=L/2) ---
    for ax_idx, (z_cut, title) in enumerate(
        [(0, "断面 z=0 (端部)"), (length / 2, "断面 z=L/2 (中央)")]
    ):
        ax = axes[ax_idx]
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)

        # 各素線の中心位置を z_cut で補間
        for si in range(mesh.n_strands):
            ns, ne = mesh.strand_node_ranges[si]
            coords = mesh.node_coords[ns:ne]
            z_vals = coords[:, 2]
            # z_cut に最も近い節点を見つけて補間
            if z_cut <= z_vals[0]:
                cx, cy = coords[0, 0], coords[0, 1]
            elif z_cut >= z_vals[-1]:
                cx, cy = coords[-1, 0], coords[-1, 1]
            else:
                idx = np.searchsorted(z_vals, z_cut) - 1
                t_frac = (z_cut - z_vals[idx]) / (z_vals[idx + 1] - z_vals[idx])
                cx = coords[idx, 0] + t_frac * (coords[idx + 1, 0] - coords[idx, 0])
                cy = coords[idx, 1] + t_frac * (coords[idx + 1, 1] - coords[idx, 1])

            # 素線芯（塗りつぶし）
            ax.fill(
                cx + r_wire * np.cos(theta_circ),
                cy + r_wire * np.sin(theta_circ),
                color="steelblue",
                alpha=0.7,
                edgecolor="navy",
                linewidth=0.8,
            )
            # 被膜環
            ax.plot(
                cx + r_coat * np.cos(theta_circ),
                cy + r_coat * np.sin(theta_circ),
                color="orange",
                linewidth=1.5,
                label="被膜" if si == 0 else None,
            )

        # シース外筒
        ax.plot(
            r_sheath_in * np.cos(theta_circ),
            r_sheath_in * np.sin(theta_circ),
            "r--",
            linewidth=1.5,
            label="シース内面",
        )
        ax.plot(
            r_sheath_out * np.cos(theta_circ),
            r_sheath_out * np.sin(theta_circ),
            "r-",
            linewidth=2.0,
            label="シース外面",
        )

        # エンベロープ
        ax.plot(
            r_env * np.cos(theta_circ),
            r_env * np.sin(theta_circ),
            "g:",
            linewidth=1.0,
            label="エンベロープ",
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # --- 側面図（xz平面）: ヘリカル中心線 ---
    ax = axes[2]
    ax.set_title("側面図 (xz平面)", fontsize=11)
    colors = plt.cm.Set1(np.linspace(0, 1, mesh.n_strands))
    for si in range(mesh.n_strands):
        ns, ne = mesh.strand_node_ranges[si]
        coords = mesh.node_coords[ns:ne]
        label = f"#{si}" if si < 3 else (f"#{si}" if si == mesh.n_strands - 1 else None)
        ax.plot(
            coords[:, 2] * 1e3, coords[:, 0] * 1e3, color=colors[si], linewidth=1.0, label=label
        )
    ax.axhline(
        y=r_sheath_out * 1e3, color="r", linestyle="-", linewidth=1.5, alpha=0.5, label="シース外面"
    )
    ax.axhline(y=-r_sheath_out * 1e3, color="r", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("7本撚線 断面構造: 素線 + 被膜 + シース", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "twisted_wire_cross_section.png")
    plt.close(fig)
    print("  -> twisted_wire_cross_section.png")


def plot_fiber_stress_contour():
    """円形断面のファイバー応力コンター図.

    片持ち梁の固定端断面における繊維応力分布を可視化。
    曲げにより上下で引張/圧縮の対称分布が現れることを確認。
    """
    plt = _setup_matplotlib()
    from matplotlib.tri import Triangulation

    from xkep_cae.elements.beam_timo3d import (
        beam3d_section_forces,
        timo_beam3d_ke_global,
    )
    from xkep_cae.sections.beam import BeamSection
    from xkep_cae.sections.fiber import FiberSection

    # 片持ち梁パラメータ
    n_elems = 10
    L = 1.0
    E = 200e9
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 0.02  # 20mm直径
    section = BeamSection.circle(d)
    kappa = 5.0 / 6.0
    P = 5000.0  # 先端荷重 [N]
    R = d / 2.0

    # 解を求める
    n_nodes = n_elems + 1
    ndof = n_nodes * 6
    coords_all = np.zeros((n_nodes, 3))
    coords_all[:, 2] = np.linspace(0, L, n_nodes)
    connectivity = [(i, i + 1) for i in range(n_elems)]

    K = np.zeros((ndof, ndof))
    for n1, n2 in connectivity:
        ec = coords_all[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            ec, E, G, section.A, section.Iy, section.Iz, section.J, kappa, kappa
        )
        edofs = np.array([6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)])
        K[np.ix_(edofs, edofs)] += Ke

    f = np.zeros(ndof)
    f[6 * (n_nodes - 1) + 1] = P  # y方向先端荷重

    fixed = np.arange(6)
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    # 固定端要素(elem 0)と中央要素の断面力を取得
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ファイバー断面を生成
    fiber_sec = FiberSection.circle(d, nr=8, nt=16)

    for ax_idx, (elem_idx, title_pos) in enumerate(
        [
            (0, "固定端 (z=0)"),
            (n_elems // 2, f"中央 (z={L / 2:.1f})"),
            (n_elems - 1, f"先端 (z={L:.1f})"),
        ]
    ):
        n1, n2 = connectivity[elem_idx]
        ec = coords_all[np.array([n1, n2])]
        edofs = np.array([6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)])
        u_elem = u[edofs]
        f1, f2 = beam3d_section_forces(
            ec, u_elem, E, G, section.A, section.Iy, section.Iz, section.J, kappa, kappa
        )

        # 繊維応力 σ_i = N/A + My*z_i/Iy - Mz*y_i/Iz
        # f1 は node1 側の断面力
        sf = f1
        sigma = (
            sf.N / section.A + sf.My * fiber_sec.z / section.Iy - sf.Mz * fiber_sec.y / section.Iz
        )

        # Delaunay三角形分割
        tri = Triangulation(fiber_sec.y * 1e3, fiber_sec.z * 1e3)

        ax = axes[ax_idx]
        vmax = max(abs(sigma.max()), abs(sigma.min()))
        if vmax < 1e-6:
            vmax = 1.0
        contour = ax.tricontourf(
            tri, sigma / 1e6, levels=20, cmap="RdBu_r", vmin=-vmax / 1e6, vmax=vmax / 1e6
        )
        ax.tricontour(tri, sigma / 1e6, levels=10, colors="k", linewidths=0.3, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title_pos}\nN={sf.N:.0f}N, My={sf.My:.1f}Nm", fontsize=9)
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("z [mm]")

        # 断面外形
        theta_c = np.linspace(0, 2 * np.pi, 64)
        ax.plot(R * 1e3 * np.cos(theta_c), R * 1e3 * np.sin(theta_c), "k-", linewidth=1.5)

        cb = fig.colorbar(contour, ax=ax, shrink=0.85)
        cb.set_label("σ [MPa]")

    fig.suptitle(
        f"断面繊維応力分布 (片持ち梁 P={P / 1000:.0f}kN, L={L:.1f}m, d={d * 1e3:.0f}mm)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fiber_stress_cross_section.png")
    plt.close(fig)
    print("  -> fiber_stress_cross_section.png")


# =====================================================================
# 3D梁表面レンダリング — 撚線パイプ形状 + 応力/曲率コンター
# =====================================================================


def _project_3d_to_2d(coords_3d, elev_deg=25.0, azim_deg=45.0):
    """四元数で3D座標を任意視点に回転し、XY平面に投影する.

    Args:
        coords_3d: (N, 3) 3D座標
        elev_deg: 仰角（度）
        azim_deg: 方位角（度）

    Returns:
        coords_2d: (N, 2) 投影座標 (横, 縦)
        depth: (N,) 奥行き（Z'）。描画順序用。
    """
    from xkep_cae.math.quaternion import (
        quat_from_axis_angle,
        quat_multiply,
        quat_to_rotation_matrix,
    )

    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    # 方位角回転（Z軸周り）+ 仰角回転（X軸周り）
    q_azim = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -azim)
    q_elev = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -elev)
    q_view = quat_multiply(q_elev, q_azim)
    R = quat_to_rotation_matrix(q_view)

    rotated = (R @ coords_3d.T).T  # (N, 3)
    coords_2d = rotated[:, :2]
    depth = rotated[:, 2]
    return coords_2d, depth


def _beam_surface_polys_2d(coords, radius, n_circ=12, elev_deg=25.0, azim_deg=45.0):
    """梁中心線から円管表面メッシュを生成し、2D投影された四角形リストを返す.

    Returns:
        polys: list of (4, 2) — 各四角形の2D頂点座標
        depths: list of float — 各四角形の平均奥行き（描画順序用）
    """
    # まず3Dサーフェス座標を生成
    n_pts = len(coords)
    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)

    surface_pts = []  # (n_pts * n_circ, 3)
    for i in range(n_pts):
        if i == 0:
            tang = coords[1] - coords[0]
        elif i == n_pts - 1:
            tang = coords[-1] - coords[-2]
        else:
            tang = coords[i + 1] - coords[i - 1]
        tang = tang / (np.linalg.norm(tang) + 1e-30)

        if abs(tang[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        n1 = np.cross(tang, up)
        n1 = n1 / (np.linalg.norm(n1) + 1e-30)
        n2 = np.cross(tang, n1)

        for th in theta:
            offset = radius * (np.cos(th) * n1 + np.sin(th) * n2)
            surface_pts.append(coords[i] + offset)

    surface_pts = np.array(surface_pts)  # (n_pts * n_circ, 3)
    proj_2d, depth = _project_3d_to_2d(surface_pts, elev_deg, azim_deg)

    # 四角形ポリゴンを構築
    polys = []
    depths = []
    for i in range(n_pts - 1):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            idx00 = i * n_circ + j
            idx01 = i * n_circ + j_next
            idx10 = (i + 1) * n_circ + j
            idx11 = (i + 1) * n_circ + j_next
            poly_verts = np.array(
                [
                    proj_2d[idx00],
                    proj_2d[idx01],
                    proj_2d[idx11],
                    proj_2d[idx10],
                ]
            )
            avg_depth = (depth[idx00] + depth[idx01] + depth[idx10] + depth[idx11]) / 4.0
            polys.append(poly_verts)
            depths.append(avg_depth)

    return polys, depths


def _beam_surface_mesh(coords, radius, n_circ=12):
    """梁中心線座標列から円管表面メッシュを生成.

    Args:
        coords: (n_pts, 3) 中心線座標
        radius: パイプ半径
        n_circ: 円周方向分割数

    Returns:
        X, Y, Z: (n_pts, n_circ) の表面座標
    """
    n_pts = len(coords)
    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)

    X = np.zeros((n_pts, n_circ))
    Y = np.zeros((n_pts, n_circ))
    Z = np.zeros((n_pts, n_circ))

    for i in range(n_pts):
        # 接線ベクトル
        if i == 0:
            tang = coords[1] - coords[0]
        elif i == n_pts - 1:
            tang = coords[-1] - coords[-2]
        else:
            tang = coords[i + 1] - coords[i - 1]
        tang = tang / (np.linalg.norm(tang) + 1e-30)

        # 法線・従法線（Frenet近似）
        if abs(tang[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        n1 = np.cross(tang, up)
        n1 = n1 / (np.linalg.norm(n1) + 1e-30)
        n2 = np.cross(tang, n1)

        for j, th in enumerate(theta):
            offset = radius * (np.cos(th) * n1 + np.sin(th) * n2)
            X[i, j] = coords[i, 0] + offset[0]
            Y[i, j] = coords[i, 1] + offset[1]
            Z[i, j] = coords[i, 2] + offset[2]

    return X, Y, Z


def plot_twisted_wire_3d_surface():
    """撚線の2D投影表面レンダリング（パイプ形状 + 曲率コンター）.

    mplot3dのアスペクト比問題を回避するため、四元数ベースの
    3D→2D投影方式を使用（status-126で置換）。
    """
    plt = _setup_matplotlib()
    from matplotlib.collections import PolyCollection

    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

    wire_d = 2.0e-3
    pitch = 40.0e-3
    mesh = make_twisted_wire_mesh(
        n_strands=7,
        wire_diameter=wire_d,
        pitch=pitch,
        length=pitch,
        n_elems_per_strand=32,
    )
    r_wire = wire_d / 2.0
    n_circ = 12
    elev, azim = 25.0, 45.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # --- 1. 2D投影表面レンダリング（全体） ---
    colors_strand = plt.cm.Set2(np.linspace(0, 1, mesh.n_strands))

    all_polys = []
    all_depths = []
    all_colors = []
    for si in range(mesh.n_strands):
        ns, ne = mesh.strand_node_ranges[si]
        coords = mesh.node_coords[ns:ne]
        polys, depths = _beam_surface_polys_2d(coords, r_wire, n_circ, elev, azim)
        all_polys.extend(polys)
        all_depths.extend(depths)
        all_colors.extend([colors_strand[si]] * len(polys))

    # 深度ソート（奥→手前の順に描画）
    order = np.argsort(all_depths)
    sorted_polys = [all_polys[i] for i in order]
    sorted_colors = [all_colors[i] for i in order]

    pc = PolyCollection(sorted_polys, facecolors=sorted_colors, edgecolors="none", alpha=0.7)
    ax1.add_collection(pc)
    _setup_2d_projected_ax(ax1, mesh.node_coords, elev, azim)
    ax1.set_title("7本撚線 2D投影パイプ表面", fontsize=11)

    # --- 2. 曲率コンター付き2D投影表面 ---
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    all_curvatures = []

    for si in range(mesh.n_strands):
        ns, ne = mesh.strand_node_ranges[si]
        coords = mesh.node_coords[ns:ne]
        curvature = np.zeros(len(coords))
        for k in range(1, len(coords) - 1):
            t_prev = coords[k] - coords[k - 1]
            t_next = coords[k + 1] - coords[k]
            ds = (np.linalg.norm(t_prev) + np.linalg.norm(t_next)) / 2.0
            if ds > 1e-15:
                dt = t_next / (np.linalg.norm(t_next) + 1e-30) - t_prev / (
                    np.linalg.norm(t_prev) + 1e-30
                )
                curvature[k] = np.linalg.norm(dt) / ds
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
        all_curvatures.append(curvature)

    kappa_max = max(c.max() for c in all_curvatures)
    if kappa_max < 1e-15:
        kappa_max = 1.0
    sm.set_clim(0, kappa_max)

    all_polys2 = []
    all_depths2 = []
    all_colors2 = []
    for si in range(mesh.n_strands):
        ns, ne = mesh.strand_node_ranges[si]
        coords = mesh.node_coords[ns:ne]
        curvature = all_curvatures[si]
        polys, depths = _beam_surface_polys_2d(coords, r_wire, n_circ, elev, azim)
        n_pts = len(coords)
        for pi, (poly, dep) in enumerate(zip(polys, depths, strict=True)):
            # ポリゴン→要素番号から曲率を取得
            elem_idx = pi // n_circ
            if elem_idx < n_pts - 1:
                kappa_avg = (curvature[elem_idx] + curvature[elem_idx + 1]) / 2.0
            else:
                kappa_avg = curvature[-1]
            all_polys2.append(poly)
            all_depths2.append(dep)
            all_colors2.append(sm.to_rgba(kappa_avg))

    order2 = np.argsort(all_depths2)
    sorted_polys2 = [all_polys2[i] for i in order2]
    sorted_colors2 = [all_colors2[i] for i in order2]

    pc2 = PolyCollection(sorted_polys2, facecolors=sorted_colors2, edgecolors="none", alpha=0.8)
    ax2.add_collection(pc2)
    _setup_2d_projected_ax(ax2, mesh.node_coords, elev, azim)
    ax2.set_title("曲率コンター κ [1/m]", fontsize=11)

    cbar = fig.colorbar(sm, ax=ax2, shrink=0.8)
    cbar.set_label("κ [1/m]")

    fig.suptitle("撚線2D投影 梁表面レンダリング + 曲率コンター", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "twisted_wire_3d_surface.png")
    plt.close(fig)
    print("  -> twisted_wire_3d_surface.png")


def plot_beam_3d_stress_contour():
    """片持ち梁の2D投影表面に応力コンターを描画.

    mplot3dのアスペクト比問題を回避するため、四元数ベースの
    3D→2D投影方式を使用（status-126で置換）。
    """
    plt = _setup_matplotlib()
    from matplotlib.collections import PolyCollection

    from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
    from xkep_cae.sections.beam import BeamSection

    L = 1.0
    E = 200e9
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 0.04
    section = BeamSection.circle(d)
    kappa = 5.0 / 6.0
    P = 10000.0
    n_elems = 20
    n_nodes = n_elems + 1
    ndof = n_nodes * 6
    r = d / 2.0

    # 節点座標（z軸方向）
    coords_all = np.zeros((n_nodes, 3))
    coords_all[:, 2] = np.linspace(0, L, n_nodes)
    connectivity = [(i, i + 1) for i in range(n_elems)]

    # 剛性行列アセンブリ
    K = np.zeros((ndof, ndof))
    for n1, n2 in connectivity:
        ec = coords_all[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            ec, E, G, section.A, section.Iy, section.Iz, section.J, kappa, kappa
        )
        edofs = np.array([6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)])
        K[np.ix_(edofs, edofs)] += Ke

    f = np.zeros(ndof)
    f[6 * (n_nodes - 1) + 1] = P  # y方向先端荷重

    fixed = np.arange(6)
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    # 変形後座標
    deformed = coords_all.copy()
    for i in range(n_nodes):
        deformed[i] += u[6 * i : 6 * i + 3]

    # 各要素の曲げ応力（最外縁）: σ = M*y/I, M ≈ P*(L-z)
    sigma_nodes = np.zeros(n_nodes)
    for i in range(n_nodes):
        z_pos = coords_all[i, 2]
        M = P * (L - z_pos)
        sigma_nodes[i] = M * r / section.Iy

    # 2D投影表面生成
    n_circ = 16
    elev, azim = 20.0, -60.0

    sigma_max = sigma_nodes.max()
    if sigma_max < 1e-6:
        sigma_max = 1.0
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, sigma_max / 1e6))

    # 変形後のパイプ表面2Dポリゴン
    polys_def, depths_def = _beam_surface_polys_2d(deformed, r, n_circ, elev, azim)

    # 各ポリゴンの応力色
    colors_stress = []
    for pi in range(len(polys_def)):
        elem_idx = pi // n_circ
        if elem_idx < n_nodes - 1:
            sigma_avg = (sigma_nodes[elem_idx] + sigma_nodes[elem_idx + 1]) / 2.0
        else:
            sigma_avg = sigma_nodes[-1]
        colors_stress.append(sm.to_rgba(sigma_avg / 1e6))

    # 深度ソート
    order = np.argsort(depths_def)
    sorted_polys = [polys_def[i] for i in order]
    sorted_colors = [colors_stress[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    pc = PolyCollection(sorted_polys, facecolors=sorted_colors, edgecolors="none", alpha=0.85)
    ax.add_collection(pc)

    # 未変形形状を線で表示
    undef_2d, _ = _project_3d_to_2d(coords_all, elev, azim)
    ax.plot(undef_2d[:, 0], undef_2d[:, 1], "k--", linewidth=1, alpha=0.5, label="未変形")

    _setup_2d_projected_ax(ax, deformed, elev, azim)
    ax.set_title(
        f"片持ち梁2D投影応力コンター (P={P / 1000:.0f}kN, d={d * 1e3:.0f}mm)",
        fontsize=12,
    )
    ax.legend(fontsize=9)

    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("σ_bending [MPa]")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beam_3d_stress_contour.png")
    plt.close(fig)
    print("  -> beam_3d_stress_contour.png")


# =====================================================================
# 接触力ベクトル場 3D 表示（quiver3D）
# =====================================================================


def _run_for_visualization(n_strands, *, layers=(1,), force_per_node=5.0):
    """可視化用に NCP 解析を実行し (mesh, result, mgr) を返す.

    径方向圧縮荷重で接触を活性化する。
    """
    from xkep_cae.contact.pair import ContactConfig, ContactManager
    from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp
    from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
    from xkep_cae.sections.beam import BeamSection

    NDOF = 6
    E = 200e9
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    wire_d = 0.002
    section = BeamSection.circle(wire_d)
    kappa_s = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    pitch = 0.040

    mesh = make_twisted_wire_mesh(
        n_strands,
        wire_d,
        pitch,
        length=0.0,
        n_elems_per_strand=16,
        n_pitches=1.0,
        gap=0.0001,
    )
    nc = mesh.node_coords
    conn = mesh.connectivity
    ndof_total = mesh.n_nodes * NDOF

    # 剛性行列アセンブリ
    K = np.zeros((ndof_total, ndof_total))
    for elem in conn:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = nc[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa_s,
            kappa_s,
        )
        edofs = np.array(
            [NDOF * n1 + d for d in range(NDOF)] + [NDOF * n2 + d for d in range(NDOF)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    # 境界条件: 全素線始端固定 + 中心素線全拘束
    fixed = set()
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        for d in range(NDOF):
            fixed.add(NDOF * nodes[0] + d)
        if sid == 0:
            for node in nodes:
                for d in range(NDOF):
                    fixed.add(NDOF * node + d)
    fixed_dofs = np.array(sorted(fixed), dtype=int)

    # 径方向圧縮荷重
    f_ext = np.zeros(ndof_total)
    for sid in range(1, mesh.n_strands):
        info = mesh.strand_infos[sid]
        if info.layer not in layers:
            continue
        nodes = mesh.strand_nodes(sid)
        mid_node = nodes[len(nodes) // 2]
        pos = nc[mid_node]
        r_xy = np.linalg.norm(pos[:2])
        if r_xy > 1e-10:
            r_dir = -pos[:2] / r_xy
            f_ext[NDOF * mid_node + 0] = r_dir[0] * force_per_node
            f_ext[NDOF * mid_node + 1] = r_dir[1] * force_per_node

    elem_layer_map = mesh.build_elem_layer_map()
    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=E,
            beam_I=section.Iy,
            k_pen_scaling="sqrt",
            k_t_ratio=0.1,
            mu=0.0,
            g_on=0.001,
            g_off=0.002,
            use_friction=False,
            use_line_search=False,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=1.0,
            k_pen_max=1e12,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            linear_solver="auto",
            no_deactivation_within_step=True,
            preserve_inactive_lambda=True,
            lambda_warmstart_neighbor=True,
            chattering_window=3,
            k_pen_continuation=True,
            k_pen_continuation_start=0.1,
            k_pen_continuation_steps=5,
            adjust_initial_penetration=True,
            contact_force_ramp=True,
            contact_force_ramp_iters=5,
            adaptive_timestepping=True,
            dt_grow_factor=1.3,
            dt_shrink_factor=0.5,
            dt_grow_iter_threshold=8,
            dt_shrink_iter_threshold=20,
            dt_contact_change_threshold=0.3,
            residual_scaling="rms",
        ),
    )

    result = newton_raphson_contact_ncp(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        mesh.radii,
        n_load_steps=40,
        max_iter=100,
        tol_force=1e-4,
        tol_ncp=1e-4,
        broadphase_margin=0.01,
        show_progress=False,
        adaptive_omega=True,
        omega_init=0.3,
        omega_min=0.02,
        omega_max=0.8,
        omega_shrink=0.5,
        omega_growth=1.1,
        active_set_update_interval=5,
        du_norm_cap=3.0,
        adaptive_timestepping=True,
    )

    return mesh, result, mgr


def _draw_wire_2d(
    ax, coords, radius, color, alpha=0.7, n_circ=12, elev_deg=25.0, azim_deg=45.0, zorder_base=0
):
    """撚線1本を2D投影してPolyCollectionで描画.

    深度ソートにより正しい前後関係を表現する。
    """
    from matplotlib.collections import PolyCollection

    polys, depths = _beam_surface_polys_2d(
        coords,
        radius,
        n_circ,
        elev_deg,
        azim_deg,
    )
    if not polys:
        return

    # 深度順にソート（奥→手前）
    sorted_idx = np.argsort(depths)
    sorted_polys = [polys[i] for i in sorted_idx]

    pc = PolyCollection(
        sorted_polys,
        facecolors=color,
        edgecolors="none",
        alpha=alpha,
        zorder=zorder_base,
    )
    ax.add_collection(pc)


def _draw_wires_2d(
    ax, mesh, coords_array, radius, n_circ=12, elev_deg=25.0, azim_deg=45.0, layer_colors=None
):
    """全素線を2D投影して描画（層ごとに色分け）."""
    if layer_colors is None:
        layer_colors = {
            0: "#e41a1c",
            1: "#377eb8",
            2: "#4daf4a",
            3: "#984ea3",
        }
    for si in range(mesh.n_strands):
        ns, ne = mesh.strand_node_ranges[si]
        strand_coords = coords_array[ns:ne]
        layer = mesh.strand_infos[si].layer
        color = layer_colors.get(layer, "#999999")
        _draw_wire_2d(
            ax,
            strand_coords,
            radius,
            color,
            alpha=0.7,
            n_circ=n_circ,
            elev_deg=elev_deg,
            azim_deg=azim_deg,
        )


def _setup_2d_projected_ax(ax, coords_3d, elev_deg, azim_deg, margin_frac=0.05):
    """2D投影後の軸範囲を等方アスペクト比で設定."""
    proj_2d, _ = _project_3d_to_2d(coords_3d, elev_deg, azim_deg)
    x_range = proj_2d[:, 0].max() - proj_2d[:, 0].min()
    y_range = proj_2d[:, 1].max() - proj_2d[:, 1].min()
    margin = max(x_range, y_range) * margin_frac
    ax.set_xlim(proj_2d[:, 0].min() - margin, proj_2d[:, 0].max() + margin)
    ax.set_ylim(proj_2d[:, 1].min() - margin, proj_2d[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def plot_contact_force_vectors_3d():
    """接触力ベクトル場の2D投影表示 + 接触診断パネル.

    NCP解析を実行し、4パネルで接触状態を可視化:
    1. 接触力ベクトル場（2D投影、法線力を矢印で表示）
    2. 接触ギャップ分布ヒストグラム（貫入量の分布）
    3. 接触力ノルムの荷重ステップ推移
    4. 断面方向（端面）の接触力ベクトル
    """
    plt = _setup_matplotlib()

    from xkep_cae.contact.pair import ContactStatus

    print("    Running 7-strand NCP for contact force visualization...")
    mesh, result, mgr = _run_for_visualization(7, layers=(1,))

    if not result.converged:
        print("    WARNING: NCP did not converge, plotting partial result")

    wire_d = 0.002
    r_wire = wire_d / 2.0
    n_circ = 12
    ndof_per_node = 6

    # 変形後座標
    deformed = mesh.node_coords.copy()
    for i in range(mesh.n_nodes):
        deformed[i] += result.u[ndof_per_node * i : ndof_per_node * i + 3]

    active_pairs = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
    all_pairs = mgr.pairs

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- Panel 1: 側面ビュー + 接触力ベクトル ---
    ax1 = axes[0, 0]
    elev, azim = 15.0, 45.0
    _draw_wires_2d(
        ax1,
        mesh,
        deformed,
        r_wire,
        n_circ,
        elev,
        azim,
        layer_colors={0: "#cccccc", 1: "#aaccee"},
    )

    if active_pairs:
        f_max = max(p.state.p_n for p in active_pairs)
        if f_max < 1e-15:
            f_max = 1.0

        for pair in active_pairs:
            na = pair.nodes_a
            s_param = pair.state.s
            pos_3d = (1.0 - s_param) * deformed[na[0]] + s_param * deformed[na[1]]
            normal_3d = pair.state.normal
            p_n = pair.state.p_n

            scale = 5.0 * wire_d * (p_n / f_max)
            tip_3d = pos_3d + normal_3d * scale

            pts_3d = np.array([pos_3d, tip_3d])
            pts_2d, _ = _project_3d_to_2d(pts_3d, elev, azim)

            color_val = p_n / f_max
            color = plt.cm.hot_r(color_val)
            ax1.annotate(
                "",
                xy=(pts_2d[1, 0], pts_2d[1, 1]),
                xytext=(pts_2d[0, 0], pts_2d[0, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.5 + 2.0 * color_val,
                ),
                zorder=100,
            )

    _setup_2d_projected_ax(ax1, deformed, elev, azim)
    ax1.set_title(
        f"接触力ベクトル場 (active={len(active_pairs)})",
        fontsize=10,
    )
    ax1.set_xlabel("投影 x [m]")
    ax1.set_ylabel("投影 y [m]")

    # --- Panel 2: ギャップ分布ヒストグラム ---
    ax2 = axes[0, 1]
    gaps = [p.state.gap for p in all_pairs]
    gaps_active = [p.state.gap for p in active_pairs]

    if gaps:
        ax2.hist(gaps, bins=40, alpha=0.5, color="steelblue", label="全ペア")
    if gaps_active:
        ax2.hist(gaps_active, bins=20, alpha=0.7, color="crimson", label="ACTIVE")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1, label="g=0（接触面）")
    ax2.set_xlabel("ギャップ g [m]")
    ax2.set_ylabel("頻度")
    ax2.set_title("接触ギャップ分布", fontsize=10)
    ax2.legend(fontsize=8)

    # 貫入統計
    penetrations = [g for g in gaps if g < 0]
    if penetrations:
        ax2.text(
            0.02,
            0.95,
            f"貫入ペア: {len(penetrations)}\n"
            f"最大貫入: {min(penetrations):.2e} m\n"
            f"平均貫入: {np.mean(penetrations):.2e} m",
            transform=ax2.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    # --- Panel 3: 接触力推移 ---
    ax3 = axes[1, 0]
    cf_hist = result.contact_force_history
    if cf_hist:
        ax3.plot(range(len(cf_hist)), cf_hist, "o-", color="crimson", markersize=3, linewidth=1.5)
    ax3.set_xlabel("荷重ステップ")
    ax3.set_ylabel("接触力ノルム [N]")
    ax3.set_title("接触力ノルムの推移", fontsize=10)
    ax3.set_yscale("log") if cf_hist and max(cf_hist) > 0 else None

    # --- Panel 4: 断面ビュー（端面方向）+ 接触力 ---
    ax4 = axes[1, 1]
    elev_end, azim_end = 0.0, 0.0  # Z軸方向から見た断面

    _draw_wires_2d(
        ax4,
        mesh,
        deformed,
        r_wire,
        n_circ,
        elev_end,
        azim_end,
        layer_colors={0: "#cccccc", 1: "#aaccee"},
    )

    if active_pairs:
        for pair in active_pairs:
            na = pair.nodes_a
            s_param = pair.state.s
            pos_3d = (1.0 - s_param) * deformed[na[0]] + s_param * deformed[na[1]]
            normal_3d = pair.state.normal
            p_n = pair.state.p_n
            scale = 5.0 * wire_d * (p_n / f_max)
            tip_3d = pos_3d + normal_3d * scale

            pts_3d = np.array([pos_3d, tip_3d])
            pts_2d, _ = _project_3d_to_2d(pts_3d, elev_end, azim_end)

            color_val = p_n / f_max
            color = plt.cm.hot_r(color_val)
            ax4.annotate(
                "",
                xy=(pts_2d[1, 0], pts_2d[1, 1]),
                xytext=(pts_2d[0, 0], pts_2d[0, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.5 + 2.0 * color_val,
                ),
                zorder=100,
            )

    _setup_2d_projected_ax(ax4, deformed, elev_end, azim_end)
    ax4.set_title("断面ビュー（端面方向）+ 接触力", fontsize=10)
    ax4.set_xlabel("投影 x [m]")
    ax4.set_ylabel("投影 y [m]")

    fig.suptitle(
        f"接触診断パネル (7本撚線, converged={result.converged})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "contact_force_vectors_3d.png")
    plt.close(fig)
    print("  -> contact_force_vectors_3d.png")


def plot_deformed_twisted_wire_3d():
    """変形後撚線の2D投影レンダリング（NCP解の変位適用）.

    複数視点（側面・端面・斜め）から変形前後を比較する。
    """
    plt = _setup_matplotlib()

    print("    Running 7-strand NCP for deformed visualization...")
    mesh, result, mgr = _run_for_visualization(7, layers=(1,))

    wire_d = 0.002
    r_wire = wire_d / 2.0
    n_circ = 12
    ndof_per_node = 6

    deformed = mesh.node_coords.copy()
    for i in range(mesh.n_nodes):
        deformed[i] += result.u[ndof_per_node * i : ndof_per_node * i + 3]

    views = [
        ("斜めビュー", 25.0, 45.0),
        ("端面ビュー", 0.0, 0.0),
        ("側面ビュー", 0.0, 90.0),
    ]
    layer_colors_ref = {0: "#cccccc", 1: "#bbbbbb"}
    layer_colors_def = {0: "#e41a1c", 1: "#377eb8"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, (title, elev, azim) in enumerate(views):
        # 変形前
        ax_ref = axes[0, col]
        _draw_wires_2d(
            ax_ref,
            mesh,
            mesh.node_coords,
            r_wire,
            n_circ,
            elev,
            azim,
            layer_colors=layer_colors_ref,
        )
        _setup_2d_projected_ax(ax_ref, mesh.node_coords, elev, azim)
        ax_ref.set_title(f"変形前 — {title}", fontsize=10)

        # 変形後
        ax_def = axes[1, col]
        _draw_wires_2d(
            ax_def,
            mesh,
            deformed,
            r_wire,
            n_circ,
            elev,
            azim,
            layer_colors=layer_colors_def,
        )
        _setup_2d_projected_ax(ax_def, deformed, elev, azim)
        ax_def.set_title(f"変形後 — {title}", fontsize=10)

    fig.suptitle(
        f"撚線 変形前後比較 (converged={result.converged})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deformed_twisted_wire_3d.png")
    plt.close(fig)
    print("  -> deformed_twisted_wire_3d.png")


def plot_twisted_wire_3d_surface_multi(n_strands_list=None):
    """19本/37本撚線の2D投影レンダリング（初期形状、複数視点）."""
    plt = _setup_matplotlib()

    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

    if n_strands_list is None:
        n_strands_list = [19, 37]

    wire_d = 2.0e-3
    pitch = 40.0e-3
    r_wire = wire_d / 2.0
    n_circ = 12

    views = [
        ("斜め", 25.0, 45.0),
        ("端面", 0.0, 0.0),
    ]

    fig, axes = plt.subplots(
        len(n_strands_list),
        len(views),
        figsize=(8 * len(views), 8 * len(n_strands_list)),
    )
    if len(n_strands_list) == 1:
        axes = axes[np.newaxis, :]

    layer_colors = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3"}

    for row, n_s in enumerate(n_strands_list):
        mesh = make_twisted_wire_mesh(
            n_strands=n_s,
            wire_diameter=wire_d,
            pitch=pitch,
            length=pitch,
            n_elems_per_strand=16,
        )
        layers = set(info.layer for info in mesh.strand_infos)
        layer_str = "+".join(f"L{ly}" for ly in sorted(layers))

        for col, (view_name, elev, azim) in enumerate(views):
            ax = axes[row, col]
            _draw_wires_2d(
                ax,
                mesh,
                mesh.node_coords,
                r_wire,
                n_circ,
                elev,
                azim,
                layer_colors=layer_colors,
            )
            _setup_2d_projected_ax(ax, mesh.node_coords, elev, azim)
            ax.set_title(f"{n_s}本撚線 ({layer_str}) — {view_name}", fontsize=10)

    fig.suptitle("撚線 2D投影レンダリング（初期形状）", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "twisted_wire_3d_surface_multi.png")
    plt.close(fig)
    print("  -> twisted_wire_3d_surface_multi.png")


# =====================================================================
# メイン
# =====================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating verification plots to {OUTPUT_DIR}/")
    print()

    print("[Phase 2.1-2.2] 梁要素")
    plot_cantilever_eb_timo()

    print("[Phase 2.3] 3D梁要素")
    plot_beam3d_torsion_bending()

    print("[Phase 2.5] Cosserat rod")
    plot_cosserat_convergence()

    print("[Phase 2.6] 数値試験フレームワーク")
    plot_numerical_tests_accuracy()

    print("[Phase 3] Euler elastica")
    plot_euler_elastica_moment()
    plot_euler_elastica_tip_load()

    print("[Phase 4.1] 弾塑性構成則")
    plot_stress_strain_isotropic()
    plot_hysteresis_loop()
    plot_bauschinger_comparison()
    plot_load_displacement_bar()

    print("[Phase 4.2] ファイバーモデル")
    plot_fiber_moment_curvature()
    plot_fiber_cantilever_load_displacement()

    print("[Phase C] 梁–梁接触")
    plot_contact_crossing_beam()
    plot_contact_penetration_control()
    plot_contact_friction_stick_slip()

    print("[Phase S3] チューニングタスク検証")
    tuning_result = plot_tuning_scaling_analysis()
    plot_tuning_contact_topology(tuning_result)
    plot_tuning_timing_breakdown(tuning_result)
    plot_tuning_wire_cross_section(tuning_result)
    plot_tuning_acceptance_summary(tuning_result)
    plot_tuning_sensitivity_heatmap()

    print("[Phase 3+5] 幾何学非線形 — CR応力/曲率コンター")
    plot_cr_stress_curvature_contour()

    print("[Phase 5] 動的解析 — エネルギー時刻歴")
    plot_dynamics_energy_history()

    print("[Phase 5] 動的解析 — 変位応答")
    plot_dynamics_displacement_response()

    print("[Phase 4.7] 撚線断面構造 — 被膜/シース")
    plot_twisted_wire_3d_cross_section()

    print("[Phase 4.7] 断面繊維応力コンター")
    plot_fiber_stress_contour()

    print("[3D] 撚線3D梁表面レンダリング + 曲率コンター")
    plot_twisted_wire_3d_surface()

    print("[3D] 片持ち梁3D応力コンター")
    plot_beam_3d_stress_contour()

    print("[3D] 接触力ベクトル場 + 接触診断パネル")
    plot_contact_force_vectors_3d()

    print("[3D] 変形後撚線 2D投影レンダリング")
    plot_deformed_twisted_wire_3d()

    print("[3D] 19本/37本撚線 2D投影レンダリング")
    plot_twisted_wire_3d_surface_multi()

    print()
    print("Done. All verification plots generated.")


if __name__ == "__main__":
    main()
