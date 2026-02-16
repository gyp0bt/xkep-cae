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
        E, G, fs.A, fs.Iy, fs.Iz, fs.J, kappa_cowper, kappa_cowper,
    )
    plas_pp = Plasticity1D(E=E, iso=IsotropicHardening(sigma_y0=sigma_y, H_iso=0.0))

    kappa_y_limit = sigma_y / (E * h / 2.0)
    kappas = np.linspace(0, 15 * kappa_y_limit, 200)
    moments_pp = []
    state = CosseratFiberPlasticState.create(fs.n_fibers)
    for kap in kappas:
        strain = np.array([0.0, 0.0, 0.0, 0.0, kap, 0.0])
        stress, _, state_new = _compute_generalized_stress_fiber(
            strain, C_elastic, plas_pp, state, fs,
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
            strain, C_elastic, plas_iso, state, fs,
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
        kappas / kappa_y_limit, M_elastic / M_y,
        "k--", lw=1, label="Elastic (EI*kappa)",
    )
    ax.plot(
        kappas / kappa_y_limit, moments_pp / M_y,
        "b-", lw=2, label="Fiber: perfectly plastic (H=0)",
    )
    ax.plot(
        kappas / kappa_y_limit, moments_iso / M_y,
        "r-", lw=2, label="Fiber: isotropic hardening (H=1000)",
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
                n_elems, L, rod, mat, u_, _states, plas, fs,
                stiffness=False, internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_, _states=states):
            K, _, _ = assemble_cosserat_beam_fiber(
                n_elems, L, rod, mat, u_, _states, plas, fs,
                stiffness=True, internal_force=False,
            )
            return sp.csr_matrix(K)

        result = newton_raphson(
            f_ext, fixed_dofs, _Kt, _fint,
            n_load_steps=1, u0=u, show_progress=False,
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
        theta_elastic * 1000, moments / M_y,
        "k--", lw=1.5, label="Elastic (analytical)",
    )
    ax.plot(
        theta_tips * 1000, moments / M_y,
        "ro-", ms=4, lw=2, label="Fiber model (NR)",
    )
    ax.axhline(1.0, color="gray", ls="-.", lw=1, alpha=0.5, label="M_y (elastic limit)")
    ax.axhline(1.5, color="gray", ls=":", lw=1, alpha=0.5, label="M_p = 1.5*M_y")
    ax.set_xlabel("Tip rotation [mrad]")
    ax.set_ylabel("M / M_y")
    ax.set_title(
        f"Fiber Model: Cantilever Tip Moment ({n_elems} elements, "
        f"sigma_y={sigma_y}, H={H_iso})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fiber_cantilever_moment.png")
    plt.close(fig)
    print("  -> fiber_cantilever_moment.png")


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

    print()
    print("Done. All verification plots generated.")


if __name__ == "__main__":
    main()
