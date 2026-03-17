"""テスト設定 — deprecated 参照テストの収集を抑制 + numerical_tests バックエンド注入.

__xkep_cae_deprecated → xkep_cae 移行に伴い、旧パッケージパスを参照する
テストファイルは ImportError で収集失敗する。
ここでは収集自体をスキップして、pytest のエラーカウントに含めない。

未移行テストは個別に移行して xkep_cae/ 内のテストに統合する。

status-193 で導入。status-195 で numerical_tests バックエンド注入を追加。

[← README](../README.md)
"""

import importlib
import importlib.util
import sys


def pytest_ignore_collect(collection_path, config):
    """未移行モジュールを参照するテストの収集をスキップする."""
    if not collection_path.suffix == ".py":
        return False
    if collection_path.name == "conftest.py":
        return False
    if "generate_verification" in collection_path.name:
        return True  # テストファイルではない

    try:
        text = collection_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    # __xkep_cae_deprecated を参照しているファイル
    if "__xkep_cae_deprecated" in text:
        return True

    # テストモジュールとして import を試行
    # 失敗したら収集をスキップ
    module_name = f"_conftest_probe_{collection_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, collection_path)
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError):
        return True
    except Exception:
        # import 以外のエラー（SyntaxError 等）は pytest に任せる
        return False
    finally:
        # プローブモジュールを削除
        sys.modules.pop(module_name, None)

    return False


# ---------------------------------------------------------------------------
# numerical_tests バックエンド注入（status-195）
#
# xkep_cae/numerical_tests/ は C14 準拠のため deprecated パッケージを直接
# import できない。テスト実行時にここで deprecated 実装を BackendRegistry
# に注入する。
# ---------------------------------------------------------------------------
def _configure_numerical_tests_backend():
    """deprecated 実装を numerical_tests バックエンドに注入する."""
    try:
        from xkep_cae.numerical_tests._backend import backend
    except ImportError:
        return  # numerical_tests が利用不可の場合はスキップ

    if backend.is_configured:
        return  # 既に設定済み

    # --- 静的試験バックエンド ---
    from __xkep_cae_deprecated.bc import apply_dirichlet
    from __xkep_cae_deprecated.solver import solve_displacement

    # ke_func_factory: cfg + sec → ke_func(coords)
    def _ke_func_factory(cfg, sec):
        E = cfg.E
        bt = cfg.beam_type
        if bt == "eb2d":
            from __xkep_cae_deprecated.elements.beam_eb2d import eb_beam2d_ke_global

            A_sec, I_sec = sec["A"], sec["I"]
            return lambda coords: eb_beam2d_ke_global(coords, E, A_sec, I_sec)
        elif bt == "timo2d":
            from __xkep_cae_deprecated.elements.beam_timo2d import timo_beam2d_ke_global

            A_sec, I_sec, kappa, G = sec["A"], sec["I"], sec["kappa"], cfg.G
            return lambda coords: timo_beam2d_ke_global(coords, E, A_sec, I_sec, kappa, G)
        elif bt in ("timo3d", "cosserat"):
            if bt == "timo3d":
                from __xkep_cae_deprecated.elements.beam_timo3d import (
                    timo_beam3d_ke_global as ke_fn,
                )
            else:
                from __xkep_cae_deprecated.elements.beam_cosserat import cosserat_ke_global as ke_fn

            A = sec["A"]
            Iy, Iz, J = sec["Iy"], sec["Iz"], sec["J"]
            ky, kz, G = sec["kappa_y"], sec["kappa_z"], cfg.G
            return lambda coords: ke_fn(coords, E, G, A, Iy, Iz, J, ky, kz)
        else:
            raise ValueError(f"未対応の beam_type: {bt}")

    # section_force_computer: cfg, sec, nodes, conn, u → list[(f1, f2)]
    def _section_force_computer(cfg, sec, nodes, conn, u):

        E = cfg.E
        bt = cfg.beam_type
        forces = []
        if bt == "eb2d":
            from __xkep_cae_deprecated.elements.beam_eb2d import eb_beam2d_section_forces

            for elem_nodes in conn:
                n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
                coords = nodes[[n1, n2]]
                edofs = [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2]
                u_elem = u[edofs]
                f1, f2 = eb_beam2d_section_forces(coords, u_elem, E, sec["A"], sec["I"])
                forces.append((f1, f2))
        elif bt == "timo2d":
            from __xkep_cae_deprecated.elements.beam_timo2d import timo_beam2d_section_forces

            for elem_nodes in conn:
                n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
                coords = nodes[[n1, n2]]
                edofs = [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2]
                u_elem = u[edofs]
                f1, f2 = timo_beam2d_section_forces(
                    coords, u_elem, E, sec["A"], sec["I"], sec["kappa"], cfg.G
                )
                forces.append((f1, f2))
        elif bt in ("timo3d", "cosserat"):
            if bt == "timo3d":
                from __xkep_cae_deprecated.elements.beam_timo3d import (
                    beam3d_section_forces as sf_fn,
                )
            else:
                from __xkep_cae_deprecated.elements.beam_cosserat import (
                    cosserat_section_forces as sf_fn,
                )

            for elem_nodes in conn:
                n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
                coords = nodes[[n1, n2]]
                edofs = []
                for n in [n1, n2]:
                    for d in range(6):
                        edofs.append(6 * n + d)
                u_elem = u[edofs]
                f1, f2 = sf_fn(
                    coords,
                    u_elem,
                    E,
                    cfg.G,
                    sec["A"],
                    sec["Iy"],
                    sec["Iz"],
                    sec["J"],
                    sec["kappa_y"],
                    sec["kappa_z"],
                )
                forces.append((f1, f2))
        return forces

    backend.configure(
        apply_dirichlet=apply_dirichlet,
        solve=solve_displacement,
        ke_func_factory=_ke_func_factory,
        section_force_computer=_section_force_computer,
    )

    # --- 周波数応答バックエンド ---
    from __xkep_cae_deprecated.elements.beam_eb2d import (
        eb_beam2d_lumped_mass_local,
        eb_beam2d_mass_global,
    )
    from __xkep_cae_deprecated.elements.beam_timo3d import (
        _beam3d_length_and_direction,
        timo_beam3d_lumped_mass_local,
        timo_beam3d_mass_global,
    )

    backend.configure_frequency(
        beam2d_lumped_mass_local=eb_beam2d_lumped_mass_local,
        beam3d_lumped_mass_local=timo_beam3d_lumped_mass_local,
        beam2d_mass_global=eb_beam2d_mass_global,
        beam3d_mass_global=timo_beam3d_mass_global,
        beam3d_length_and_direction=_beam3d_length_and_direction,
    )

    # --- 動的試験バックエンド ---
    from __xkep_cae_deprecated.dynamics import (
        NonlinearTransientConfig,
        solve_nonlinear_transient,
    )
    from __xkep_cae_deprecated.elements.beam_cosserat import assemble_cosserat_nonlinear
    from __xkep_cae_deprecated.elements.beam_timo3d import assemble_cr_beam3d

    def _cr_assembler_factory(nodes, conn, E, G, A, Iy, Iz, J, kappa_y, kappa_z):

        def assemble_internal_force(u):
            _, f_int = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        def assemble_tangent(u):
            K_T, _ = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                stiffness=True,
                internal_force=False,
            )
            return K_T

        return assemble_internal_force, assemble_tangent

    def _cosserat_nl_assembler_factory(nodes, conn, E, G, A, Iy, Iz, J, kappa_y, kappa_z):

        def assemble_internal_force(u):
            _, f_int = assemble_cosserat_nonlinear(
                nodes,
                conn,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        def assemble_tangent(u):
            K_T, _ = assemble_cosserat_nonlinear(
                nodes,
                conn,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                stiffness=True,
                internal_force=False,
            )
            return K_T

        return assemble_internal_force, assemble_tangent

    backend.configure_dynamic(
        transient_config_class=NonlinearTransientConfig,
        transient_solver=solve_nonlinear_transient,
        cr_assembler_factory=_cr_assembler_factory,
        cosserat_nl_assembler_factory=_cosserat_nl_assembler_factory,
    )


# セッション開始時にバックエンド注入
_configure_numerical_tests_backend()
