"""接触付き Newton-Raphson ソルバー（Outer/Inner 分離）.

Phase C2: 法線接触のみ（摩擦なし）。

設計方針:
- 既存の ``newton_raphson()`` を内部利用する「包装関数」方式
- Outer loop: 接触候補検出 + 幾何更新 + AL乗数更新
- Inner loop: 最近接点 (s,t) 固定で Newton-Raphson

収束判定（Outer）:
- 最近接パラメータ |Δs|, |Δt| が閾値以下
- または Inner が 1 反復で収束

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §6, §12
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.assembly import compute_contact_force, compute_contact_stiffness
from xkep_cae.contact.law_normal import (
    initialize_penalty_stiffness,
    update_al_multiplier,
)
from xkep_cae.contact.pair import ContactManager, ContactStatus


@dataclass
class ContactSolveResult:
    """接触付き非線形解析の結果.

    Attributes:
        u: (ndof,) 最終変位ベクトル
        converged: 収束したかどうか
        n_load_steps: 荷重増分ステップ数
        total_newton_iterations: 全ステップの合計 Newton 反復回数
        total_outer_iterations: 全ステップの合計 Outer 反復回数
        n_active_final: 最終的な ACTIVE ペア数
        load_history: 各ステップの荷重係数
        displacement_history: 各ステップの変位
        contact_force_history: 各ステップの接触力ノルム
    """

    u: np.ndarray
    converged: bool
    n_load_steps: int
    total_newton_iterations: int
    total_outer_iterations: int
    n_active_final: int
    load_history: list[float] = field(default_factory=list)
    displacement_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)


def _deformed_coords(
    node_coords_ref: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """参照座標 + 変位から変形座標を計算する.

    Args:
        node_coords_ref: (n_nodes, 3) 参照節点座標
        u: (ndof_total,) 変位ベクトル
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        coords_def: (n_nodes, 3) 変形後座標
    """
    n_nodes = node_coords_ref.shape[0]
    coords_def = node_coords_ref.copy()
    for i in range(n_nodes):
        coords_def[i, 0] += u[i * ndof_per_node + 0]
        coords_def[i, 1] += u[i * ndof_per_node + 1]
        coords_def[i, 2] += u[i * ndof_per_node + 2]
    return coords_def


def _update_gaps_fixed_st(
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> None:
    """Inner loop 中に gap のみ更新する（s, t, normal は固定）.

    Outer loop で確定した最近接パラメータ (s, t) を保持したまま、
    現在の変位 u に基づいて gap を再計算する。
    これにより、Inner NR の接触力 f_c が u に依存し、
    接触接線剛性 K_c との整合性が保たれる。

    Args:
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        u: (ndof_total,) 現在の変位ベクトル
        ndof_per_node: 1節点あたりの DOF 数
    """
    coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)

    for pair in manager.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        s = pair.state.s
        t = pair.state.t

        xA0 = coords_def[pair.nodes_a[0]]
        xA1 = coords_def[pair.nodes_a[1]]
        xB0 = coords_def[pair.nodes_b[0]]
        xB1 = coords_def[pair.nodes_b[1]]

        PA = (1.0 - s) * xA0 + s * xA1
        PB = (1.0 - t) * xB0 + t * xB1
        dist = float(np.linalg.norm(PA - PB))

        pair.state.gap = dist - pair.radius_a - pair.radius_b


def newton_raphson_with_contact(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    n_load_steps: int = 10,
    max_iter: int = 30,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_energy: float = 1e-10,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
) -> ContactSolveResult:
    """接触付き Newton-Raphson（Outer/Inner 分離）.

    各荷重ステップで:
    1. Outer loop: 候補検出 + 幾何更新 + k_pen 初期化
    2. Inner loop: NR 反復（最近接固定、接触力/剛性を追加）
    3. Outer 収束判定: |Δs|, |Δt| < tol_geometry
    4. AL 乗数更新

    Args:
        f_ext_total: (ndof,) 最終外荷重
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ（k_pen_scale, g_on, g_off 等設定済み）
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径（スカラー or 配列）
        n_load_steps: 荷重増分数
        max_iter: Inner Newton の最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_energy: エネルギーノルム収束判定
        show_progress: 進捗表示
        u0: 初期変位
        ndof_per_node: 1節点あたりの DOF 数
        broadphase_margin: broadphase 探索マージン
        broadphase_cell_size: broadphase セルサイズ

    Returns:
        ContactSolveResult
    """
    import scipy.sparse.linalg as spla

    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    n_outer_max = manager.config.n_outer_max
    tol_geometry = manager.config.tol_geometry

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    total_newton = 0
    total_outer = 0

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    if f_ext_ref_norm < 1e-30:
        f_ext_ref_norm = 1.0

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        f_ext = lam * f_ext_total

        step_converged = False

        for outer in range(n_outer_max):
            total_outer += 1

            # --- Outer: 変形座標を計算し、候補検出 + 幾何更新 ---
            coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)

            manager.detect_candidates(
                coords_def,
                connectivity,
                radii,
                margin=broadphase_margin,
                cell_size=broadphase_cell_size,
            )
            manager.update_geometry(coords_def)

            # k_pen 未設定のペアを初期化
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE and pair.state.k_pen <= 0.0:
                    # EA/L ベースの推定（簡易版: k_pen_scale をそのまま使用）
                    initialize_penalty_stiffness(
                        pair,
                        k_pen=manager.config.k_pen_scale,
                        k_t_ratio=manager.config.k_t_ratio,
                    )

            # 前回の (s, t) を保存（Outer 収束判定用）
            prev_st = []
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    prev_st.append((pair.state.s, pair.state.t))
                else:
                    prev_st.append(None)

            # --- Inner: NR 反復（最近接点固定）---
            inner_converged = False
            energy_ref = None

            for it in range(max_iter):
                total_newton += 1

                # gap 更新（s, t, normal 固定で変位に基づく gap 再計算）
                _update_gaps_fixed_st(manager, node_coords_ref, u, ndof_per_node)

                # 構造内力
                f_int = assemble_internal_force(u)

                # 接触内力
                f_c = compute_contact_force(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                )

                # 残差
                residual = f_ext - f_int - f_c
                residual[fixed_dofs] = 0.0
                res_norm = float(np.linalg.norm(residual))

                # 力ノルム判定
                if res_norm / f_ext_ref_norm < tol_force:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, ||R||/||f|| = {res_norm / f_ext_ref_norm:.3e} "
                            f"(force converged, {manager.n_active} active)"
                        )
                    break

                # 接線剛性（構造 + 接触）
                K_T = assemble_tangent(u)
                K_c = compute_contact_stiffness(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                )
                K_total = K_T + K_c

                # BC 適用
                K_bc = K_total.tolil()
                r_bc = residual.copy()
                for dof in fixed_dofs:
                    K_bc[dof, :] = 0.0
                    K_bc[:, dof] = 0.0
                    K_bc[dof, dof] = 1.0
                    r_bc[dof] = 0.0

                du = spla.spsolve(K_bc.tocsr(), r_bc)

                # エネルギーノルム
                energy = abs(float(np.dot(du, residual)))
                if energy_ref is None:
                    energy_ref = energy if energy > 1e-30 else 1.0

                u_norm = float(np.linalg.norm(u))
                du_norm = float(np.linalg.norm(du))

                if show_progress and it % 5 == 0:
                    print(
                        f"  Step {step}/{n_load_steps}, outer {outer}, iter {it}, "
                        f"||R||/||f|| = {res_norm / f_ext_ref_norm:.3e}, "
                        f"||du||/||u|| = {du_norm / max(u_norm, 1e-30):.3e}, "
                        f"active={manager.n_active}"
                    )

                u += du

                # 変位ノルム判定
                if u_norm > 1e-30 and du_norm / u_norm < tol_disp:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, ||du||/||u|| = {du_norm / u_norm:.3e} "
                            f"(disp converged)"
                        )
                    break

                if energy / energy_ref < tol_energy:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, energy = {energy / energy_ref:.3e} "
                            f"(energy converged)"
                        )
                    break

            if not inner_converged:
                if show_progress:
                    print(
                        f"  WARNING: Step {step}, outer {outer} "
                        f"did not converge in {max_iter} iterations."
                    )
                return ContactSolveResult(
                    u=u,
                    converged=False,
                    n_load_steps=step,
                    total_newton_iterations=total_newton,
                    total_outer_iterations=total_outer,
                    n_active_final=manager.n_active,
                    load_history=load_history,
                    displacement_history=disp_history,
                    contact_force_history=contact_force_history,
                )

            # --- Outer 収束判定 ---
            # 幾何更新して (s,t) の変化を検査
            coords_def_new = _deformed_coords(node_coords_ref, u, ndof_per_node)
            manager.update_geometry(coords_def_new)

            max_ds = 0.0
            max_dt = 0.0
            idx = 0
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE and prev_st[idx] is not None:
                    s_old, t_old = prev_st[idx]
                    max_ds = max(max_ds, abs(pair.state.s - s_old))
                    max_dt = max(max_dt, abs(pair.state.t - t_old))
                idx += 1

            # AL 乗数更新
            for pair in manager.pairs:
                update_al_multiplier(pair)

            if show_progress:
                print(
                    f"  Step {step}, outer {outer}: "
                    f"max|Δs|={max_ds:.3e}, max|Δt|={max_dt:.3e}, "
                    f"active={manager.n_active}"
                )

            if max_ds < tol_geometry and max_dt < tol_geometry:
                step_converged = True
                break

        if not step_converged:
            # Outer ループ上限到達でも Inner は収束している → 受容
            if show_progress:
                print(
                    f"  Step {step}: outer loop reached {n_outer_max} "
                    f"(max|Δs|={max_ds:.3e}, max|Δt|={max_dt:.3e}). Accepting."
                )
            step_converged = True

        # 接触力ノルム記録
        f_c_final = compute_contact_force(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
        )
        fc_norm = float(np.linalg.norm(f_c_final))

        load_history.append(lam)
        disp_history.append(u.copy())
        contact_force_history.append(fc_norm)

    return ContactSolveResult(
        u=u,
        converged=True,
        n_load_steps=n_load_steps,
        total_newton_iterations=total_newton,
        total_outer_iterations=total_outer,
        n_active_final=manager.n_active,
        load_history=load_history,
        displacement_history=disp_history,
        contact_force_history=contact_force_history,
    )
