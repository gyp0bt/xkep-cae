"""数値試験バックエンド — 直接実装モジュール.

BackendRegistry パターンを廃止し、要素剛性・BC・ソルバー・質量行列・動的ソルバー
を純粋関数として直接提供する。

status-208: BackendRegistry 完全廃止（O2 条例違反解消）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from xkep_cae.elements._beam_cr import (
    _beam3d_length_and_direction,
    timo_beam3d_ke_global,
    timo_beam3d_lumped_mass_local,
    timo_beam3d_mass_global,
)
from xkep_cae.numerical_tests.core import NumericalTestConfig


# ---------------------------------------------------------------------------
# 2D 梁要素剛性行列（EB / Timoshenko）
# ---------------------------------------------------------------------------
def _eb2d_ke(coords: np.ndarray, E: float, I: float, A: float) -> np.ndarray:  # noqa: E741
    """2D Euler-Bernoulli 梁の剛性行列 (6x6).

    DOFs: [ux1, uy1, θz1, ux2, uy2, θz2]
    """
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    L = np.sqrt(dx**2 + dy**2)

    EA_L = E * A / L
    EI_L3 = E * I / L**3
    EI_L2 = E * I / L**2
    EI_L = E * I / L

    Ke = np.array(
        [
            [EA_L, 0, 0, -EA_L, 0, 0],
            [0, 12 * EI_L3, 6 * EI_L2, 0, -12 * EI_L3, 6 * EI_L2],
            [0, 6 * EI_L2, 4 * EI_L, 0, -6 * EI_L2, 2 * EI_L],
            [-EA_L, 0, 0, EA_L, 0, 0],
            [0, -12 * EI_L3, -6 * EI_L2, 0, 12 * EI_L3, -6 * EI_L2],
            [0, 6 * EI_L2, 2 * EI_L, 0, -6 * EI_L2, 4 * EI_L],
        ],
        dtype=float,
    )
    return Ke


def _timo2d_ke(
    coords: np.ndarray,
    E: float,
    G: float,
    I: float,  # noqa: E741
    A: float,
    kappa: float,
) -> np.ndarray:
    """2D Timoshenko 梁の剛性行列 (6x6).

    DOFs: [ux1, uy1, θz1, ux2, uy2, θz2]
    """
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    L = np.sqrt(dx**2 + dy**2)

    Phi = 12.0 * E * I / (kappa * G * A * L**2)
    denom = 1.0 + Phi

    EA_L = E * A / L
    EI_L3 = E * I / L**3
    EI_L2 = E * I / L**2
    EI_L = E * I / L

    Ke = np.array(
        [
            [EA_L, 0, 0, -EA_L, 0, 0],
            [0, 12 * EI_L3 / denom, 6 * EI_L2 / denom, 0, -12 * EI_L3 / denom, 6 * EI_L2 / denom],
            [
                0,
                6 * EI_L2 / denom,
                (4 + Phi) * EI_L / denom,
                0,
                -6 * EI_L2 / denom,
                (2 - Phi) * EI_L / denom,
            ],
            [-EA_L, 0, 0, EA_L, 0, 0],
            [0, -12 * EI_L3 / denom, -6 * EI_L2 / denom, 0, 12 * EI_L3 / denom, -6 * EI_L2 / denom],
            [
                0,
                6 * EI_L2 / denom,
                (2 - Phi) * EI_L / denom,
                0,
                -6 * EI_L2 / denom,
                (4 + Phi) * EI_L / denom,
            ],
        ],
        dtype=float,
    )
    return Ke


# ---------------------------------------------------------------------------
# 要素剛性行列ファクトリ
# ---------------------------------------------------------------------------
def _ke_func_factory(cfg: Any, sec: dict) -> Callable[[np.ndarray], np.ndarray]:
    """beam_type に応じた要素剛性行列関数 coords → Ke を返す."""
    beam_type = cfg.beam_type
    E = cfg.E
    G = cfg.G

    if beam_type == "eb2d":

        def ke_func(coords: np.ndarray) -> np.ndarray:
            return _eb2d_ke(coords, E, sec["I"], sec["A"])

    elif beam_type == "timo2d":

        def ke_func(coords: np.ndarray) -> np.ndarray:
            return _timo2d_ke(coords, E, G, sec["I"], sec["A"], sec["kappa"])

    elif beam_type in ("timo3d", "cosserat"):

        def ke_func(coords: np.ndarray) -> np.ndarray:
            return timo_beam3d_ke_global(
                coords,
                E,
                G,
                sec["A"],
                sec["Iy"],
                sec["Iz"],
                sec["J"],
                sec["kappa_y"],
                sec["kappa_z"],
            )

    else:
        raise ValueError(f"未対応の beam_type: {beam_type}")

    return ke_func


# ---------------------------------------------------------------------------
# Dirichlet 境界条件の適用
# ---------------------------------------------------------------------------
def _apply_dirichlet(
    K: sp.spmatrix,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[sp.spmatrix, np.ndarray]:
    """Dirichlet 境界条件をペナルティ法で適用する.

    固定 DOF の対角項を大きな値に設定し、右辺を 0 にする。
    """
    K = K.tolil()
    f = f.copy()
    big = K.diagonal().max() * 1.0e10
    if big == 0.0:
        big = 1.0e20
    for dof in fixed_dofs:
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = big
        f[dof] = 0.0
    return K.tocsr(), f


# ---------------------------------------------------------------------------
# 線形ソルバー
# ---------------------------------------------------------------------------
def _solve_linear(
    K: sp.spmatrix,
    f: np.ndarray,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict]:
    """疎行列線形連立方程式 K·u = f を解く."""
    u = spla.spsolve(K.tocsc(), f)
    return u, {"solver": "spsolve"}


# ---------------------------------------------------------------------------
# 断面力データクラス
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _NodeForces2DOutput:
    """2D 梁の節点断面力."""

    N: float
    V: float
    M: float


@dataclass(frozen=True)
class _NodeForces3DOutput:
    """3D 梁の節点断面力."""

    N: float
    Vy: float
    Vz: float
    Mx: float
    My: float
    Mz: float


# ---------------------------------------------------------------------------
# 断面力計算
# ---------------------------------------------------------------------------
def _section_force_computer(
    cfg: NumericalTestConfig,
    sec: dict,
    nodes: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
) -> list:
    """各要素の断面力を変位から計算する.

    Returns:
        list of (node1_forces, node2_forces) タプル
    """
    is_3d = cfg.beam_type in ("timo3d", "cosserat")
    ke_func = _ke_func_factory(cfg, sec)

    element_forces = []
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        Ke = ke_func(coords)

        if is_3d:
            edofs = np.empty(12, dtype=int)
            for i, n in enumerate([n1, n2]):
                for d in range(6):
                    edofs[6 * i + d] = 6 * n + d
            u_e = u[edofs]
            f_e = Ke @ u_e
            f1 = _NodeForces3DOutput(
                N=f_e[0], Vy=f_e[1], Vz=f_e[2], Mx=f_e[3], My=f_e[4], Mz=f_e[5]
            )
            f2 = _NodeForces3DOutput(
                N=f_e[6], Vy=f_e[7], Vz=f_e[8], Mx=f_e[9], My=f_e[10], Mz=f_e[11]
            )
        else:
            edofs = np.array(
                [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2],
                dtype=int,
            )
            u_e = u[edofs]
            f_e = Ke @ u_e
            f1 = _NodeForces2DOutput(N=f_e[0], V=f_e[1], M=f_e[2])
            f2 = _NodeForces2DOutput(N=f_e[3], V=f_e[4], M=f_e[5])

        element_forces.append((f1, f2))

    return element_forces


# ---------------------------------------------------------------------------
# 2D 質量行列
# ---------------------------------------------------------------------------
def _beam2d_lumped_mass_local(rho: float, A: float, L: float) -> np.ndarray:
    """2D 梁の局所集中質量行列 (6x6, 対角)."""
    m = rho * A * L
    rot_inertia = m * L**2 / 78.0
    diag = np.array(
        [m / 2.0, m / 2.0, rot_inertia, m / 2.0, m / 2.0, rot_inertia],
        dtype=float,
    )
    return np.diag(diag)


def _beam2d_mass_global(
    coords: np.ndarray,
    rho: float,
    A: float,
) -> np.ndarray:
    """2D 梁の全体座標系の整合質量行列 (6x6).

    DOFs: [ux1, uy1, θz1, ux2, uy2, θz2]
    """
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    L = np.sqrt(dx**2 + dy**2)
    m = rho * A * L

    coeff = m / 420.0
    Me = np.zeros((6, 6), dtype=float)

    # 軸方向
    Me[0, 0] = m / 3.0
    Me[0, 3] = m / 6.0
    Me[3, 0] = m / 6.0
    Me[3, 3] = m / 3.0

    # 横方向（整合）
    M_trans = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    idx = [1, 2, 4, 5]
    for i_loc, i_glob in enumerate(idx):
        for j_loc, j_glob in enumerate(idx):
            Me[i_glob, j_glob] = M_trans[i_loc, j_loc]

    return Me


# ---------------------------------------------------------------------------
# 3D 質量行列ラッパー（元の _beam_cr 関数を直接公開）
# ---------------------------------------------------------------------------
beam3d_lumped_mass_local = timo_beam3d_lumped_mass_local
beam3d_mass_global = timo_beam3d_mass_global
beam3d_length_and_direction = _beam3d_length_and_direction


# ---------------------------------------------------------------------------
# CR 梁アセンブラファクトリ
# ---------------------------------------------------------------------------
def _cr_assembler_factory(
    nodes: np.ndarray,
    conn: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
) -> tuple[Callable, Callable]:
    """CR 梁用の (assemble_internal_force, assemble_tangent) タプルを返す."""
    from xkep_cae.elements._beam_assembly import assemble_cr_beam3d

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
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
            sparse=False,
            analytical_tangent=True,
        )
        return f_int

    def assemble_tangent(u: np.ndarray) -> np.ndarray:
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
            sparse=False,
            analytical_tangent=True,
        )
        return K_T

    return assemble_internal_force, assemble_tangent


def _cosserat_nl_assembler_factory(
    nodes: np.ndarray,
    conn: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
) -> tuple[Callable, Callable]:
    """Cosserat rod 非線形用の (assemble_internal_force, assemble_tangent) を返す.

    線形領域では CR 梁と同じ定式化を使用する。
    """
    return _cr_assembler_factory(nodes, conn, E, G, A, Iy, Iz, J, kappa_y, kappa_z)


# ---------------------------------------------------------------------------
# 過渡応答ソルバー（Newmark-β / HHT-α）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _TransientConfigInput:
    """過渡応答解析の設定."""

    dt: float = 1e-4
    n_steps: int = 100
    beta: float = 0.25
    gamma: float = 0.5
    alpha_hht: float = 0.0
    max_iter: int = 20
    tol_force: float = 1e-6


@dataclass(frozen=True)
class _TransientResultOutput:
    """過渡応答解析の結果."""

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    converged: bool
    failed_step: int | None
    iterations_per_step: tuple[int, ...]


def _transient_solver(
    *,
    M: np.ndarray,
    f_ext: Callable[[float], np.ndarray],
    u0: np.ndarray,
    v0: np.ndarray,
    config: _TransientConfigInput,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    assemble_tangent: Callable[[np.ndarray], np.ndarray],
    C: np.ndarray | None = None,
    fixed_dofs: np.ndarray | None = None,
    show_progress: bool = False,
) -> _TransientResultOutput:
    """Newmark-β / HHT-α 過渡応答ソルバー.

    非線形 Newton-Raphson 反復を各ステップで実行する。
    """
    dt = config.dt
    n_steps = config.n_steps
    beta = config.beta
    gamma = config.gamma
    alpha = config.alpha_hht
    ndof = len(u0)

    if C is None:
        C = np.zeros((ndof, ndof), dtype=float)

    # Newmark 定数
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a7 = dt * gamma

    # 固定 DOF マスク
    free_mask = np.ones(ndof, dtype=bool)
    if fixed_dofs is not None and len(fixed_dofs) > 0:
        free_mask[fixed_dofs] = False
    free_dofs_arr = np.where(free_mask)[0]

    # 初期加速度
    f0 = f_ext(0.0)
    f_int_0 = assemble_internal_force(u0)
    r0 = f0 - f_int_0 - C @ v0
    # M * a0 = r0 (free DOFs only)
    M_ff = M[np.ix_(free_dofs_arr, free_dofs_arr)]
    a0_vec = np.zeros(ndof, dtype=float)
    if len(free_dofs_arr) > 0:
        a0_vec[free_dofs_arr] = np.linalg.solve(M_ff, r0[free_dofs_arr])

    # 履歴保存
    u_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    v_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    a_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    t_hist = np.zeros(n_steps + 1, dtype=float)
    iters_hist: list[int] = []

    u_hist[0] = u0
    v_hist[0] = v0
    a_hist[0] = a0_vec
    t_hist[0] = 0.0

    u_n = u0.copy()
    v_n = v0.copy()
    a_n = a0_vec.copy()
    converged_all = True
    failed_step = None

    for step in range(1, n_steps + 1):
        t_new = step * dt
        f_new = f_ext(t_new)

        # Newmark 予測値
        u_pred = u_n + dt * v_n + 0.5 * dt**2 * (1.0 - 2.0 * beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        u_new = u_pred.copy()

        # Newton-Raphson 反復
        converged_step = False
        for it in range(config.max_iter):
            # 仮の加速度・速度
            a_new = a0 * (u_new - u_n) - a2 * v_n - a3 * a_n
            v_new = v_pred + a7 * a_new

            # 内力
            f_int = assemble_internal_force(u_new)

            # HHT-α 混合
            f_ext_eff = (1.0 + alpha) * f_new - alpha * f_ext(t_new - dt)
            f_int_eff = (1.0 + alpha) * f_int - alpha * assemble_internal_force(u_n)

            # 残差
            R = f_ext_eff - f_int_eff - M @ a_new - C @ v_new

            # 収束判定（free DOFs）
            R_free = R[free_dofs_arr]
            norm_r = np.linalg.norm(R_free)
            ref_norm = max(np.linalg.norm(f_new[free_dofs_arr]), 1.0)
            if norm_r / ref_norm < config.tol_force:
                converged_step = True
                iters_hist.append(it + 1)
                break

            # 有効剛性行列
            K_T = assemble_tangent(u_new)
            K_eff = (1.0 + alpha) * K_T + a0 * M + a1 * C

            # 増分計算 (free DOFs)
            K_ff = K_eff[np.ix_(free_dofs_arr, free_dofs_arr)]
            du_free = np.linalg.solve(K_ff, R_free)
            u_new[free_dofs_arr] += du_free

        if not converged_step:
            converged_all = False
            failed_step = step
            iters_hist.append(config.max_iter)

        # 最終更新
        a_new = a0 * (u_new - u_n) - a2 * v_n - a3 * a_n
        v_new = v_pred + a7 * a_new

        u_hist[step] = u_new
        v_hist[step] = v_new
        a_hist[step] = a_new
        t_hist[step] = t_new

        u_n = u_new.copy()
        v_n = v_new.copy()
        a_n = a_new.copy()

    return _TransientResultOutput(
        time=t_hist,
        displacement=u_hist,
        velocity=v_hist,
        acceleration=a_hist,
        converged=converged_all,
        failed_step=failed_step,
        iterations_per_step=tuple(iters_hist),
    )
