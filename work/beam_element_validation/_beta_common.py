"""Phase β 共通ヘルパ — 1 要素 explicit central difference + Rayleigh damping.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase β 計画に沿い、CR Timoshenko 3D 梁要素 1 つに対して
explicit 中央差分時間積分を直接ドライブ（assembler 経由なし）するヘルパ群:

1. ``solve_explicit_central_diff``: 1 要素 12 DOF の explicit 中央差分ソルバ
   （集中質量 lumped または整合質量 consistent 選択可、Rayleigh 質量比例減衰）
2. ``compute_strain_energy_cr``: corotational 歪エネルギー
   ``SE = (1/2) d_cr^T K_e_local d_cr``（``timo_beam3d_cr_internal_force`` の
   d_cr 計算を再現、大回転でも正確）
3. ``compute_natural_frequencies_fe``: 1 要素 FE 系の固有値
   ``K_aa φ = ω² M_aa φ``（active DOF のみ）

Phase α と同様に ``timo_beam3d_cr_internal_force`` を直接呼び出し、
foundation を最小単位で検証する（status-389 §1.1 の意図）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from work.beam_element_validation._alpha_common import BeamSection
from xkep_cae.elements._beam_cr import (
    _beam3d_length_and_direction,
    _build_local_axes,
    _rodrigues_rotation,
    _rotmat_to_rotvec,
    _rotvec_to_rotmat,
    timo_beam3d_cr_internal_force,
    timo_beam3d_ke_global,
    timo_beam3d_ke_local,
    timo_beam3d_mass_global,
)


@dataclass
class ExplicitDynamicResult:
    """explicit 中央差分時間積分の結果.

    Attributes:
        t: 時間 (n_steps+1,) [s]
        u_history: 変位履歴 (n_steps+1, 12) [mm, rad]
        v_history: 速度履歴 (n_steps+1, 12) [mm/s, rad/s]
        KE_history: 運動エネルギー履歴 (n_steps+1,) [N·mm]
        SE_history: 歪エネルギー履歴 (n_steps+1,) [N·mm]
        L_chord_history: 変形後 chord 長履歴 (n_steps+1,) [mm]
        dt: 時間刻み [s]
        n_steps: ステップ数
    """

    t: np.ndarray
    u_history: np.ndarray
    v_history: np.ndarray
    KE_history: np.ndarray
    SE_history: np.ndarray
    L_chord_history: np.ndarray
    dt: float
    n_steps: int

    @property
    def total_energy(self) -> np.ndarray:
        return self.KE_history + self.SE_history


def compute_strain_energy_cr(
    section: BeamSection,
    u_elem: np.ndarray,
) -> float:
    """CR 定式化の歪エネルギー ``SE = (1/2) d_cr^T K_e_local d_cr``.

    ``timo_beam3d_cr_internal_force`` 内部の d_cr 計算を再現。
    大回転でも正確（線形化 ``0.5 u^T f_int`` よりも厳密）。
    """
    coords = section.coords_2node()
    bk = section.beam_kwargs()

    L_0, e_x_0 = _beam3d_length_and_direction(coords)
    R_0 = _build_local_axes(e_x_0, None)

    x1_def = coords[0] + u_elem[0:3]
    x2_def = coords[1] + u_elem[6:9]
    coords_def = np.array([x1_def, x2_def])
    L_def, e_x_def = _beam3d_length_and_direction(coords_def)

    R_rod = _rodrigues_rotation(e_x_0, e_x_def)
    R_cr = R_0 @ R_rod.T

    R_node1 = _rotvec_to_rotmat(u_elem[3:6])
    R_node2 = _rotvec_to_rotmat(u_elem[9:12])
    R_def1 = R_cr @ R_node1 @ R_0.T
    R_def2 = R_cr @ R_node2 @ R_0.T
    theta_def1 = _rotmat_to_rotvec(R_def1)
    theta_def2 = _rotmat_to_rotvec(R_def2)

    d_cr = np.zeros(12, dtype=float)
    d_cr[3:6] = theta_def1
    d_cr[6] = L_def - L_0
    d_cr[9:12] = theta_def2

    Ke_local = timo_beam3d_ke_local(
        bk["E"],
        bk["G"],
        bk["A"],
        bk["Iy"],
        bk["Iz"],
        bk["J"],
        L_0,
        bk["kappa_y"],
        bk["kappa_z"],
    )
    return 0.5 * float(d_cr @ Ke_local @ d_cr)


def compute_natural_frequencies_fe(
    section: BeamSection,
    rho: float,
    fixed_dofs: list[int],
    use_lumped: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """1 要素 FE 系の固有値 K_aa φ = ω² M_aa φ を解いて (ω 配列, mode shapes) を返す.

    Returns:
        omegas: (n_active,) 固有角振動数 [rad/s], 昇順
        modes: (n_active, n_active) 各列がモード形状（M-orthonormal）
    """
    from scipy.linalg import eigh

    coords = section.coords_2node()
    bk = section.beam_kwargs()
    K = timo_beam3d_ke_global(coords, **bk)
    M = timo_beam3d_mass_global(coords, rho, bk["A"], bk["Iy"], bk["Iz"], lumped=use_lumped)

    a_idx = np.array(sorted(set(range(12)) - set(fixed_dofs)), dtype=int)
    K_aa = K[np.ix_(a_idx, a_idx)]
    M_aa = M[np.ix_(a_idx, a_idx)]

    # Lumped mass may have zeros on diagonal for unused DOFs (none in our case but defensive)
    eigvals, eigvecs = eigh(K_aa, M_aa)
    omegas = np.sqrt(np.maximum(eigvals, 0.0))
    return omegas, eigvecs


def solve_explicit_central_diff(
    section: BeamSection,
    rho: float,
    fixed_dofs: list[int],
    t_total: float,
    n_steps: int,
    initial_velocity: dict[int, float] | None = None,
    initial_displacement: dict[int, float] | None = None,
    prescribed_disp: dict[int, Callable[[float], tuple[float, float]]] | None = None,
    F_ext_func: Callable[[float], np.ndarray] | None = None,
    damping_alpha: float = 0.0,
    use_lumped: bool = True,
    log_every: int = 1,
) -> ExplicitDynamicResult:
    """1 要素 12 DOF explicit 中央差分時間積分 (leap-frog Verlet).

    更新則:
        v_{k+1/2} = v_{k-1/2} + dt · a_k
        u_{k+1}   = u_k + dt · v_{k+1/2}
        a_{k+1}   = M^{-1} · (F_ext - C·v - f_int(u_{k+1}))
                  = -f_int_a / M_diag - α · v_a   (mass-prop damping, F_ext=0)

    Args:
        section: 梁断面.
        rho: 密度 [ton/mm³].
        fixed_dofs: u=0 固定 DOF index list.
        t_total: 解析総時間 [s].
        n_steps: 時間ステップ数. dt = t_total / n_steps.
        initial_velocity: {DOF: v_0 値} 初期速度（active DOF のみ有効）.
        initial_displacement: {DOF: u_0 値} 初期変位（処方ではない、initial 値）.
        prescribed_disp: {DOF: (t -> (u, v)) ramp 関数} 各時刻の処方変位＋速度.
        F_ext_func: t -> (12,) 外力時刻歴（None なら F_ext=0）.
        damping_alpha: 質量比例 Rayleigh 減衰 α (C = α·M).
        use_lumped: True なら集中質量, False なら整合質量（consistent）.
            consistent の場合は M_aa を分解しておき各 step で linear solve.
        log_every: log_every=1 なら毎ステップ履歴を保存.

    Returns:
        ExplicitDynamicResult: 解析結果（時系列＋エネルギー＋幾何）.
    """
    coords = section.coords_2node()
    bk = section.beam_kwargs()

    fixed_set = set(fixed_dofs)
    prescribed_set = set(prescribed_disp.keys()) if prescribed_disp else set()
    if fixed_set & prescribed_set:
        raise ValueError(f"DOF {fixed_set & prescribed_set} は fixed と prescribed 両方に指定")
    a_idx = np.array(sorted(set(range(12)) - fixed_set - prescribed_set), dtype=int)

    M_full = timo_beam3d_mass_global(coords, rho, bk["A"], bk["Iy"], bk["Iz"], lumped=use_lumped)
    M_diag = np.diag(M_full).copy() if use_lumped else None
    M_aa = None if use_lumped else M_full[np.ix_(a_idx, a_idx)]

    dt = t_total / n_steps

    u = np.zeros(12)
    v = np.zeros(12)
    if initial_displacement:
        for d, val in initial_displacement.items():
            u[d] = val
    if initial_velocity:
        for d, val in initial_velocity.items():
            v[d] = val

    # Apply prescribed values at t=0
    if prescribed_disp:
        for d, ramp in prescribed_disp.items():
            u_p, v_p = ramp(0.0)
            u[d] = u_p
            v[d] = v_p

    # Initial f_int and acceleration
    f_int = timo_beam3d_cr_internal_force(coords, u, **bk)
    F0 = F_ext_func(0.0) if F_ext_func else np.zeros(12)
    a = np.zeros(12)
    if use_lumped:
        a[a_idx] = (F0[a_idx] - f_int[a_idx] - damping_alpha * M_diag[a_idx] * v[a_idx]) / M_diag[
            a_idx
        ]
    else:
        r_a = F0[a_idx] - f_int[a_idx] - damping_alpha * (M_aa @ v[a_idx])
        a[a_idx] = np.linalg.solve(M_aa, r_a)

    # Half-step velocity initialization (leap-frog start)
    v_half = v.copy()
    v_half[a_idx] = v[a_idx] + 0.5 * dt * a[a_idx]

    n_log = (n_steps // log_every) + 1
    t_hist = np.zeros(n_log)
    u_hist = np.zeros((n_log, 12))
    v_hist = np.zeros((n_log, 12))
    KE_hist = np.zeros(n_log)
    SE_hist = np.zeros(n_log)
    L_chord_hist = np.zeros(n_log)

    def _log(idx: int, t_val: float) -> None:
        t_hist[idx] = t_val
        u_hist[idx] = u
        v_hist[idx] = v
        # KE = 0.5 v^T M v
        if use_lumped:
            KE_hist[idx] = 0.5 * float(np.sum(M_diag * v * v))
        else:
            KE_hist[idx] = 0.5 * float(v @ (M_full @ v))
        SE_hist[idx] = compute_strain_energy_cr(section, u)
        p1 = coords[0] + u[0:3]
        p2 = coords[1] + u[6:9]
        L_chord_hist[idx] = float(np.linalg.norm(p2 - p1))

    _log(0, 0.0)

    log_idx = 1
    for k in range(n_steps):
        t_k1 = (k + 1) * dt

        # u_{k+1} = u_k + dt · v_{k+1/2}
        u[a_idx] = u[a_idx] + dt * v_half[a_idx]

        # Apply prescribed displacement at t_{k+1}
        if prescribed_disp:
            for d, ramp in prescribed_disp.items():
                u_p, v_p = ramp(t_k1)
                u[d] = u_p
                # v at integer step (for energy)
                v[d] = v_p

        # Compute f_int(u_{k+1}) and F_ext(t_{k+1})
        f_int = timo_beam3d_cr_internal_force(coords, u, **bk)
        F_ext = F_ext_func(t_k1) if F_ext_func else np.zeros(12)

        # a_{k+1} = M^{-1} · (F_ext - f_int - C·v_half_a) on active DOFs
        if use_lumped:
            a[a_idx] = (F_ext[a_idx] - f_int[a_idx]) / M_diag[a_idx] - damping_alpha * v_half[a_idx]
        else:
            r_a = F_ext[a_idx] - f_int[a_idx] - damping_alpha * (M_aa @ v_half[a_idx])
            a[a_idx] = np.linalg.solve(M_aa, r_a)

        # v_{k+3/2} = v_{k+1/2} + dt · a_{k+1}
        v_half[a_idx] = v_half[a_idx] + dt * a[a_idx]

        # Reconstruct v at integer step (for KE diagnostics)
        v[a_idx] = v_half[a_idx] - 0.5 * dt * a[a_idx]

        if log_every == 1 or (k + 1) % log_every == 0:
            if log_idx < n_log:
                _log(log_idx, t_k1)
                log_idx += 1

    # Truncate logs if log_every causes off-by-one
    return ExplicitDynamicResult(
        t=t_hist[:log_idx],
        u_history=u_hist[:log_idx],
        v_history=v_hist[:log_idx],
        KE_history=KE_hist[:log_idx],
        SE_history=SE_hist[:log_idx],
        L_chord_history=L_chord_hist[:log_idx],
        dt=dt,
        n_steps=n_steps,
    )


def measure_period_zero_crossings(
    t: np.ndarray,
    signal: np.ndarray,
    skip_initial: int = 0,
) -> tuple[float, int]:
    """信号のゼロクロッシング間隔から周期を計測.

    Args:
        t: 時刻 (n,)
        signal: 信号 (n,)
        skip_initial: 初期過渡を除去するためスキップする ZC 数

    Returns:
        period_avg: ゼロクロッシング間隔の平均 × 2 [s]（半周期 × 2 = 1 周期）
        n_crossings_used: 使用した ZC 数
    """
    sgn = np.sign(signal)
    sgn[sgn == 0] = 1.0
    cross_idx = np.where(np.diff(sgn) != 0)[0]
    if len(cross_idx) < 2 + skip_initial:
        return 0.0, 0
    cross_idx = cross_idx[skip_initial:]
    # Linear interpolation for accurate zero crossing time
    t_cross = []
    for i in cross_idx:
        s0, s1 = signal[i], signal[i + 1]
        if s1 == s0:
            t_cross.append(t[i])
        else:
            frac = -s0 / (s1 - s0)
            t_cross.append(t[i] + frac * (t[i + 1] - t[i]))
    t_cross_arr = np.array(t_cross)
    if len(t_cross_arr) < 2:
        return 0.0, 0
    half_periods = np.diff(t_cross_arr)
    period_avg = 2.0 * float(np.mean(half_periods))
    return period_avg, len(t_cross_arr)
