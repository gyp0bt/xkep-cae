"""数値試験フレームワーク — 周波数応答試験.

片端保持（カンチレバー）のもう片端への変位付加 or 加速度付加による
周波数応答関数（FRF）を計算する。

整合質量行列（consistent mass matrix）を内部で構築し、
動的剛性行列 K_dyn = K - ω²M + iωC を周波数スイープで解く。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.numerical_tests.core import (
    FrequencyResponseConfig,
    FrequencyResponseResult,
    _build_section_props,
    generate_beam_mesh_2d,
    generate_beam_mesh_3d,
)
from xkep_cae.numerical_tests.runner import _assemble_2d, _assemble_3d, _get_ke_func

# ---------------------------------------------------------------------------
# DOF名 → インデックスマッピング
# ---------------------------------------------------------------------------
_DOF_MAP_2D = {"ux": 0, "uy": 1, "theta_z": 2}
_DOF_MAP_3D = {"ux": 0, "uy": 1, "uz": 2, "theta_x": 3, "theta_y": 4, "theta_z": 5}


def _dof_name_to_local_index(dof_name: str, beam_type: str) -> int:
    """DOF名をノード内ローカルインデックスに変換する."""
    if beam_type == "timo3d":
        if dof_name not in _DOF_MAP_3D:
            raise ValueError(f"3D梁のDOF名は {list(_DOF_MAP_3D.keys())}: {dof_name}")
        return _DOF_MAP_3D[dof_name]
    else:
        if dof_name not in _DOF_MAP_2D:
            raise ValueError(f"2D梁のDOF名は {list(_DOF_MAP_2D.keys())}: {dof_name}")
        return _DOF_MAP_2D[dof_name]


# ---------------------------------------------------------------------------
# 整合質量行列（Consistent Mass Matrix）
# ---------------------------------------------------------------------------
def _beam2d_consistent_mass_local(
    rho: float,
    A: float,
    L: float,
) -> np.ndarray:
    """2D梁の局所整合質量行列 (6x6).

    DOF順: [u1, v1, θz1, u2, v2, θz2]
    Euler-Bernoulli型の質量行列（Timoshenko梁にも実用的に適用可能）。

    Args:
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        L: 要素長 [m]

    Returns:
        Me: (6, 6) 局所質量行列
    """
    m = rho * A * L
    Me = np.zeros((6, 6), dtype=float)

    # 軸方向 (u1, u2) → DOF 0, 3
    Me[0, 0] = m / 3.0
    Me[0, 3] = m / 6.0
    Me[3, 0] = m / 6.0
    Me[3, 3] = m / 3.0

    # 横方向曲げ (v1, θz1, v2, θz2) → DOF 1, 2, 4, 5
    coeff = m / 420.0
    bend_idx = [1, 2, 4, 5]
    M_bend = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    for i_loc, i_glob in enumerate(bend_idx):
        for j_loc, j_glob in enumerate(bend_idx):
            Me[i_glob, j_glob] = M_bend[i_loc, j_loc]

    return Me


def _beam3d_consistent_mass_local(
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
    L: float,
) -> np.ndarray:
    """3D梁の局所整合質量行列 (12x12).

    DOF順: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Args:
        rho: 密度
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        L: 要素長

    Returns:
        Me: (12, 12) 局所質量行列
    """
    m = rho * A * L
    Me = np.zeros((12, 12), dtype=float)

    # 軸方向 (u1, u2) → DOF 0, 6
    Me[0, 0] = m / 3.0
    Me[0, 6] = m / 6.0
    Me[6, 0] = m / 6.0
    Me[6, 6] = m / 3.0

    # ねじり (θx1, θx2) → DOF 3, 9
    Ip = Iy + Iz  # 極慣性モーメント
    m_torsion = rho * Ip * L
    Me[3, 3] = m_torsion / 3.0
    Me[3, 9] = m_torsion / 6.0
    Me[9, 3] = m_torsion / 6.0
    Me[9, 9] = m_torsion / 3.0

    # xy面内曲げ (v1, θz1, v2, θz2) → DOF 1, 5, 7, 11
    coeff = m / 420.0
    M_xy = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    xy_idx = [1, 5, 7, 11]
    for i_loc, i_glob in enumerate(xy_idx):
        for j_loc, j_glob in enumerate(xy_idx):
            Me[i_glob, j_glob] = M_xy[i_loc, j_loc]

    # xz面内曲げ (w1, θy1, w2, θy2) → DOF 2, 4, 8, 10
    # dw/dx = -θy なので符号反転: signs = [+1, -1, +1, -1]
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    M_xz_base = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    M_xz = M_xz_base * np.outer(signs, signs)
    xz_idx = [2, 4, 8, 10]
    for i_loc, i_glob in enumerate(xz_idx):
        for j_loc, j_glob in enumerate(xz_idx):
            Me[i_glob, j_glob] = M_xz[i_loc, j_loc]

    return Me


# ---------------------------------------------------------------------------
# 質量行列の全体座標系への変換
# ---------------------------------------------------------------------------
def _beam2d_mass_global(
    coords: np.ndarray,
    rho: float,
    A: float,
) -> np.ndarray:
    """2D梁の全体座標系の質量行列."""
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    L = np.sqrt(dx**2 + dy**2)
    c, s = dx / L, dy / L

    Me_local = _beam2d_consistent_mass_local(rho, A, L)

    # 座標変換行列（2D梁: 剛性と同じ変換）
    T = np.zeros((6, 6), dtype=float)
    T[0, 0] = c
    T[0, 1] = s
    T[1, 0] = -s
    T[1, 1] = c
    T[2, 2] = 1.0
    T[3, 3] = c
    T[3, 4] = s
    T[4, 3] = -s
    T[4, 4] = c
    T[5, 5] = 1.0

    return T.T @ Me_local @ T


def _beam3d_mass_global(
    coords: np.ndarray,
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
) -> np.ndarray:
    """3D梁の全体座標系の質量行列."""
    from xkep_cae.elements.beam_timo3d import (
        _beam3d_length_and_direction,
        _build_local_axes,
        _transformation_matrix_3d,
    )

    L, e_x = _beam3d_length_and_direction(coords)
    R = _build_local_axes(e_x)
    Me_local = _beam3d_consistent_mass_local(rho, A, Iy, Iz, L)
    T = _transformation_matrix_3d(R)

    return T.T @ Me_local @ T


# ---------------------------------------------------------------------------
# 全体質量行列アセンブリ
# ---------------------------------------------------------------------------
def _assemble_mass_2d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rho: float,
    A: float,
) -> np.ndarray:
    """2D梁の全体質量行列をアセンブルする."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    M = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        Me = _beam2d_mass_global(coords, rho, A)
        edofs = np.array(
            [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2], dtype=int
        )
        for ii in range(6):
            for jj in range(6):
                M[edofs[ii], edofs[jj]] += Me[ii, jj]
    return M


def _assemble_mass_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
) -> np.ndarray:
    """3D梁の全体質量行列をアセンブルする."""
    n_nodes = len(nodes)
    ndof = 6 * n_nodes
    M = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        Me = _beam3d_mass_global(coords, rho, A, Iy, Iz)
        edofs = np.empty(12, dtype=int)
        for i, n in enumerate([n1, n2]):
            for d in range(6):
                edofs[6 * i + d] = 6 * n + d
        for ii in range(12):
            for jj in range(12):
                M[edofs[ii], edofs[jj]] += Me[ii, jj]
    return M


# ---------------------------------------------------------------------------
# 集中質量行列（Lumped Mass Matrix）— HRZ法
# ---------------------------------------------------------------------------
def _beam2d_lumped_mass_local(
    rho: float,
    A: float,
    L: float,
) -> np.ndarray:
    """2D梁の局所集中質量行列 (6x6, 対角).

    HRZ (Hinton-Rock-Zienkiewicz) 法による集中化。
    整合質量行列の対角成分を取り、並進方向で全質量が保存されるよう
    スケーリングする。回転DOFにも小さな慣性を付与し非特異性を確保。

    DOF順: [u1, v1, θz1, u2, v2, θz2]

    HRZ結果:
        並進: m/2 （各節点・各方向）
        回転: m·L²/78 （各節点）

    Args:
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        L: 要素長 [m]

    Returns:
        Me: (6, 6) 対角集中質量行列
    """
    m = rho * A * L
    # HRZ法: 並進 m/2, 回転 m*L²/78
    diag = np.array([m / 2.0, m / 2.0, m * L**2 / 78.0, m / 2.0, m / 2.0, m * L**2 / 78.0])
    return np.diag(diag)


def _beam3d_lumped_mass_local(
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
    L: float,
) -> np.ndarray:
    """3D梁の局所集中質量行列 (12x12, 対角).

    HRZ法による集中化。

    DOF順: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    HRZ結果:
        並進: m/2 （各節点・各方向）
        ねじり: ρ·Ip·L/2 （各節点）
        曲げ回転: m·L²/78 （各節点・各方向）

    Args:
        rho: 密度
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        L: 要素長

    Returns:
        Me: (12, 12) 対角集中質量行列
    """
    m = rho * A * L
    Ip = Iy + Iz
    m_torsion = rho * Ip * L
    rot_inertia = m * L**2 / 78.0

    # [u, v, w, θx, θy, θz] × 2ノード
    diag = np.array(
        [
            m / 2.0,
            m / 2.0,
            m / 2.0,
            m_torsion / 2.0,
            rot_inertia,
            rot_inertia,
            m / 2.0,
            m / 2.0,
            m / 2.0,
            m_torsion / 2.0,
            rot_inertia,
            rot_inertia,
        ]
    )
    return np.diag(diag)


def _assemble_lumped_mass_2d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rho: float,
    A: float,
) -> np.ndarray:
    """2D梁の全体集中質量行列をアセンブルする.

    集中質量は対角行列なので座標変換不要（回転不変）。
    """
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    M = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        L = np.sqrt(dx**2 + dy**2)
        Me = _beam2d_lumped_mass_local(rho, A, L)
        edofs = np.array(
            [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2],
            dtype=int,
        )
        for ii in range(6):
            M[edofs[ii], edofs[ii]] += Me[ii, ii]
    return M


def _assemble_lumped_mass_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
) -> np.ndarray:
    """3D梁の全体集中質量行列をアセンブルする.

    集中質量は対角行列なので座標変換不要（回転不変）。
    """
    from xkep_cae.elements.beam_timo3d import _beam3d_length_and_direction

    n_nodes = len(nodes)
    ndof = 6 * n_nodes
    M = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        L, _ = _beam3d_length_and_direction(coords)
        Me = _beam3d_lumped_mass_local(rho, A, Iy, Iz, L)
        edofs = np.empty(12, dtype=int)
        for i, n in enumerate([n1, n2]):
            for d in range(6):
                edofs[6 * i + d] = 6 * n + d
        for ii in range(12):
            M[edofs[ii], edofs[ii]] += Me[ii, ii]
    return M


# ---------------------------------------------------------------------------
# 周波数応答試験ランナー
# ---------------------------------------------------------------------------
def run_frequency_response(
    cfg: FrequencyResponseConfig,
) -> FrequencyResponseResult:
    """周波数応答試験を実行する.

    片端保持（カンチレバー）のもう片端への変位付加 or 加速度付加。

    処理の流れ:
    1. メッシュ生成
    2. 全体剛性行列 K と全体質量行列 M をアセンブル
    3. Rayleigh減衰行列 C = αM + βK を構築
    4. 各周波数で動的剛性行列 K_dyn = K - ω²M + iωC を構築
    5. 励起タイプに応じて FRF を計算

    displacement 励起:
        自由端に単位変位を付加し、応答変位を計算。
        動的剛性行列を分割して縮約系を解く。

    acceleration 励起:
        自由端に単位加速度相当の慣性力を付加し、応答変位を計算。
        F = M の該当列 × ω² × 1.0（単位加速度）

    Returns:
        FrequencyResponseResult
    """
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    is_3d = cfg.beam_type in ("timo3d", "cosserat")
    dof_per_node = 6 if is_3d else 3

    # メッシュ
    if is_3d:
        nodes, conn = generate_beam_mesh_3d(cfg.n_elems, cfg.length)
    else:
        nodes, conn = generate_beam_mesh_2d(cfg.n_elems, cfg.length)

    n_nodes = len(nodes)
    n_elems = cfg.n_elems
    ndof = dof_per_node * n_nodes

    # NumericalTestConfig互換のダミーcfgを作って ke_func を取得
    _dummy_cfg = type(
        "Cfg",
        (),
        {
            "beam_type": cfg.beam_type,
            "E": cfg.E,
            "nu": cfg.nu,
            "G": cfg.G,
        },
    )()
    ke_func = _get_ke_func(_dummy_cfg, sec)

    # 剛性行列
    if is_3d:
        K, _ = _assemble_3d(nodes, conn, ke_func)
    else:
        K, _ = _assemble_2d(nodes, conn, ke_func)

    # 質量行列
    if is_3d:
        M = _assemble_mass_3d(nodes, conn, cfg.rho, sec["A"], sec["Iy"], sec["Iz"])
    else:
        M = _assemble_mass_2d(nodes, conn, cfg.rho, sec["A"])

    # Rayleigh減衰
    C = cfg.damping_alpha * M + cfg.damping_beta * K

    # 固定端（節点0の全DOF）
    fixed_dofs = set(range(dof_per_node))
    all_dofs = set(range(ndof))
    free_dofs = sorted(all_dofs - fixed_dofs)
    free_dofs_arr = np.array(free_dofs, dtype=int)

    # 縮約行列（固定DOFを除去）
    K_ff = K[np.ix_(free_dofs_arr, free_dofs_arr)]
    M_ff = M[np.ix_(free_dofs_arr, free_dofs_arr)]
    C_ff = C[np.ix_(free_dofs_arr, free_dofs_arr)]

    # 励起DOFの特定（自由端 = 節点n_elems）
    tip_node = n_elems
    excite_local = _dof_name_to_local_index(cfg.excitation_dof, cfg.beam_type)
    excite_global = dof_per_node * tip_node + excite_local

    # 応答DOF
    if cfg.response_dof is not None:
        resp_local = _dof_name_to_local_index(cfg.response_dof, cfg.beam_type)
        resp_global = dof_per_node * tip_node + resp_local
    else:
        resp_global = excite_global  # デフォルト: 励起点と同じ

    # 縮約系内でのインデックス
    excite_idx_red = free_dofs.index(excite_global)
    resp_idx_red = free_dofs.index(resp_global)

    # 周波数スイープ
    freqs = np.linspace(cfg.freq_min, cfg.freq_max, cfg.n_freq)
    omegas = 2.0 * np.pi * freqs
    n_free = len(free_dofs)

    H = np.zeros(cfg.n_freq, dtype=complex)

    if cfg.excitation_type == "displacement":
        # 変位励起: 自由端の特定DOFに単位変位 u_p = 1 を付加
        # 縮約系から励起DOFを除外して解く
        #
        # K_dyn * u = f
        # 分割: prescribed DOF (p) と残りの free DOF (r)
        # K_rr * u_r = -K_rp * u_p
        # H(ω) = u_resp / u_excite = u_resp / 1.0

        remaining = [i for i in range(n_free) if i != excite_idx_red]
        remaining_arr = np.array(remaining, dtype=int)

        for i_f, omega in enumerate(omegas):
            K_dyn = K_ff - omega**2 * M_ff + 1j * omega * C_ff

            K_rr = K_dyn[np.ix_(remaining_arr, remaining_arr)]
            K_rp = K_dyn[remaining_arr, excite_idx_red]

            # u_p = 1.0
            rhs = -K_rp * 1.0

            u_r = np.linalg.solve(K_rr, rhs)

            # 全自由DOFの変位を再構成
            u_full = np.zeros(n_free, dtype=complex)
            u_full[excite_idx_red] = 1.0
            u_full[remaining_arr] = u_r

            H[i_f] = u_full[resp_idx_red]

    elif cfg.excitation_type == "acceleration":
        # 加速度励起: 自由端に単位加速度を付加
        # F = M の該当列（自由端DOF に対応）の抽出
        # 動的方程式: K_dyn * u = F_inertia
        # F_inertia = M_ff[:, excite_idx] (= 質量行列の列)
        # H(ω) = u_resp / a_input（単位加速度あたりの変位応答）

        M_col = M_ff[:, excite_idx_red].copy()

        for i_f, omega in enumerate(omegas):
            K_dyn = K_ff - omega**2 * M_ff + 1j * omega * C_ff
            f_excite = M_col  # 単位加速度の慣性力
            u_resp = np.linalg.solve(K_dyn, f_excite)
            H[i_f] = u_resp[resp_idx_red]

    magnitude = np.abs(H)
    phase_deg = np.degrees(np.angle(H))

    # ピーク検出（固有振動数の推定）
    nat_freqs = _detect_peaks(freqs, magnitude)

    return FrequencyResponseResult(
        config=cfg,
        frequencies=freqs,
        transfer_function=H,
        magnitude=magnitude,
        phase_deg=phase_deg,
        natural_frequencies=nat_freqs,
        node_coords=nodes,
    )


# ---------------------------------------------------------------------------
# ピーク検出
# ---------------------------------------------------------------------------
def _detect_peaks(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    prominence_ratio: float = 0.1,
) -> np.ndarray:
    """FRFの振幅ピークから固有振動数を推定する.

    scipy.signal を使わない簡易実装。
    """
    if len(magnitude) < 3:
        return np.array([])

    peaks = []
    threshold = prominence_ratio * np.max(magnitude)

    for i in range(1, len(magnitude) - 1):
        if magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1]:
            if magnitude[i] > threshold:
                peaks.append(freqs[i])

    return np.array(peaks)
