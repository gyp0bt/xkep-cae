"""数値試験フレームワーク — 周波数応答試験.

片端保持（カンチレバー）のもう片端への変位付加 or 加速度付加による
周波数応答関数（FRF）を計算する。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.numerical_tests._backend import (
    _beam2d_lumped_mass_local,
    _beam2d_mass_global,
    _ke_func_factory,
    beam3d_length_and_direction,
    beam3d_lumped_mass_local,
    beam3d_mass_global,
)
from xkep_cae.numerical_tests.core import (
    FrequencyResponseConfig,
    FrequencyResponseResult,
    _build_section_props,
    _generate_beam_mesh_2d,
    _generate_beam_mesh_3d,
)
from xkep_cae.numerical_tests.runner import _assemble_2d, _assemble_3d

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
# 質量行列 — backend 経由ラッパー（後方互換エイリアス）
# ---------------------------------------------------------------------------
def _local_beam2d_lumped_mass(rho: float, A: float, L: float) -> np.ndarray:
    """2D梁の集中質量行列（ローカル）."""
    return _beam2d_lumped_mass_local(rho, A, L)


def _local_beam3d_lumped_mass(rho: float, A: float, Iy: float, Iz: float, L: float) -> np.ndarray:
    """3D梁の集中質量行列（ローカル）."""
    return beam3d_lumped_mass_local(rho, A, Iy, Iz, L)


def _global_beam2d_mass(coords: np.ndarray, rho: float, A: float) -> np.ndarray:
    """2D梁の全体座標系の整合質量行列."""
    return _beam2d_mass_global(coords, rho, A)


def _global_beam3d_mass(
    coords: np.ndarray, rho: float, A: float, Iy: float, Iz: float
) -> np.ndarray:
    """3D梁の全体座標系の整合質量行列."""
    return beam3d_mass_global(coords, rho, A, Iy, Iz)


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
        Me = _global_beam2d_mass(coords, rho, A)
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
        Me = _global_beam3d_mass(coords, rho, A, Iy, Iz)
        edofs = np.empty(12, dtype=int)
        for i, n in enumerate([n1, n2]):
            for d in range(6):
                edofs[6 * i + d] = 6 * n + d
        for ii in range(12):
            for jj in range(12):
                M[edofs[ii], edofs[jj]] += Me[ii, jj]
    return M


def _assemble_lumped_mass_2d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rho: float,
    A: float,
) -> np.ndarray:
    """2D梁の全体集中質量行列をアセンブルする."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    M = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        L = np.sqrt(dx**2 + dy**2)
        Me = _local_beam2d_lumped_mass(rho, A, L)
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
    """3D梁の全体集中質量行列をアセンブルする."""
    n_nodes = len(nodes)
    ndof = 6 * n_nodes
    M = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        L, _ = beam3d_length_and_direction(coords)
        Me = _local_beam3d_lumped_mass(rho, A, Iy, Iz, L)
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
def _run_frequency_response(
    cfg: FrequencyResponseConfig,
) -> FrequencyResponseResult:
    """周波数応答試験を実行する."""
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    is_3d = cfg.beam_type in ("timo3d", "cosserat")
    dof_per_node = 6 if is_3d else 3

    if is_3d:
        nodes, conn = _generate_beam_mesh_3d(cfg.n_elems, cfg.length)
    else:
        nodes, conn = _generate_beam_mesh_2d(cfg.n_elems, cfg.length)

    n_nodes = len(nodes)
    n_elems = cfg.n_elems
    ndof = dof_per_node * n_nodes

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
    ke_func = _ke_func_factory(_dummy_cfg, sec)

    if is_3d:
        K, _ = _assemble_3d(nodes, conn, ke_func)
    else:
        K, _ = _assemble_2d(nodes, conn, ke_func)

    if is_3d:
        M = _assemble_mass_3d(nodes, conn, cfg.rho, sec["A"], sec["Iy"], sec["Iz"])
    else:
        M = _assemble_mass_2d(nodes, conn, cfg.rho, sec["A"])

    C = cfg.damping_alpha * M + cfg.damping_beta * K

    fixed_dofs = set(range(dof_per_node))
    all_dofs = set(range(ndof))
    free_dofs = sorted(all_dofs - fixed_dofs)
    free_dofs_arr = np.array(free_dofs, dtype=int)

    K_ff = K[np.ix_(free_dofs_arr, free_dofs_arr)]
    M_ff = M[np.ix_(free_dofs_arr, free_dofs_arr)]
    C_ff = C[np.ix_(free_dofs_arr, free_dofs_arr)]

    tip_node = n_elems
    excite_local = _dof_name_to_local_index(cfg.excitation_dof, cfg.beam_type)
    excite_global = dof_per_node * tip_node + excite_local

    if cfg.response_dof is not None:
        resp_local = _dof_name_to_local_index(cfg.response_dof, cfg.beam_type)
        resp_global = dof_per_node * tip_node + resp_local
    else:
        resp_global = excite_global

    excite_idx_red = free_dofs.index(excite_global)
    resp_idx_red = free_dofs.index(resp_global)

    freqs = np.linspace(cfg.freq_min, cfg.freq_max, cfg.n_freq)
    omegas = 2.0 * np.pi * freqs
    n_free = len(free_dofs)

    H = np.zeros(cfg.n_freq, dtype=complex)

    if cfg.excitation_type == "displacement":
        remaining = [i for i in range(n_free) if i != excite_idx_red]
        remaining_arr = np.array(remaining, dtype=int)

        for i_f, omega in enumerate(omegas):
            K_dyn = K_ff - omega**2 * M_ff + 1j * omega * C_ff
            K_rr = K_dyn[np.ix_(remaining_arr, remaining_arr)]
            K_rp = K_dyn[remaining_arr, excite_idx_red]
            rhs = -K_rp * 1.0
            u_r = np.linalg.solve(K_rr, rhs)
            u_full = np.zeros(n_free, dtype=complex)
            u_full[excite_idx_red] = 1.0
            u_full[remaining_arr] = u_r
            H[i_f] = u_full[resp_idx_red]

    elif cfg.excitation_type == "acceleration":
        M_col = M_ff[:, excite_idx_red].copy()
        for i_f, omega in enumerate(omegas):
            K_dyn = K_ff - omega**2 * M_ff + 1j * omega * C_ff
            f_excite = M_col
            u_resp = np.linalg.solve(K_dyn, f_excite)
            H[i_f] = u_resp[resp_idx_red]

    magnitude = np.abs(H)
    phase_deg = np.degrees(np.angle(H))
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
    """FRFの振幅ピークから固有振動数を推定する."""
    if len(magnitude) < 3:
        return np.array([])

    peaks = []
    threshold = prominence_ratio * np.max(magnitude)

    for i in range(1, len(magnitude) - 1):
        if magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1]:
            if magnitude[i] > threshold:
                peaks.append(freqs[i])

    return np.array(peaks)
