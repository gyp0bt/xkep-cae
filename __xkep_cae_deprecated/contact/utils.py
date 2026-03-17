"""ソルバー共通ユーティリティ.

solver_ncp.py / solver_smooth_penalty.py 共通のヘルパー関数。

- deformed_coords: 参照座標 + 変位 → 変形座標
- ncp_line_search: NCP Newton ステップのバックトラッキング line search
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def deformed_coords(
    node_coords_ref: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """参照座標 + 変位から変形座標を計算する."""
    n_nodes = node_coords_ref.shape[0]
    coords_def = node_coords_ref.copy()
    for i in range(n_nodes):
        coords_def[i, 0] += u[i * ndof_per_node + 0]
        coords_def[i, 1] += u[i * ndof_per_node + 1]
        coords_def[i, 2] += u[i * ndof_per_node + 2]
    return coords_def


def ncp_line_search(
    u: np.ndarray,
    du: np.ndarray,
    f_ext: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    res_u_norm: float,
    max_steps: int = 6,
    f_c: np.ndarray | None = None,
    diverge_factor: float = 1000.0,
) -> float:
    """NCP Newton ステップの発散防止 line search.

    diverge_factor で残差増大の許容倍率を設定:
      - 1000.0: 安全弁（Full NR 用、一時的増加を許容）
      - 1.5: 厳密バックトラッキング（Modified NR 用、振動抑制）
    """
    f_c_vec = f_c if f_c is not None else np.zeros_like(f_ext)
    alpha = 1.0
    for _ in range(max_steps):
        u_try = u + alpha * du
        try:
            f_int_try = assemble_internal_force(u_try)
        except Exception:
            alpha *= 0.5
            continue
        R_try = f_int_try + f_c_vec - f_ext
        R_try[fixed_dofs] = 0.0
        if not np.all(np.isfinite(R_try)):
            alpha *= 0.5
            continue
        r_try = float(np.linalg.norm(R_try))
        if r_try < diverge_factor * max(res_u_norm, 1e-30):
            return alpha
        alpha *= 0.5
    return alpha
