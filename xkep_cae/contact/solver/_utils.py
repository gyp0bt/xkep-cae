"""ソルバー共通ユーティリティ（プライベート）.

deformed_coords / ncp_line_search を新パッケージに移植。
__xkep_cae_deprecated/contact/utils.py からのコピー。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess


def _deformed_coords(
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


def _ncp_line_search(
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
    """NCP Newton ステップの発散防止 line search."""
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


@dataclass(frozen=True)
class DeformedCoordsInput:
    """変形座標計算の入力."""

    node_coords_ref: np.ndarray
    u: np.ndarray
    ndof_per_node: int = 6


@dataclass(frozen=True)
class DeformedCoordsOutput:
    """変形座標計算の出力."""

    coords: np.ndarray


class DeformedCoordsProcess(
    SolverProcess[DeformedCoordsInput, DeformedCoordsOutput],
):
    """参照座標 + 変位から変形座標を計算する Process."""

    meta = ProcessMeta(
        name="DeformedCoords",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )

    def process(self, input_data: DeformedCoordsInput) -> DeformedCoordsOutput:
        coords = _deformed_coords(
            input_data.node_coords_ref,
            input_data.u,
            input_data.ndof_per_node,
        )
        return DeformedCoordsOutput(coords=coords)


@dataclass(frozen=True)
class NCPLineSearchInput:
    """NCP line search の入力."""

    u: np.ndarray
    du: np.ndarray
    f_ext: np.ndarray
    fixed_dofs: np.ndarray
    assemble_internal_force: Callable[[np.ndarray], np.ndarray]
    res_u_norm: float
    max_steps: int = 6
    f_c: np.ndarray | None = None
    diverge_factor: float = 1000.0


@dataclass(frozen=True)
class NCPLineSearchOutput:
    """NCP line search の出力."""

    alpha: float


class NCPLineSearchProcess(
    SolverProcess[NCPLineSearchInput, NCPLineSearchOutput],
):
    """NCP Newton ステップの発散防止 line search Process."""

    meta = ProcessMeta(
        name="NCPLineSearch",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )

    def process(self, input_data: NCPLineSearchInput) -> NCPLineSearchOutput:
        alpha = _ncp_line_search(
            input_data.u,
            input_data.du,
            input_data.f_ext,
            input_data.fixed_dofs,
            input_data.assemble_internal_force,
            input_data.res_u_norm,
            input_data.max_steps,
            input_data.f_c,
            input_data.diverge_factor,
        )
        return NCPLineSearchOutput(alpha=alpha)
