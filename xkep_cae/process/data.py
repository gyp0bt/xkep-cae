"""プロセス間 Input/Output データ契約.

dataclass(frozen=True) で不変性を保証する。
既存 NCPSolverInput/NCPSolveResult へのラッパー変換は Phase 3 で実装。

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class MeshData:
    """メッシュ生成結果."""

    node_coords: np.ndarray  # (n_nodes, 3)
    connectivity: np.ndarray  # (n_elems, 2)
    radii: np.ndarray | float
    n_strands: int
    layer_ids: np.ndarray | None = None  # 同層除外用


@dataclass(frozen=True)
class BoundaryData:
    """境界条件."""

    fixed_dofs: np.ndarray
    prescribed_dofs: np.ndarray | None = None
    prescribed_values: np.ndarray | None = None
    f_ext_total: np.ndarray | None = None
    f_ext_base: np.ndarray | None = None


@dataclass(frozen=True)
class ContactSetupData:
    """接触設定結果."""

    manager: object  # ContactManager（循環参照回避のため object）
    k_pen: float
    use_friction: bool
    mu: float | None = None
    contact_mode: str = "smooth_penalty"  # 基軸構成


@dataclass(frozen=True)
class AssembleCallbacks:
    """アセンブリコールバック."""

    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix]
    assemble_internal_force: Callable[[np.ndarray], np.ndarray]
    ul_assembler: object | None = None


@dataclass(frozen=True)
class SolverInputData:
    """NCPContactSolverProcess への統合入力.

    内部で NCPSolverInput に変換して既存ソルバーを呼び出す（ラッパー方式）。
    変換メソッドは Phase 3 で実装。
    """

    mesh: MeshData
    boundary: BoundaryData
    contact: ContactSetupData
    callbacks: AssembleCallbacks


@dataclass
class SolverResultData:
    """ソルバー結果."""

    u: np.ndarray
    converged: bool
    n_increments: int
    total_newton_iterations: int
    displacement_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    diagnostics: object | None = None


@dataclass(frozen=True)
class VerifyInput:
    """検証プロセスへの入力."""

    solver_result: SolverResultData
    mesh: MeshData
    expected: dict[str, float]  # {"max_displacement": 1.23, ...}
    tolerance: float = 0.05  # 5% 許容


@dataclass
class VerifyResult:
    """検証結果."""

    passed: bool
    checks: dict[str, tuple[float, float, bool]]  # {name: (actual, expected, ok)}
    report_markdown: str = ""
    snapshot_paths: list[str] = field(default_factory=list)
