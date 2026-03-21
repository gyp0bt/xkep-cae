"""プロセス間 Input/Output データ契約.

dataclass(frozen=True) で不変性を保証する。
SolverStrategies: ソルバー内部の振る舞いを合成するStrategy群。
設計仕様: process-architecture.md §2.4

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
    mu: float | None = None


@dataclass(frozen=True)
class AssembleCallbacks:
    """アセンブリコールバック."""

    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix]
    assemble_internal_force: Callable[[np.ndarray], np.ndarray]
    ul_assembler: object | None = None


@dataclass
class SolverStrategies:
    """ソルバー内部の振る舞いを合成するStrategy群.

    各フィールドは対応するStrategy Processインスタンス。
    設計仕様: process-architecture.md §2.4

    status-222 で一本化:
    - contact_force: HuberContactForceProcess
    - friction: CoulombReturnMappingProcess
    - time_integration: GeneralizedAlphaProcess（動的のみ）
    """

    penalty: object
    friction: object
    time_integration: object
    contact_force: object | None = None  # Phase 5後半で注入
    contact_geometry: object | None = None  # Phase 5後半で注入
    coating: object | None = None  # status-169: CoatingStrategy


def default_strategies(
    *,
    ndof: int = 0,
    ndof_per_node: int = 6,
    mass_matrix: object = None,
    damping_matrix: object = None,
    dt_physical: float = 0.0,
    rho_inf: float = 0.9,
    velocity: object = None,
    acceleration: object = None,
    k_pen: float = 1.0,
    beam_E: float = 0.0,
    beam_I: float = 0.0,
    beam_L: float = 0.0,
    mu: float = 0.15,
    line_contact: bool = False,
    use_mortar: bool = False,
    n_gauss: int = 2,
    smoothing_delta: float = 0.0,
    coating_stiffness: float = 0.0,
) -> SolverStrategies:
    """基軸構成のSolverStrategiesを生成.

    status-222 で一本化:
    - Huber ペナルティ接触力
    - Coulomb 摩擦（必須）
    - 動的のみ（GeneralizedAlpha）
    """
    from xkep_cae.contact.coating.strategy import _create_coating_strategy
    from xkep_cae.contact.contact_force.strategy import (
        _create_contact_force_strategy,
    )
    from xkep_cae.contact.friction.strategy import _create_friction_strategy
    from xkep_cae.contact.geometry.strategy import (
        _create_contact_geometry_strategy,
    )
    from xkep_cae.contact.penalty.strategy import _create_penalty_strategy
    from xkep_cae.time_integration.strategy import (
        _create_time_integration_strategy,
    )

    return SolverStrategies(
        penalty=_create_penalty_strategy(
            k_pen=k_pen,
            beam_E=beam_E,
            beam_I=beam_I,
            beam_L=beam_L,
        ),
        friction=_create_friction_strategy(
            ndof=ndof,
            ndof_per_node=ndof_per_node,
        ),
        time_integration=_create_time_integration_strategy(
            mass_matrix=mass_matrix,
            damping_matrix=damping_matrix,
            dt_physical=dt_physical,
            rho_inf=rho_inf,
            velocity=velocity,
            acceleration=acceleration,
        ),
        contact_force=_create_contact_force_strategy(
            ndof=ndof,
            ndof_per_node=ndof_per_node,
            smoothing_delta=smoothing_delta,
        ),
        contact_geometry=_create_contact_geometry_strategy(
            line_contact=line_contact,
            use_mortar=use_mortar,
            n_gauss=n_gauss,
        ),
        coating=_create_coating_strategy(
            coating_stiffness=coating_stiffness,
        ),
    )


@dataclass(frozen=True)
class ContactFrictionInputData:
    """摩擦接触解析の統一入力（準静的/動的の自動判定）.

    動的パラメータ (mass_matrix, dt_physical) が指定されると動的解析
    （Generalized-α）、未指定なら準静的解析を自動選択する。
    TimeIntegrationStrategy が内部で QuasiStatic / GeneralizedAlpha を振り分ける。

    固定構成（王道構成）:
    - contact_mode = "smooth_penalty"
    - use_friction = True
    - line_contact = True
    - adaptive_timestepping = True
    """

    mesh: MeshData
    boundary: BoundaryData
    contact: ContactSetupData
    callbacks: AssembleCallbacks
    u0: np.ndarray | None = None
    # 動的解析パラメータ（全て Optional — 未指定で準静的）
    mass_matrix: sp.spmatrix | None = None
    dt_physical: float = 0.0
    rho_inf: float = 0.9
    damping_matrix: sp.spmatrix | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    # NR ソルバーパラメータ（Optional、smooth_penalty の線形収束対応）
    max_nr_attempts: int = 50
    tol_force: float = 1e-8
    tol_disp: float = 1e-8
    divergence_window: int = 5
    du_norm_cap: float = 0.0  # NR ステップ上限（||du|| < cap * ||u||、0=制限なし）

    @property
    def is_dynamic(self) -> bool:
        """動的解析かどうか."""
        return self.mass_matrix is not None and self.dt_physical > 0.0


@dataclass
class SolverResultData:
    """ソルバー結果."""

    u: np.ndarray
    converged: bool
    n_increments: int
    total_attempts: int
    displacement_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)
    load_history: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    diagnostics: object | None = None
    # エネルギー診断履歴（動的解析時に記録）
    energy_history: object | None = None  # EnergyHistory
    n_cutbacks: int = 0  # カットバック総数
    # 全インクリメント診断（IncrementDiagnosticsOutput のリスト）
    increment_diagnostics: list = field(default_factory=list)


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
