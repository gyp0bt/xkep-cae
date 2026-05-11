"""プロセス間 Input/Output データ契約.

dataclass(frozen=True) で不変性を保証する。
SolverStrategies: ソルバー内部の振る舞いを合成するStrategy群。
設計仕様: process-architecture.md §2.4

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class MeshData:
    """メッシュ生成結果."""

    node_coords: np.ndarray  # (n_nodes, 3)
    connectivity: np.ndarray  # (n_elems, 2)
    radii: np.ndarray | float
    n_strands: int
    strand_ids: np.ndarray | None = None  # 同素線除外用（要素→素線ID）


@dataclass(frozen=True)
class BoundaryData:
    """境界条件."""

    fixed_dofs: np.ndarray
    prescribed_dofs: np.ndarray | None = None
    prescribed_values: np.ndarray | None = None
    f_ext_total: np.ndarray | None = None
    f_ext_base: np.ndarray | None = None
    mpc_transform: object | None = None  # MPCEliminationResult（循環参照回避で object）
    mpc_groups: list | None = None  # MPCGroup リスト（UL更新時のT再構築用, status-283）
    # 処方変位の時間関数（status-286: 揺動サイクル対応）
    # prescribed_func(load_frac) -> ndarray[len(prescribed_dofs)]
    # state.u[prescribed_dofs] に書き込む絶対値を返す。
    # 設定時は prescribed_values の代わりに使用される。
    # 未設定（None）なら従来通り (load_frac - ul_frac_base) * prescribed_values。
    prescribed_func: Callable[[float], np.ndarray] | None = None


@dataclass(frozen=True)
class ContactPairSnapshotEntry:
    """1接触ペアの状態スナップショット（軽量版）.

    status-333: CR梁接触動解析でのインクリメント毎接触力・滑り量記録用。
    _ContactPairOutput / _ContactStateOutput の全フィールドではなく、
    M-κ検証に必要な最小限のフィールドのみ保持。
    """

    elem_a: int  # 要素A ID
    elem_b: int  # 要素B ID
    p_n: float  # 法線接触力
    gap: float  # ギャップ（>0: 離間）
    slip_s: float  # 接線スリップ s成分
    slip_t: float  # 接線スリップ t成分
    stick: bool  # True: スティック, False: スライディング
    dissipation: float  # 累積摩擦散逸エネルギー


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


@dataclass(frozen=True)
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
    # status-366 Phase 2: 接触法線減衰 escape hatch（候補 (e)、status-363 §4）.
    # default インスタンス（ContactNormalDampingProcess）。NR ループは
    # ContactFrictionInputData.contact_damping_coefficient > 0 のときのみ発動。
    damping: object | None = None


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
    huber_delta_h: float = 0.0,
    penalty_exponent: float = 1.0,
    active_ema_alpha: float = 0.0,
    coating_stiffness: float = 0.0,
    solver_mode: str = "implicit",
    mass_lumping: str = "row_sum",
    mass_scaling_beta: float = 1.0,
    mass_scaling_beta_outside: float = 1.0,
    mass_proportional_damping_alpha: float = 0.0,
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
    from xkep_cae.contact.damping.strategy import ContactNormalDampingProcess
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
            solver_mode=solver_mode,
            mass_lumping=mass_lumping,
            mass_scaling_beta=mass_scaling_beta,
            mass_scaling_beta_outside=mass_scaling_beta_outside,
            mass_proportional_damping_alpha=mass_proportional_damping_alpha,
        ),
        contact_force=_create_contact_force_strategy(
            ndof=ndof,
            ndof_per_node=ndof_per_node,
            smoothing_delta=smoothing_delta,
            huber_delta_h=huber_delta_h,
            penalty_exponent=penalty_exponent,
            active_ema_alpha=active_ema_alpha,
        ),
        contact_geometry=_create_contact_geometry_strategy(
            line_contact=line_contact,
            use_mortar=use_mortar,
            n_gauss=n_gauss,
        ),
        coating=_create_coating_strategy(
            coating_stiffness=coating_stiffness,
        ),
        damping=ContactNormalDampingProcess(),
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
    max_increments: int = 10000  # 最大インクリメント数（0=無制限）
    compute_condition_number: bool = False  # 条件数診断（低速）
    dof_scale_rot: float = 1.0  # 回転 DOF の NR 更新スケーリング（status-241）
    # 接触力リラクゼーション（status-247: NR 2-サイクル対策）
    contact_relax_omega: float = 0.5  # リラクゼーション係数
    stall_window: int = 4  # ストール検知窓
    tangent_fd_diagnostic: bool = False  # ストール時にFD接線診断を実行（status-257）
    kc_component_fd_diagnostic: bool = False  # K_c 成分分解 FD 診断（status-343/344）
    chattering_delta_h_boost: float = 4.0  # チャタリング時Huber delta_hブースト倍率（status-268）
    chattering_extra_attempts: int = 20  # ブースト時の追加NR反復上限（status-268）
    nr_min_restore: bool = False  # status-277: OFF（不正確な状態の持ち越し防止）
    nr_min_restore_window: int = 3  # 最小値からN回連続増加でリストア発動
    # チャタリング検知→接触凍結モード（status-284: 陽解法スイッチ）
    chattering_freeze_enabled: bool = True  # 接触凍結モード有効化
    chattering_freeze_max_cycles: int = 5  # 凍結→再評価の最大サ��クル数
    chattering_freeze_nr_max: int = 15  # 凍結中の構造NR最大反復数
    chattering_freeze_tol_factor: float = 10.0  # 凍結中の収束判定緩和倍率
    # pair-wise relaxation（status-374/375: 候補 (g3) Phase 2 NR 配線）.
    # status-284 全体凍結を pair granularity に拡張する escape hatch。
    # `pairwise_freeze_enabled=True` のとき、`PairwiseFreezingProcess` を NR
    # 反復ごとに呼び、per-pair active flip 履歴 ≥ flip_threshold のペアの DOF
    # ブロックを snapshot 値に固定する（既存全体凍結 chattering_freeze_* は
    # 排他で無効化）。default OFF。
    pairwise_freeze_enabled: bool = False
    pairwise_freeze_flip_threshold: int = 3
    pairwise_freeze_skip_type_d: bool = True
    # Augmented Lagrangian 外側ループ（status-376: 候補 (g2)、status-221 凍結解除）.
    # `al_outer_enabled=True` のとき NR を内側、`al_n_uzawa_max` 回まで Uzawa 更新で
    # 外側ループ。法線成分のみ AL（摩擦は status-147 NCP 鞍点系符号問題回避のため対象外）。
    # default OFF。19 本撚線 Type D stall 候補 (g) 最後のサブライン（数理台帳 §5）。
    al_outer_enabled: bool = False
    al_n_uzawa_max: int = 2
    # 接触残差 / active flip backtracking line search（status-362: 仮説 C 候補 (c)）
    # mixed (C+D) 領域（active flip + tangent 不整合）の直接抑制。default OFF。
    contact_backtracking_enabled: bool = False
    contact_backtracking_max_steps: int = 4
    contact_backtracking_active_flip_threshold: int = 3
    contact_backtracking_active_flip_ratio: float = 0.3
    contact_backtracking_residual_ratio: float = 2.0
    contact_backtracking_alpha_decay: float = 0.5
    contact_backtracking_min_alpha: float = 0.0625
    contact_backtracking_mixed_only: bool = True
    contact_backtracking_rate_threshold: float = 0.85
    # 接触法線減衰 escape hatch（status-366 Phase 2、候補 (e) / status-363 §4）
    # f_damp = -c_n * v_n * n̂ をペア単位で NR 残差に加算。
    # K_damp = c_n * (γ/(β·dt)) * (g_shape ⊗ g_shape) を接線剛性に加算。
    # c_n = 0 で完全無効化（default）。19 本撚線 Type D stall の
    # active×mixed 振動を粘性で escape させる MCDD 凍結解除待機中の最有力候補。
    contact_damping_coefficient: float = 0.0
    # E_damp_total / E_strain 許容上限（ContactDampingEnergyMonitorProcess が監査）.
    # 0 = チェック無効（default）。推奨: 0.05〜0.20（5〜20% 散逸許容）。
    contact_damping_energy_budget_ratio: float = 0.0
    # Hertz型非線形ペナルティ（status-285）
    penalty_exponent: float = 1.0  # 1.0=線形, 1.5=Hertz型
    # active 履歴平滑化（status-371: 候補 (g1)）.
    # NR 反復間で p_n を低域通過化し、active 集合振動を抑制する escape hatch:
    #     p_n_eff = α·p_n_new + (1-α)·p_n_prev   （α ∈ (0,1]、1.0=平滑化なし）
    # 0.0 = 完全に無効（既定、回帰防止）。19 本撚線 Type D stall の
    # NR alg 側動力学（status-370 結果 B）に対する候補 (g1) の opt-in 入口。
    # 0.1〜0.5 の範囲で掃引推奨。値が小さいほど強い平滑化（前反復に重み）。
    active_ema_alpha: float = 0.0
    # チェックポイント復元: frac途中再開（status-279）
    load_frac_start: float = 0.0  # >0: 指定fracから荷重増分を再開
    # チェックポイント保存（status-286: pickle API化）
    # checkpoint_path が非空なら、load_frac >= checkpoint_frac 到達時に pickle 保存。
    checkpoint_path: str = ""
    checkpoint_frac: float = 1.0  # 保存トリガーの load_frac 閾値
    # checkpoint復元モード（status-286: 自工程保証）
    # True にすると初期接触検出をスキップし、manager の既存 pairs をそのまま使う。
    # checkpoint から復元した manager_pairs を信頼する。
    skip_initial_detection: bool = False
    # M-κ追跡（status-333: CR梁接触動解析でのM-κヒステリシス直接取得）
    # track_mk=True かつ mk_moment_dofs/mk_curvature_func が設定されると、
    # 各収束インクリメントで (κ, M) を moment_curvature_history に記録。
    track_mk: bool = False
    mk_moment_dofs: tuple[
        int, ...
    ] = ()  # f_intから曲げモーメントを取得するDOFインデックス群（合算）
    mk_curvature_func: Callable[[float], float] | None = None  # load_frac → κ
    # 接触ペア履歴記録（status-333: 素線間接触力・滑り量の直接観測）
    # track_contact_pairs=True で各収束インクリメントの活性接触ペア状態をスナップショット保存。
    track_contact_pairs: bool = False
    # 時間積分 solver_mode（status-378 Phase 2）.
    # "implicit"（default）: Generalized-α + NR ループ（NewtonDynamicProcess）.
    # "explicit": 陽的中央差分 + 集中質量（ExplicitDynamicProcess + ExplicitCentralDifferenceProcess）.
    # 設計仕様: xkep_cae/time_integration/docs/time_integration_explicit.md
    solver_mode: str = "implicit"
    # 陽解法 driver パラメータ（solver_mode="explicit" のみ有効）.
    explicit_courant_safety: float = 0.9
    explicit_courant_check_interval: int = 50
    explicit_mass_lumping: str = "row_sum"
    # 質量スケーリング（status-379 Phase 3、候補 (h1)、Belytschko §6.4.2）.
    # `mass_scaling_beta > 1.0` で M_lump → β² · M_lump、Δt_c → β · Δt_c に拡大。
    # `mass_scaling_auto=True` のとき ExplicitDynamicProcess の Courant 監視で
    # β を逆算して set_mass_scaling_beta() で上方更新（max β は cap）。
    # 準静的近似 E_kin / E_strain < energy_budget_ratio を超えると警告出力。
    # default β=1.0 / auto=False で既存陽解法挙動完全不変。
    explicit_mass_scaling_beta: float = 1.0
    explicit_mass_scaling_auto: bool = False
    explicit_mass_scaling_max_beta: float = 100.0
    # status-384 候補 (z1c): 2 段階質量スケーリング外側 β.
    # `explicit_mass_scaling_selective=True` のとき、mask=False（梁 DOF 等）
    # に適用する β_outside² 倍化係数。default 1.0 で z1b 単独動作と等価。
    # 7 本/19 本撚線で beam dt 1.6 μs が auto-tune cap を支配する場合、modest な
    # β_outside（例 10）で beam DOF dt を拡大することで stiff β² の cap 到達を回避。
    explicit_mass_scaling_beta_outside: float = 1.0
    # status-381 h-bug-3 緩和: auto-tune が β を一気にジャンプさせると、
    # set_mass_scaling_beta() の KE 保存リスケール後も相空間に急峻な不連続を
    # 導入する。1 update あたり β を `mass_scaling_max_growth_per_update` 倍まで
    # に制限し、複数 update に分けて滑らかに成長させる。default 4.0。
    explicit_mass_scaling_max_growth_per_update: float = 4.0
    explicit_kinetic_energy_budget_ratio: float = 0.05
    # status-382 候補 (p3): 質量比例 Rayleigh damping 係数 α.
    # `C = α · M` を等価適用し $a_n -= α · v_{n−1/2}$ で動的緩和を加速。
    # M に独立に α が damping 率として作用するため Courant 安定性および β
    # スケーリングと無関係。default 0.0（無効）。
    explicit_mass_proportional_damping_alpha: float = 0.0
    # status-382 候補 (p1): BC 完了後の動的緩和ステップ数.
    # `load_frac=1.0` 到達後、BC を保持したまま `explicit_relax_steps` 回 explicit
    # ステップを継続し、過渡応答を damping で平衡まで吸収。default 0（無効）。
    # `explicit_mass_proportional_damping_alpha > 0` を推奨（damping なしでは
    # 振動が減衰せず収束しない）。
    explicit_relax_steps: int = 0
    # 緩和フェーズの収束判定許容値（||R||/||f_ref|| < tol で早期終了）.
    # 0.0 で無効（max steps まで実行）。
    explicit_relax_tol: float = 0.0
    # status-383 候補 (q1): explicit 中の UL update_reference 周期化.
    # `solver_mode="explicit"` のとき、UL `update_reference()` を毎増分ではなく
    # `explicit_ul_update_interval` 増分ごとに呼び出す。default 1 で既存挙動不変。
    # >1 のとき u_incr が累積し、relax phase の `f_int(u_incr)` が非ゼロとなり
    # 構造を平衡へ駆動できる（status-382 §3 真の根本原因への対応）。
    # CR 梁の大回転線形化破綻を避けるため、過大な値（>50 程度）は推奨しない。
    explicit_ul_update_interval: int = 1
    # status-396 候補 (z3): explicit-TL 固定モード (UL update_reference 完全停止).
    # `True` のとき、`solver_mode="explicit"` でも UL `update_reference()` を
    # 一切呼ばず、reference 配置を初期配置に固定したまま陽解法で解く（TL モード）.
    # status-395 Phase γ-3 で多要素 explicit + TL の foundation 健全性が機械精度級で
    # 確定したため、explicit_ul_update_interval=N の中間動作（毎 N 増分更新）を
    # 経由せず TL 固定で運用するための独立フィールド.
    # `explicit_ul_update_interval` とは AND ゲート評価され、本フィールドが True
    # の場合は interval 値に関わらず update_reference 呼出はゼロ.
    # default False で既存挙動完全不変.
    explicit_ul_disable_update: bool = False
    # status-384 候補 (z1b): selective mass scaling.
    # True で β² 倍化を「stiff DOF」（Gerschgorin row-sum / M_lump が
    # median × `explicit_mass_scaling_stiff_threshold_ratio` を超える DOF）に限定。
    # 梁 DOF は β=1 を維持し、接触ペナルティ等の高 K 局在 DOF のみ質量倍化する
    # ことで β² 過剰減衰（status-381 §5 解の精度 50% アンダー）を回避する。
    # default False で全 DOF 一律スケーリング（既存挙動）。
    explicit_mass_scaling_selective: bool = False
    explicit_mass_scaling_stiff_threshold_ratio: float = 10.0

    @property
    def is_dynamic(self) -> bool:
        """動的解析かどうか."""
        return self.mass_matrix is not None and self.dt_physical > 0.0


@dataclass(frozen=True)
class SolverResultData:
    """ソルバー結果."""

    u: np.ndarray
    converged: bool
    n_increments: int
    total_attempts: int
    displacement_history: tuple = ()
    contact_force_history: tuple = ()
    load_history: tuple = ()
    elapsed_seconds: float = 0.0
    diagnostics: object | None = None
    # エネルギー診断履歴（動的解析時に記録）
    energy_history: object | None = None  # EnergyHistory
    n_cutbacks: int = 0  # カットバック総数
    # 全インクリメント診断（IncrementDiagnosticsOutput のリスト）
    increment_diagnostics: tuple = ()
    # 最終接触マネージャ状態（status-299: 揺動フェーズへの引き継ぎ用）
    final_contact_manager: object | None = None
    # 最終速度・加速度（status-299: 動的解析の揺動フェーズ引き継ぎ）
    final_velocity: np.ndarray | None = None
    final_acceleration: np.ndarray | None = None
    # UL参照配置（status-299: 2フェーズリスタートでの内力整合用）
    final_ul_ref_base: np.ndarray | None = None
    final_node_coords_ref: np.ndarray | None = None
    # M-κ 履歴（status-331: 散逸エネルギー検証用, status-333: CR梁M-κ追跡拡張）
    # tuple of (curvature, moment) at each converged step
    moment_curvature_history: tuple = ()
    # 接触ペア履歴（status-333: インクリメント毎の接触ペアスナップショット）
    # tuple of (load_frac, tuple[ContactPairSnapshotEntry, ...])
    contact_pair_history: tuple = ()
    # 接触法線減衰エネルギー履歴（status-366 Phase 2、候補 (e)）
    # 成功インクリメント毎の (load_frac, E_damp_cumulative) タプル列。
    # ContactFrictionInputData.contact_damping_coefficient > 0 のみ非空。
    # ContactDampingEnergyMonitorProcess で E_damp / E_strain を監査。
    damping_energy_history: tuple = ()


@dataclass(frozen=True)
class VerifyInput:
    """検証プロセスへの入力."""

    solver_result: SolverResultData
    mesh: MeshData
    expected: dict[str, float]  # {"max_displacement": 1.23, ...}
    tolerance: float = 0.05  # 5% 許容


@dataclass(frozen=True)
class VerifyResult:
    """検証結果."""

    passed: bool
    checks: dict[str, tuple[float, float, bool]]  # {name: (actual, expected, ok)}
    report_markdown: str = ""
    snapshot_paths: tuple[str, ...] = ()  # frozen 対応: list → tuple
