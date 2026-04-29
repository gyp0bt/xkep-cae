"""StrandBendingOscillationProcess — 7本撚線曲げ揺動 Process.

端部剛体結合（MPC DOF消去）+ 曲げ処方変位 + 揺動サイクルの
撚線曲げ揺動解析を実行する BatchProcess。

物理モデル:
  - 7本撚線メッシュ（StrandMeshProcess）
  - 端部: MPC剛体結合（MPCEliminationProcess）
    - 各端面の全素線端部節点 → 参照点に結合
  - 境界条件:
    - 左端参照点: 全DOF固定（固定端）
    - 右端参照点: 処方回転（曲げ揺動）
  - 接触: smooth_penalty + Coulomb摩擦
  - 動的ソルバー（GeneralizedAlpha）

status-253: DOF消去MPC + 端部剛体結合 → 7本撚線曲げ揺動。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from xkep_cae.constraints.mpc_elimination import (
    MPCEliminationConfig,
    MPCEliminationProcess,
    MPCGroup,
)
from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput
from xkep_cae.contact.solver._newton_steps import (
    LinearSolveInput,
    LinearSolveProcess,
)
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.core import (
    AssembleCallbacks,
    BatchProcess,
    BoundaryData,
    ContactFrictionInputData,
    ContactSetupData,
    MeshData,
    ProcessMeta,
    SolverResultData,
)
from xkep_cae.elements._beam_assembler import (
    ULCRBeamAssemblerInput,
    ULCRBeamAssemblerProcess,
    ULCRFiberBeamAssemblerInput,
    ULCRFiberBeamAssemblerProcess,
)
from xkep_cae.elements.fiber import (
    CircularFiberSection,
    Elastic1D,
)
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.numerical_tests.three_point_bend_jig import _circle_section

# ====================================================================
# 拡張系 UL アセンブララッパー
# ====================================================================


class _ExtendedULAssemblerWrapper:
    """ULアセンブラを拡張DOF系にラップする.

    ul_assembler は梁ノードのみ (ndof_beam) を扱うが、
    MPC参照点ノードを含む拡張系 (ndof_total) との整合性が必要。
    u_total_accum / coords_ref をゼロパディングで拡張し、
    checkpoint / rollback を委譲する。

    status-283: update_reference / coords_ref を拡張系でサポート。
    """

    def __init__(
        self,
        assembler: object,
        ndof_beam: int,
        ndof_total: int,
        ref_node_coords: np.ndarray | None = None,
    ) -> None:
        self._asm = assembler
        self._ndof_beam = ndof_beam
        self._ndof_total = ndof_total
        # 参照点ノードの座標（梁ノード以降の追加ノード）
        self._ref_node_coords = ref_node_coords.copy() if ref_node_coords is not None else None
        self._ref_node_coords_ckpt = ref_node_coords.copy() if ref_node_coords is not None else None

    @property
    def u_total_accum(self) -> np.ndarray:
        u = np.zeros(self._ndof_total)
        u[: self._ndof_beam] = self._asm.u_total_accum
        return u

    @property
    def coords_ref(self) -> np.ndarray:
        beam_coords = self._asm.coords_ref
        if self._ref_node_coords is not None:
            return np.vstack([beam_coords, self._ref_node_coords])
        return beam_coords

    def checkpoint(self) -> None:
        self._asm.checkpoint()
        if self._ref_node_coords is not None:
            self._ref_node_coords_ckpt = self._ref_node_coords.copy()

    def rollback(self) -> None:
        self._asm.rollback()
        if self._ref_node_coords_ckpt is not None:
            self._ref_node_coords = self._ref_node_coords_ckpt.copy()

    def update_reference(self, u_incr: np.ndarray) -> None:
        """参照配置を増分変位で更新する."""
        self._asm.update_reference(u_incr[: self._ndof_beam])
        # 参照点ノードの座標も並進変位で更新
        if self._ref_node_coords is not None:
            n_beam_nodes = self._ndof_beam // 6
            n_ref = len(self._ref_node_coords)
            for i in range(n_ref):
                node_idx = n_beam_nodes + i
                self._ref_node_coords[i] += u_incr[node_idx * 6 : node_idx * 6 + 3]


# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class StrandBendingOscillationConfig:
    """7本撚線曲げ揺動の構成.

    Attributes:
        n_strands: 素線本数
        wire_radius: 素線半径 [mm]
        pitch_length: ピッチ長 [mm]
        gap: 素線間ギャップ [mm]
        n_elements_per_pitch: ピッチあたりの要素数
        n_pitches: ピッチ数
        E: ヤング率 [MPa]
        nu: ポアソン比
        rho: 密度 [ton/mm³]
        bending_curvature: 曲げ曲率 κ [1/mm]
        n_cycles: 揺動サイクル数
        n_increments_per_cycle: 1サイクルあたりのインクリメント数
        rho_inf: Generalized-α 数値減衰パラメータ
        mu: 摩擦係数
        k_pen: ペナルティ剛性（0=自動）
        max_nr_attempts: NR最大反復数
        tol_force: NR力収束判定
        max_increments: 最大インクリメント数
    """

    n_strands: int = 7
    wire_radius: float = 0.5  # mm (R=0.5mm → d=1mm)
    pitch_length: float = 100.0  # mm
    gap: float = 0.0  # mm (自動引き上げ)
    n_elements_per_pitch: int = 16
    n_pitches: float = 1.0
    E: float = 130.0e3  # MPa (銅)
    nu: float = 0.3
    rho: float = 8.96e-9  # ton/mm³ (銅)
    bending_curvature: float = 0.001  # 1/mm (曲率)
    n_cycles: int = 1
    n_increments_per_cycle: int = 20
    rho_inf: float = 0.9
    mu: float = 0.15
    k_pen: float = 0.0  # 0 = 自動
    max_nr_attempts: int = 50
    tol_force: float = 1e-8
    max_increments: int = 10000
    lumped_mass: bool = True
    exclude_same_strand: bool = True
    exclude_end_elements: int = (
        0  # >0: 各素線の端部N要素を接触候補から除外（status-296: MPC安定化）
    )
    tangent_fd_diagnostic: bool = False  # ストール時FD接線診断（status-257）
    kc_component_fd_diagnostic: bool = False  # K_c 成分分解 FD 診断（status-343/344）
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
    smoothing_delta: float = 0.0  # 0=自動推定（1000/wire_radius）, >0=手動指定
    huber_delta_h: float = 0.0  # >0: Huber遷移幅を直接指定（k_penスケール非依存, status-261）
    du_norm_cap: float = 0.0  # NR更新キャップ（0=制限なし）
    # チェックポイント復元（status-278: 中盤からの対策効果検証用）
    # pickle ファイルパスを指定すると、保存された u0/vel/acc から再開。
    # load_frac_start 以降の荷重増分のみ実行される。
    resume_checkpoint: str = ""  # チェックポイントファイルパス（空=通常実行）
    # free_end_mode: MPC端部剛体結合を使わず、各素線端部ノードに直接
    # 処方変位（θ_z）を与えるモード。並進DOFは自由。(status-280)
    free_end_mode: bool = False
    # contact_enabled: Falseで接触計算を無効化（radii=0で検出スキップ）
    contact_enabled: bool = True
    # loading_mode: "rotation"=端部θ_y処方（従来）, "moment"=端部M_y荷重（status-281）
    # moment モードでは EI*θ/L の曲げモーメントを各素線右端に印加。
    # エラスティカ理論により大変形でも M-θ 線形 → NR安定。
    loading_mode: str = "rotation"
    # Hertz型非線形ペナルティ（status-285）
    # 1.0=線形ペナルティ（従来）, 1.5=Hertz型（p_n ∝ δ^1.5）
    penalty_exponent: float = 1.0
    # チェックポイント保存（status-286: 曲げ完了時点のpickle保存）
    checkpoint_path: str = ""  # 非空で曲げ完了時にpickle保存
    # 揺動サイクル（status-286: sin波処方変位）
    # n_oscillation_cycles > 0 で揺動フェーズを有効化。
    # 曲げフェーズ（frac=0→1.0）完了後、揺動フェーズ（frac=0→1.0）を実行。
    # 揺動の処方変位: θ(frac) = θ_max * sin(2π * n_oscillation_cycles * frac)
    # resume_checkpoint が設定されている場合、曲げスキップで揺動のみ実行。
    n_oscillation_cycles: int = 0  # 0=曲げのみ, >0=曲げ後に揺動
    # 揺動振幅 [mm]（status-299: 先端横変位揺動）
    # >0: 揺動フェーズでθ_y回転処方の代わりに先端u_z横変位±amplitude[mm]で処方。
    # 曲げ完了位置を中心にsin波振動: u_z(frac) = u_z_bend + amplitude * sin(2π*n_osc*frac)
    # 0: 従来のθ_y回転cos波揺動（status-286互換）
    oscillation_amplitude: float = 0.0  # [mm], 0=θ回転揺動
    # 被膜パラメータ（status-301: 1000本モデルの縮小モデル用）
    coating_stiffness: float = 0.0  # 被膜剛性 [N/mm], 0=被膜なし
    coating_damping: float = 0.0  # 被膜減衰 [N·s/mm]
    coating_mu: float = 0.0  # 被膜摩擦係数
    coating_k_t_ratio: float = 0.5  # 接線剛性比
    coating_thickness: float = 0.0  # 被膜厚さ [mm], >0: core_radius = wire_radius - thickness
    coating_barrier: bool = True  # バリア関数被膜モデル（status-303）
    # ファイバー梁モード（status-330 / Phase F5）
    # True: 素線メッシュの代わりに1本のファイバー梁として解く。
    # 内部摩擦はセクション積分で処理。接触計算なし。
    use_fiber_beam: bool = False
    # ファイバー梁の材料則。None=Elastic1D（弾性）。
    # Fiber1DMaterialStrategy 準拠オブジェクトを外部から渡す。
    fiber_material: object | None = None
    # ファイバー断面離散化: "strip"=y方向ストリップ, "polar"=極座標格子
    fiber_section_type: str = "strip"
    # ファイバー数（strip: n_fiber, polar: n_radial）
    fiber_n_fiber: int = 60
    # polar 離散化の周方向分割数
    fiber_n_theta: int = 16
    # M-κ追跡 + 接触ペア履歴記録（status-333: CR梁接触動解析のM-κヒステリシス直接取得）
    # track_contact_mk=True で各収束インクリメントの (κ, M) を記録。
    # track_contact_pairs=True で各収束インクリメントの接触ペア状態をスナップショット保存。
    track_contact_mk: bool = False
    track_contact_pairs: bool = False
    # 接触法線減衰 escape hatch（status-365 Phase 1: 保有のみ、Phase 2 で solver 配線）
    # 候補 (e) (status-363 §4) — Type D stall の震源である active×mixed 領域に
    # 対し、f_damp = -c_n * v_n * n̂ の法線減衰を接触ペア単位で組立て。c1 = γ/(β·dt)
    # の整合接線剛性 K_damp = c_n * c1 * (g_shape ⊗ g_shape) も併せて加算。
    # 0 = 無効（default）。有効化時は ContactDampingEnergyMonitorProcess（Phase 2
    # 新設）で E_damp / E_strain 比を監査し budget 超過を警告する。
    contact_damping_coefficient: float = 0.0
    # E_damp_total / E_strain の許容上限。Phase 2 で Energy Monitor が超過を検知。
    # 0 = チェック無効（default）。推奨: 0.05〜0.20（5〜20% 散逸許容）。
    contact_damping_energy_budget_ratio: float = 0.0
    # チャタリング検知→接触凍結モード（status-284/368/369: 候補 (d) 19 本再評価）.
    # 既定値は 7 本撚線用に status-284 でチューニング済み。19 本 Type D stall
    # 本体では freeze_max_cycles / freeze_nr_max / freeze_tol_factor の掃引が
    # 必要。StrandBendingOscillationConfig から直接指定可能。
    # 掃引スクリプト 25_freeze_param_sweep_19strand.py は status-373 で削除（結果は status-368 に確定記録）。
    #
    # 19 本以上の大規模撚線向け opt-in 推奨（status-368 Case B / status-369 明記）:
    #     chattering_freeze_nr_max = 30   # default 15 の 2x
    #
    # 実測効果（status-368 19 本 90° 曲げ）: frac 0.3739 → 0.5642（+50.9%、
    # status-339 baseline 0.4839 比 +16.6%）。最終 NR Type 分布の mixed (D+E)
    # 比率が 69% → 56% に低下（BT line search と同パターン）。代償として
    # elapsed +251%（245s → 863s）。MCDD 凍結解除条件（frac=1.0 完走）未達
    # のため default 変更は実施せず（7 本系の回帰リスク回避）。19 本以上で
    # frac=1.0 が未達な系には `chattering_freeze_nr_max=30` を明示指定する。
    chattering_freeze_enabled: bool = True
    chattering_freeze_max_cycles: int = 5
    chattering_freeze_nr_max: int = 15
    chattering_freeze_tol_factor: float = 10.0
    # active 履歴平滑化（status-371: 候補 (g1) 実装、status-372: 実機 α 掃引）.
    # NR 反復間で p_n を低域通過フィルタする escape hatch:
    #     p_n_eff = α·p_n_new + (1-α)·p_n_prev
    # 0.0=無効（既定）。
    #
    # status-372 実機 α 掃引結果:
    #
    # - **7 本系 opt-in 推奨**: `active_ema_alpha=0.5` で frac=1.0 維持 +
    #   cutback 57→22（**-61% 削減**）+ elapsed -11%（298→265s）。
    #   α=0.30 でも frac=1.0 維持（cb -75%）だが elapsed ほぼ同等。
    #   **α=0.10 は早期 stall** で却下（弱平滑化が逆効果、status-262
    #   smoothing_delta 非単調性と類似）。
    # - **19 本以上は却下方向**: 全 α で gate「frac ≥ 0.6」未達。
    #   α=0.50 で frac=0.5133（baseline 0.3739 比 +37%、status-339 baseline
    #   0.4839 比 +6%）の部分改善はあるが elapsed +131% でコスト過大。
    #
    # default=0.0 で 7 本撚線回帰なし（456 contact tests 全 pass）。
    # 7 本系で cutback 削減 opt-in としては `active_ema_alpha=0.5` を推奨
    # （`docs/roadmap.md` §「撚線規模別 opt-in チューニング」表参照）。
    active_ema_alpha: float = 0.0
    # pair-wise relaxation（status-374/375: 候補 (g3) Phase 2 NR 配線）.
    # status-284 全体凍結を pair granularity に拡張する escape hatch。
    # `pairwise_freeze_enabled=True` のとき、NR 反復ごとに per-pair の active
    # 履歴 (≥ flip_threshold) で凍結対象ペアを判定し、当該ペアの DOF ブロックを
    # snapshot 値に固定する。既存全体凍結 chattering_freeze_* は pair-wise が
    # 有効な間は排他で無効化される。default OFF（19 本撚線 Type D stall opt-in）。
    pairwise_freeze_enabled: bool = False
    pairwise_freeze_flip_threshold: int = 3
    pairwise_freeze_skip_type_d: bool = True
    # Augmented Lagrangian 外側ループ（status-376: 候補 (g2)、status-221 凍結解除）.
    # `al_outer_enabled=True` のとき、NR 内側 + AL 外側 (max al_n_uzawa_max cycle) の
    # 二重ループ化。各 NR 収束後に Uzawa 更新 `λ_new = max(0, p_n_eff_converged)` を
    # per-pair で実施。法線成分のみ AL（摩擦は status-147 NCP 鞍点系符号問題回避のため
    # 対象外）。default OFF。19 本撚線 Type D stall 候補 (g) 最後のサブライン。
    # 数理台帳: docs/math/03_huber_contact_penalty.md §5。
    al_outer_enabled: bool = False
    al_n_uzawa_max: int = 2
    # 時間積分 solver_mode（status-377 Phase 1: ExplicitCentralDifferenceProcess 単体実装）.
    # "implicit"（default）: 既存 Generalized-α + NR ループ（GeneralizedAlphaProcess）.
    # "explicit": 中央差分 + 集中質量（ExplicitCentralDifferenceProcess）.
    #   候補 (g) 全候補（NR alg 側 escape hatch）が gate frac=0.6 未達だったことを受けて、
    #   19 本以上の K_c x/z カップリング不整合（status-344）を時間積分自体で安定化する目的で
    #   status-377 Phase 1 で導入。
    #
    # **Phase 1 (status-377) 制約**: Process 単体実装 + 単体テスト + 設計仕様のみ。
    # solver path への配線は Phase 2（次 status）で実施。`solver_mode="explicit"` 指定時は
    # NotImplementedError が発生し、Phase 2 待機を明示する。
    #
    # 設計仕様: xkep_cae/time_integration/docs/time_integration_explicit.md
    solver_mode: Literal["implicit", "explicit"] = "implicit"
    # 陽解法 driver パラメータ（status-378 Phase 2、solver_mode="explicit" のみ有効）.
    explicit_courant_safety: float = 0.9
    explicit_courant_check_interval: int = 50
    explicit_mass_lumping: str = "row_sum"


@dataclass(frozen=True)
class StrandBendingOscillationResult:
    """7本撚線曲げ揺動の結果.

    Attributes:
        solver_result: ソルバー結果
        mesh: メッシュデータ（参照点ノード含む）
        mpc_config: MPC構成（デバッグ用）
        n_ref_nodes: 追加された参照点ノード数
        n_strand_nodes: 元の撚線ノード数
        total_ndof: 全体DOF数
        bending_angle: 処方曲げ角度 [rad]
    """

    solver_result: SolverResultData
    mesh: MeshData
    n_ref_nodes: int
    n_strand_nodes: int
    total_ndof: int
    bending_angle: float


# ====================================================================
# 端部節点収集
# ====================================================================


def _collect_end_nodes(
    connectivity: np.ndarray,
    n_strands: int,
    strand_ids: np.ndarray,
) -> tuple[list[int], list[int]]:
    """各素線の左端/右端節点を収集する.

    Returns:
        left_nodes: 左端（最小x方向）節点のリスト
        right_nodes: 右端（最大x方向）節点のリスト
    """
    left_nodes = []
    right_nodes = []

    for s in range(n_strands):
        # この素線の要素を取得
        elem_mask = strand_ids == s
        strand_elems = connectivity[elem_mask]
        if len(strand_elems) == 0:
            continue
        # 要素に1回だけ出現する節点 = 端部節点
        node_count: dict[int, int] = {}
        for e in strand_elems:
            for n in e:
                node_count[int(n)] = node_count.get(int(n), 0) + 1
        end_nodes = [n for n, c in node_count.items() if c == 1]
        end_nodes.sort()
        if len(end_nodes) >= 2:
            left_nodes.append(end_nodes[0])
            right_nodes.append(end_nodes[-1])
        elif len(end_nodes) == 1:
            left_nodes.append(end_nodes[0])

    return left_nodes, right_nodes


def _collect_adjacent_nodes(
    connectivity: np.ndarray,
    strand_ids: np.ndarray,
    end_nodes: list[int],
) -> list[int]:
    """各端部節点の隣接ノード（1要素内側）を返す.

    力カップル方式でモーメント荷重を生成するために使用。
    """
    adj_nodes = []
    for end_node in end_nodes:
        found = False
        for _i, e in enumerate(connectivity):
            if end_node in e:
                # 端部ノードと同じ要素の他方のノード
                other = int(e[1]) if int(e[0]) == end_node else int(e[0])
                adj_nodes.append(other)
                found = True
                break
        if not found:
            adj_nodes.append(end_node)  # fallback
    return adj_nodes


# ====================================================================
# 静的 Newton-Raphson ソルバー（接触なし問題用）
# ====================================================================


def _static_nr_solve(  # noqa: PLR0912, PLR0915
    assembler: object,
    ndof: int,
    fixed_dofs: np.ndarray,
    prescribed_dofs: np.ndarray,
    prescribed_values: np.ndarray,
    f_ext_total: np.ndarray,
    n_increments: int,
    max_nr: int = 50,
    tol: float = 1e-8,
    show_progress: bool = True,
    use_ul: bool = True,
    prescribed_func: object | None = None,
    track_mk: bool = False,
    mk_curvature_func: object | None = None,
    mk_moment_dof: int = -1,
) -> SolverResultData:
    """接触なし問題用の静的NRソルバー.

    status-281: 動的ソルバー（Generalized-α）は慣性項による残差連成で
    ヘリカル複数素線の曲げ収束が困難。接触なし問題では純粋な
    静的NR法で直接求解する。

    Args:
        use_ul: True=UL定式化（毎ステップ update_reference）、
                False=TL定式化（全変位追跡、非線形材料向け）。
                status-330: ファイバー梁の非線形材料は TL が必要
                （UL の update_reference で eps_p/alpha の参照枠が
                不整合になる CR梁ULのf_int=0問題を回避）。
        prescribed_func: frac → np.ndarray。非None時、処方変位を
                frac_target * prescribed_values の代わりに
                prescribed_func(frac_target) で計算。サイクル荷重に対応。
                status-331: 散逸エネルギー検証用。
        track_mk: True で M-κ 履歴を追跡。
        mk_curvature_func: frac → float。曲率を返す関数。
        mk_moment_dof: 反力モーメントを抽出するDOFインデックス。

    Returns:
        SolverResultData: ソルバー結果
    """
    import time

    t0 = time.time()
    load_history: list[float] = []
    mk_history: list[tuple[float, float]] = []
    n_cutbacks = 0
    total_attempts = 0

    # BC用マスク
    all_constrained = set(fixed_dofs.tolist()) | set(prescribed_dofs.tolist())
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[np.array(sorted(all_constrained), dtype=int)] = False
    _linear_solver = LinearSolveProcess()

    frac = 0.0
    dt_frac = 1.0 / n_increments
    dt_min = dt_frac / 64.0
    incr = 0

    # TL モード: 全変位を追跡（update_reference を呼ばない）
    u_total = np.zeros(ndof)
    u_total_ckpt = np.zeros(ndof)

    # M-κ初期点
    if track_mk and mk_curvature_func is not None:
        mk_history.append((mk_curvature_func(0.0), 0.0))

    while frac < 1.0 - 1e-12:
        dt_try = min(dt_frac, 1.0 - frac)
        frac_target = frac + dt_try

        converged = False
        f_ref = 1.0

        # 処方変位計算: prescribed_func があればそちらを使用
        if prescribed_func is not None:
            pv_target = prescribed_func(frac_target)
            pv_incr = pv_target - (prescribed_func(frac) if use_ul else np.zeros_like(pv_target))
        else:
            pv_target = frac_target * prescribed_values
            pv_incr = (frac_target - frac) * prescribed_values

        if use_ul:
            u_incr = np.zeros(ndof)
            u_incr[prescribed_dofs] = pv_incr
        else:
            u_incr = u_total.copy()
            u_incr[prescribed_dofs] = pv_target

        for att in range(max_nr):
            total_attempts += 1
            f_int = assembler.assemble_internal_force(u_incr)
            R = f_int - f_ext_total * frac_target
            R[fixed_dofs] = 0.0
            R[prescribed_dofs] = 0.0

            res_norm = float(np.linalg.norm(R[free_mask]))
            f_ext_norm = float(np.linalg.norm(f_ext_total * frac_target))
            if f_ext_norm > 1e-30:
                f_ref = f_ext_norm
            elif att == 0:
                f_ref = max(res_norm, 1.0)

            if show_progress and att % 5 == 0:
                print(
                    f"  Static Incr {incr + 1} (frac={frac_target:.4f}), "
                    f"att {att}, ||R||/||f||={res_norm / f_ref:.3e}"
                )

            if res_norm / f_ref < tol:
                converged = True
                break
            if att > 3 and res_norm / f_ref > 1e6:
                break  # 発散

            K = assembler.assemble_tangent(u_incr)
            solve_out = _linear_solver.process(
                LinearSolveInput(
                    K_T=K,
                    R_u=R,
                    fixed_dofs=np.array(sorted(all_constrained), dtype=int),
                )
            )
            if not solve_out.success:
                break
            u_incr += solve_out.du

        if converged:
            # M-κ 履歴追跡: 収束時の反力モーメントを記録
            if track_mk and mk_curvature_func is not None and mk_moment_dof >= 0:
                f_int_conv = assembler.assemble_internal_force(u_incr)
                kappa = mk_curvature_func(frac_target)
                moment = float(f_int_conv[mk_moment_dof])
                mk_history.append((kappa, moment))

            if use_ul:
                assembler.update_reference(u_incr)
            else:
                u_total[:] = u_incr
            assembler.checkpoint()
            u_total_ckpt[:] = u_total if not use_ul else u_total_ckpt
            incr += 1
            frac = frac_target
            load_history.append(frac)
            dt_frac = min(dt_frac * 1.5, 1.0 / n_increments)
        else:
            n_cutbacks += 1
            assembler.rollback()
            if not use_ul:
                u_total[:] = u_total_ckpt
            dt_frac *= 0.5
            if dt_frac < dt_min:
                if show_progress:
                    print(f"  Static solver: dt_min到達 (frac={frac:.4f})")
                break

    elapsed = time.time() - t0
    if use_ul:
        u_out = assembler.u_total_accum
    else:
        u_out = u_total
    return SolverResultData(
        u=u_out,
        converged=frac >= 1.0 - 1e-10,
        n_increments=incr,
        total_attempts=total_attempts,
        load_history=tuple(load_history),
        elapsed_seconds=elapsed,
        n_cutbacks=n_cutbacks,
        moment_curvature_history=tuple(mk_history),
    )


# ====================================================================
# Process
# ====================================================================


class StrandBendingOscillationProcess(
    BatchProcess[StrandBendingOscillationConfig, StrandBendingOscillationResult],
):
    """7本撚線曲げ揺動 Process.

    パイプライン:
    1. StrandMeshProcess でメッシュ生成
    2. 端部参照点ノードを追加
    3. MPCEliminationProcess で端部剛体結合
    4. ULCRBeamAssemblerProcess でアセンブラ構築
    5. 曲げ処方変位を境界条件に設定
    6. ContactFrictionProcess で求解
    """

    meta = ProcessMeta(
        name="StrandBendingOscillation",
        module="batch",
        version="1.0.0",
        document_path="docs/strand_bending_oscillation.md",
    )
    uses = [
        StrandMeshProcess,
        MPCEliminationProcess,
        ULCRBeamAssemblerProcess,
        ULCRFiberBeamAssemblerProcess,
        ContactFrictionProcess,
    ]

    def process(  # noqa: C901, PLR0912, PLR0915
        self,
        input_data: StrandBendingOscillationConfig,
    ) -> StrandBendingOscillationResult:
        """撚線曲げ揺動を実行."""
        cfg = input_data

        # status-378 Phase 2: solver_mode="explicit" 時は ExplicitDynamicProcess
        # （陽的中央差分 + ContactForceAssemblyProcess）が ContactFrictionProcess
        # 内部で起動される。NR 反復を経由しない 1 増分 1 step driver。
        # 詳細: xkep_cae/time_integration/docs/time_integration_explicit.md §Phase 2

        # ── ファイバー梁モード分岐（Phase F5） ──
        if cfg.use_fiber_beam:
            return self._process_fiber_beam(cfg)

        # ── 1. メッシュ生成 ──
        mesh_result = StrandMeshProcess().process(
            StrandMeshConfig(
                n_strands=cfg.n_strands,
                wire_radius=cfg.wire_radius,
                pitch_length=cfg.pitch_length,
                gap=cfg.gap,
                n_elements_per_pitch=cfg.n_elements_per_pitch,
                n_pitches=cfg.n_pitches,
            )
        )
        mesh = mesh_result.mesh
        # 接触無効時: radii=0 で接触ペア検出をスキップ（status-280）
        if not cfg.contact_enabled:
            mesh = MeshData(
                node_coords=mesh.node_coords,
                connectivity=mesh.connectivity,
                radii=0.0,
                n_strands=mesh.n_strands,
                strand_ids=mesh.strand_ids,
            )
        strand_coords = mesh.node_coords
        strand_conn = mesh.connectivity
        n_strand_nodes = len(strand_coords)

        # ── 2. 端部節点の収集 ──
        left_nodes, right_nodes = _collect_end_nodes(strand_conn, cfg.n_strands, mesh.strand_ids)

        if cfg.free_end_mode:
            return self._process_free_end(
                cfg, mesh, strand_coords, strand_conn, n_strand_nodes, left_nodes, right_nodes
            )

        # ── 3. 参照点ノードの追加（MPCモード） ──
        # 左端参照点 = 左端節点群の重心
        left_coords = strand_coords[left_nodes]
        left_ref_coord = np.mean(left_coords, axis=0)
        # 右端参照点 = 右端節点群の重心
        right_coords = strand_coords[right_nodes]
        right_ref_coord = np.mean(right_coords, axis=0)

        # 拡張座標: 撚線ノード + 2参照点
        n_ref_nodes = 2
        ref_left_node = n_strand_nodes
        ref_right_node = n_strand_nodes + 1
        extended_coords = np.vstack(
            [strand_coords, left_ref_coord.reshape(1, 3), right_ref_coord.reshape(1, 3)]
        )
        n_total_nodes = len(extended_coords)
        ndof = n_total_nodes * 6

        # 拡張メッシュ（参照点はアセンブラに含めないが、座標系に含める）
        extended_mesh = MeshData(
            node_coords=extended_coords,
            connectivity=strand_conn,
            radii=mesh.radii,
            n_strands=cfg.n_strands,
            strand_ids=mesh.strand_ids,
        )

        # ── 4. MPC構築 ──
        mpc_groups = []
        # 左端MPC
        if left_nodes:
            mpc_groups.append(
                MPCGroup(
                    master_node=ref_left_node,
                    slave_nodes=np.array(left_nodes, dtype=int),
                    slave_coords=strand_coords[left_nodes],
                    master_coord=left_ref_coord,
                )
            )
        # 右端MPC
        if right_nodes:
            mpc_groups.append(
                MPCGroup(
                    master_node=ref_right_node,
                    slave_nodes=np.array(right_nodes, dtype=int),
                    slave_coords=strand_coords[right_nodes],
                    master_coord=right_ref_coord,
                )
            )

        mpc_result = MPCEliminationProcess().process(
            MPCEliminationConfig(
                mpc_groups=mpc_groups,
                ndof_total=ndof,
                ndof_per_node=6,
            )
        )

        # ── 5. アセンブラ構築 ──
        sec = _circle_section(cfg.wire_radius * 2.0, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))

        beam_result = ULCRBeamAssemblerProcess().process(
            ULCRBeamAssemblerInput(
                node_coords=strand_coords,  # 撚線ノードのみ
                connectivity=strand_conn,
                E=cfg.E,
                G=G,
                A=sec["A"],
                Iy=sec["Iy"],
                Iz=sec["Iz"],
                J=sec["J"],
                kappa_y=sec["kappa"],
                kappa_z=sec["kappa"],
            )
        )
        assembler = beam_result.assembler
        ndof_beam = n_strand_nodes * 6

        # 剛性/内力のラッパー: 参照点DOFを含む拡張系にゼロパディング
        def _assemble_tangent_extended(u: np.ndarray) -> sp.csr_matrix:
            u_beam = u[:ndof_beam]
            K_beam = assembler.assemble_tangent(u_beam)
            # 拡張系: 参照点DOFの行列はゼロ
            K_ext = sp.lil_matrix((ndof, ndof))
            K_ext[:ndof_beam, :ndof_beam] = K_beam
            return K_ext.tocsr()

        def _assemble_internal_force_extended(u: np.ndarray) -> np.ndarray:
            u_beam = u[:ndof_beam]
            f_beam = assembler.assemble_internal_force(u_beam)
            f_ext_padded = np.zeros(ndof)
            f_ext_padded[:ndof_beam] = f_beam
            return f_ext_padded

        # 質量行列
        M_beam = assembler.assemble_mass(cfg.rho, lumped=cfg.lumped_mass)
        M_ext = sp.lil_matrix((ndof, ndof))
        M_ext[:ndof_beam, :ndof_beam] = M_beam

        # MPC参照点の質量補強（status-278: 回転慣性NR収束不良修正）
        # lumped質量行列では参照点ノードの質量がゼロ。MPC変換 T^T M T で
        # slave ノードの質量が参照点に集約されるが、回転慣性が ~10^-7 と
        # 極めて小さく、effective_stiffness の回転対角項がほぼゼロになる。
        # → NRの回転DOF更新が発散し、残差が収束しない。
        # 対策: 参照点に slave ノードの質量を直接加算し、
        # 平行軸定理で回転慣性を計算する。
        for ref_node, end_nodes in [(ref_left_node, left_nodes), (ref_right_node, right_nodes)]:
            ref_coord = (
                strand_coords[ref_node]
                if ref_node < len(strand_coords)
                else np.mean(strand_coords[end_nodes], axis=0)
            )
            m_total = 0.0  # 並進質量の和
            I_xx, I_yy, I_zz = 0.0, 0.0, 0.0  # 回転慣性（平行軸定理）
            for en in end_nodes:
                m_n = float(M_beam[en * 6, en * 6])  # ノード並進質量
                m_total += m_n
                # 平行軸定理: I += m * r²
                if en < len(strand_coords):
                    dr = strand_coords[en] - ref_coord
                    I_xx += m_n * (dr[1] ** 2 + dr[2] ** 2)
                    I_yy += m_n * (dr[0] ** 2 + dr[2] ** 2)
                    I_zz += m_n * (dr[0] ** 2 + dr[1] ** 2)
                # ノード自身の回転慣性も加算
                for d in range(3):
                    rot_dof = en * 6 + 3 + d
                    if rot_dof < M_beam.shape[0]:
                        i_n = float(M_beam[rot_dof, rot_dof])
                        if d == 0:
                            I_xx += i_n
                        elif d == 1:
                            I_yy += i_n
                        else:
                            I_zz += i_n
            # 参照点に質量を設定
            for d in range(3):
                M_ext[ref_node * 6 + d, ref_node * 6 + d] = m_total
            M_ext[ref_node * 6 + 3, ref_node * 6 + 3] = I_xx
            M_ext[ref_node * 6 + 4, ref_node * 6 + 4] = I_yy
            M_ext[ref_node * 6 + 5, ref_node * 6 + 5] = I_zz

        M_ext = M_ext.tocsr()

        # ── 6. 境界条件 ──
        # 左端参照点: 全DOF固定
        fixed_dofs = set()
        for k in range(6):
            fixed_dofs.add(ref_left_node * 6 + k)

        # 右端参照点: u_y固定 + θ_x,θ_z固定、θ_y のみ処方変位
        # status-280: x-z面カンチレバー曲げ。
        # u_x, u_z は自由（梁端の横移動+軸短縮を許容）。
        # u_y を固定（面外変位を拘束）。
        for k in [1, 3, 5]:  # u_y, θ_x, θ_z を固定
            fixed_dofs.add(ref_right_node * 6 + k)

        # 曲げ角度 = κ * L
        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length

        # 処方変位: 右端参照点の θ_y （x-z面曲げ回転）
        prescribed_dof = ref_right_node * 6 + 4  # θ_y
        prescribed_dofs = np.array([prescribed_dof], dtype=int)
        prescribed_values = np.array([bending_angle])

        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

        # 時間パラメータ: 揺動周期を動的解析のt_totalに設定
        # 梁の固有振動数から概算
        sec_Iy = sec["Iy"]
        sec_A = sec["A"]
        f1 = (math.pi / (2.0 * strand_length**2)) * math.sqrt(
            cfg.E * sec_Iy * cfg.n_strands / (cfg.rho * sec_A * cfg.n_strands)
        )
        T1 = 1.0 / f1 if f1 > 1e-30 else 1.0
        # 揺動周期 = 少なくとも固有周期の10倍（準静的挙動）
        t_cycle = max(10.0 * T1, 1.0)
        t_total = t_cycle * cfg.n_cycles

        dt_initial = t_total / (cfg.n_increments_per_cycle * cfg.n_cycles)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs,
            prescribed_values=prescribed_values,
            f_ext_total=np.zeros(ndof),
            mpc_transform=mpc_result,
            mpc_groups=mpc_groups,  # UL更新時のT再構築用（status-283）
        )

        # ── 7. 接触設定 ──
        # smoothing_delta 自動推定: δ = 1000 / r_min（status-260: 5000→1000に変更）
        _smoothing_delta = (
            cfg.smoothing_delta if cfg.smoothing_delta > 0.0 else 1000.0 / cfg.wire_radius
        )
        contact_config = _ContactConfigInput(
            beam_E=cfg.E,
            beam_I=sec_Iy,
            mu=cfg.mu,
            adaptive_timestepping=True,
            dt_min_fraction=dt_initial / (t_total * 64.0),
            dt_max_fraction=dt_initial / t_total,
            exclude_same_strand=cfg.exclude_same_strand,
            exclude_end_elements=cfg.exclude_end_elements,
            smoothing_delta=_smoothing_delta,
            huber_delta_h=cfg.huber_delta_h,
            coating_stiffness=cfg.coating_stiffness,
            coating_damping=cfg.coating_damping,
            coating_mu=cfg.coating_mu,
            coating_k_t_ratio=cfg.coating_k_t_ratio,
            coating_thickness=cfg.coating_thickness,
            coating_barrier=cfg.coating_barrier,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=cfg.k_pen,
            mu=cfg.mu,
        )

        # ── 8. ソルバー実行 ──
        # ULアセンブラを拡張DOF系にラップ（参照点DOFのゼロパディング）
        # 参照点ノード座標（梁ノード以降の追加ノード）
        _ref_coords = extended_coords[n_strand_nodes:]
        extended_assembler = _ExtendedULAssemblerWrapper(
            assembler, ndof_beam, ndof, ref_node_coords=_ref_coords
        )

        # チェックポイント復元（status-278, status-279で途中再開対応）
        _u0 = None
        _vel0 = None
        _acc0 = None
        _frac_start = 0.0
        if cfg.resume_checkpoint:
            import pickle as _pickle

            with open(cfg.resume_checkpoint, "rb") as _f:
                _ckpt = _pickle.load(_f)
            _u0 = _ckpt["state"].u.copy()
            _vel0 = _ckpt["time_vel"]
            _acc0 = _ckpt["time_acc"]
            _frac_start = _ckpt["load_frac"]
            # ULアセンブラの累積変位を復元
            if hasattr(extended_assembler._asm, "_u_total_accum"):
                extended_assembler._asm._u_total_accum[:] = _u0[:ndof_beam]
            print(f"  [RESUME] frac={_frac_start:.4f}, ||u||={np.linalg.norm(_u0):.4e}")

        solver_input = ContactFrictionInputData(
            mesh=extended_mesh,
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=_assemble_tangent_extended,
                assemble_internal_force=_assemble_internal_force_extended,
                ul_assembler=extended_assembler,
            ),
            u0=_u0,
            mass_matrix=M_ext,
            dt_physical=t_total,
            rho_inf=cfg.rho_inf,
            velocity=_vel0,
            acceleration=_acc0,
            max_nr_attempts=cfg.max_nr_attempts,
            tol_force=cfg.tol_force,
            max_increments=cfg.max_increments,
            tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
            kc_component_fd_diagnostic=cfg.kc_component_fd_diagnostic,
            du_norm_cap=cfg.du_norm_cap,
            load_frac_start=_frac_start,
            penalty_exponent=cfg.penalty_exponent,
            # 接触 backtracking line search（status-362）
            contact_backtracking_enabled=cfg.contact_backtracking_enabled,
            contact_backtracking_max_steps=cfg.contact_backtracking_max_steps,
            contact_backtracking_active_flip_threshold=cfg.contact_backtracking_active_flip_threshold,
            contact_backtracking_active_flip_ratio=cfg.contact_backtracking_active_flip_ratio,
            contact_backtracking_residual_ratio=cfg.contact_backtracking_residual_ratio,
            contact_backtracking_alpha_decay=cfg.contact_backtracking_alpha_decay,
            contact_backtracking_min_alpha=cfg.contact_backtracking_min_alpha,
            contact_backtracking_mixed_only=cfg.contact_backtracking_mixed_only,
            contact_backtracking_rate_threshold=cfg.contact_backtracking_rate_threshold,
            # 接触法線減衰 escape hatch（status-366 Phase 2、候補 (e)）
            contact_damping_coefficient=cfg.contact_damping_coefficient,
            contact_damping_energy_budget_ratio=cfg.contact_damping_energy_budget_ratio,
            # チャタリング検知→接触凍結モード（status-368 候補 (d)）
            chattering_freeze_enabled=cfg.chattering_freeze_enabled,
            chattering_freeze_max_cycles=cfg.chattering_freeze_max_cycles,
            chattering_freeze_nr_max=cfg.chattering_freeze_nr_max,
            chattering_freeze_tol_factor=cfg.chattering_freeze_tol_factor,
            # active 履歴平滑化（status-371 候補 (g1)）
            active_ema_alpha=cfg.active_ema_alpha,
            # pair-wise relaxation（status-374/375 候補 (g3) Phase 2）
            pairwise_freeze_enabled=cfg.pairwise_freeze_enabled,
            pairwise_freeze_flip_threshold=cfg.pairwise_freeze_flip_threshold,
            pairwise_freeze_skip_type_d=cfg.pairwise_freeze_skip_type_d,
            # Augmented Lagrangian 外側ループ（status-376）
            al_outer_enabled=cfg.al_outer_enabled,
            al_n_uzawa_max=cfg.al_n_uzawa_max,
            # 陽的中央差分時間積分（status-378 Phase 2）
            solver_mode=cfg.solver_mode,
            explicit_courant_safety=cfg.explicit_courant_safety,
            explicit_courant_check_interval=cfg.explicit_courant_check_interval,
            explicit_mass_lumping=cfg.explicit_mass_lumping,
        )
        solver = ContactFrictionProcess()
        solver_result = solver.process(solver_input)

        return StrandBendingOscillationResult(
            solver_result=solver_result,
            mesh=extended_mesh,
            n_ref_nodes=n_ref_nodes,
            n_strand_nodes=n_strand_nodes,
            total_ndof=ndof,
            bending_angle=bending_angle,
        )

    def _process_free_end(  # noqa: PLR0912, PLR0915
        self,
        cfg: StrandBendingOscillationConfig,
        mesh: MeshData,
        strand_coords: np.ndarray,
        strand_conn: np.ndarray,
        n_strand_nodes: int,
        left_nodes: list[int],
        right_nodes: list[int],
    ) -> StrandBendingOscillationResult:
        """MPC不使用・端部直接処方モードで撚線曲げ揺動を実行.

        status-280: MPC端部剛体結合の代わりに、各素線端部ノードの
        θ_z を直接処方し、並進DOFは自由にする。
        - 左端: 全素線端部ノードの全6DOF固定
        - 右端: θ_z処方、θ_x/θ_y固定、u_x/u_y/u_z自由
        - 参照点ノード不要 → 拡張系不要 → MPC不要
        """
        ndof = n_strand_nodes * 6

        # ── アセンブラ構築 ──
        sec = _circle_section(cfg.wire_radius * 2.0, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))

        beam_result = ULCRBeamAssemblerProcess().process(
            ULCRBeamAssemblerInput(
                node_coords=strand_coords,
                connectivity=strand_conn,
                E=cfg.E,
                G=G,
                A=sec["A"],
                Iy=sec["Iy"],
                Iz=sec["Iz"],
                J=sec["J"],
                kappa_y=sec["kappa"],
                kappa_z=sec["kappa"],
            )
        )
        assembler = beam_result.assembler

        # 質量行列（直接使用、拡張不要）
        M = assembler.assemble_mass(cfg.rho, lumped=cfg.lumped_mass)

        # ── 境界条件 ──
        fixed_dofs: set[int] = set()
        prescribed_dofs_list: list[int] = []
        prescribed_values_list: list[float] = []
        f_ext = np.zeros(ndof)

        # 左端: 全素線端部ノードの全6DOF固定
        for n in left_nodes:
            for k in range(6):
                fixed_dofs.add(n * 6 + k)

        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length

        if cfg.loading_mode == "moment":
            # status-281: 力カップル方式モーメント荷重
            # エラスティカ理論: M = EI * θ / L（大変形でも M-θ 線形）
            # NR収束判定は並進残差のみ使用するため、純モーメントは偽収束する。
            # 対策: 端部2ノードに逆向き力 (F, -F) で力カップルを構成。
            # M_y = F_x * Δz → F_x = M_y / Δz
            sec_Iy = sec["Iy"]
            m_target = cfg.E * sec_Iy * bending_angle / strand_length
            adj_nodes = _collect_adjacent_nodes(strand_conn, mesh.strand_ids, right_nodes)
            for n_end, n_adj in zip(right_nodes, adj_nodes, strict=True):
                # 端部ノードと隣接ノードの距離（z方向投影）
                dz = abs(strand_coords[n_end][2] - strand_coords[n_adj][2])
                if dz < 1e-10:
                    dz = np.linalg.norm(strand_coords[n_end] - strand_coords[n_adj])
                f_couple = m_target / dz
                # x方向の力カップルで y軸周りモーメントを生成
                # M_y = F_x * Δz (右手系: +x力 × +z位置 = +y回転)
                f_ext[n_end * 6 + 0] = f_couple  # +F_x at tip
                f_ext[n_adj * 6 + 0] = -f_couple  # -F_x at adjacent
                # u_y を固定（面外変位を拘束）
                fixed_dofs.add(n_end * 6 + 1)
                # u_x, u_z, θ_x, θ_y, θ_z は全て自由
        else:
            # 従来: θ_y処方（x-z面曲げ）, θ_x/θ_z自由, u自由
            for n in right_nodes:
                prescribed_dofs_list.append(n * 6 + 4)
                prescribed_values_list.append(bending_angle)

        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)
        prescribed_dofs_arr = np.array(prescribed_dofs_list, dtype=int)
        prescribed_values_arr = np.array(prescribed_values_list)

        # ── 時間パラメータ ──
        sec_Iy = sec["Iy"]
        sec_A = sec["A"]
        f1 = (math.pi / (2.0 * strand_length**2)) * math.sqrt(
            cfg.E * sec_Iy * cfg.n_strands / (cfg.rho * sec_A * cfg.n_strands)
        )
        T1 = 1.0 / f1 if f1 > 1e-30 else 1.0
        t_cycle = max(10.0 * T1, 1.0)

        # status-299: 曲げ+揺動統合モード
        # n_oscillation_cycles > 0 の場合、曲げ+揺動を1回のソルバーで実行。
        # frac_bend = 曲げフェーズが全体のどこまでか
        _n_osc = cfg.n_oscillation_cycles
        # u_z直接揺動（oscillation_amplitude > 0）は prescribed_dofs が変わるため
        # 2フェーズ方式。θ_y揺動（amplitude=0）のみ統合モード。
        # 統合モード: 曲げ+揺動を1回のソルバーで実行。
        # 2フェーズ方式はUL参照配置不整合（CR梁のf_int=0問題）で使用不可。
        _combined_mode = _n_osc > 0 and cfg.loading_mode != "moment"
        if _combined_mode:
            _frac_bend = cfg.n_cycles / (cfg.n_cycles + _n_osc)
            t_total = t_cycle * (cfg.n_cycles + _n_osc)
            # 揺動振幅: oscillation_amplitude > 0 なら先端変位→θ変換
            # 先端変位 δ ≈ R * Δθ where R = 1/κ = strand_length / (κ*strand_length)
            if cfg.oscillation_amplitude > 0.0:
                _R_bend = strand_length / bending_angle if bending_angle > 1e-10 else strand_length
                _theta_osc_amp = cfg.oscillation_amplitude / _R_bend
            else:
                _theta_osc_amp = bending_angle  # 従来: 全曲げ角度で揺動

            def _combined_prescribed_func(frac: float) -> np.ndarray:
                if frac <= _frac_bend:
                    # 曲げフェーズ: θ = bending_angle * (frac / frac_bend)
                    _theta = bending_angle * (frac / _frac_bend)
                else:
                    # 揺動フェーズ: 1-cos 波形でC1連続（transition微分=0）
                    # sin(2πnt) は transition で dθ/dt ≠ 0 → 不連続。
                    # 代わりに (1-cos(2πnt))/2 を使い、最初の半周期で振幅まで
                    # 滑らかに立ち上がる。
                    _osc_frac = (frac - _frac_bend) / (1.0 - _frac_bend)
                    _phase = 2.0 * math.pi * _n_osc * _osc_frac
                    _delta_theta = _theta_osc_amp * math.sin(_phase)
                    # 最初の1/4周期にランプ適用（C1連続化）
                    _ramp_end = 0.25 / _n_osc  # 最初のn_oscの1/4周期
                    if _osc_frac < _ramp_end:
                        # sin波にハーフcos窓: 0→1の滑らかなランプ
                        _w = 0.5 * (1.0 - math.cos(math.pi * _osc_frac / _ramp_end))
                        _delta_theta *= _w
                    _theta = bending_angle + _delta_theta
                return np.full(len(prescribed_dofs_arr), _theta)

            print(
                f"  統合モード: 曲げ(frac<{_frac_bend:.3f}) + "
                f"θ揺動±{math.degrees(_theta_osc_amp):.1f}°×{_n_osc}cyc"
            )
            if cfg.oscillation_amplitude > 0.0:
                print(
                    f"  先端変位±{cfg.oscillation_amplitude:.1f}mm → "
                    f"θ振幅±{math.degrees(_theta_osc_amp):.1f}°"
                )
        else:
            _frac_bend = 1.0
            _combined_prescribed_func = None
            t_total = t_cycle * cfg.n_cycles

        _total_cycles = (cfg.n_cycles + _n_osc) if _combined_mode else cfg.n_cycles
        dt_initial = t_total / (cfg.n_increments_per_cycle * _total_cycles)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs_arr,
            prescribed_values=prescribed_values_arr,
            f_ext_total=f_ext,
            mpc_transform=None,  # MPC不使用
            prescribed_func=_combined_prescribed_func,
        )

        # ── 接触設定 ──
        _smoothing_delta = (
            cfg.smoothing_delta if cfg.smoothing_delta > 0.0 else 1000.0 / cfg.wire_radius
        )
        _cutback_depth = 256.0 if _combined_mode else 64.0
        contact_config = _ContactConfigInput(
            beam_E=cfg.E,
            beam_I=sec_Iy,
            mu=cfg.mu,
            adaptive_timestepping=True,
            dt_min_fraction=dt_initial / (t_total * _cutback_depth),
            dt_max_fraction=dt_initial / t_total,
            exclude_same_strand=cfg.exclude_same_strand,
            exclude_end_elements=cfg.exclude_end_elements,
            smoothing_delta=_smoothing_delta,
            huber_delta_h=cfg.huber_delta_h,
            coating_stiffness=cfg.coating_stiffness,
            coating_damping=cfg.coating_damping,
            coating_mu=cfg.coating_mu,
            coating_k_t_ratio=cfg.coating_k_t_ratio,
            coating_thickness=cfg.coating_thickness,
            coating_barrier=cfg.coating_barrier,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=cfg.k_pen,
            mu=cfg.mu,
        )

        # ── ソルバー実行: 曲げフェーズ ──
        _callbacks = AssembleCallbacks(
            assemble_tangent=assembler.assemble_tangent,
            assemble_internal_force=assembler.assemble_internal_force,
            ul_assembler=assembler,
        )

        # チェックポイント復元時は曲げフェーズをスキップ
        if cfg.resume_checkpoint and cfg.n_oscillation_cycles > 0:
            import pickle as _pickle

            with open(cfg.resume_checkpoint, "rb") as _f:
                _ckpt = _pickle.load(_f)
            _u_bend = _ckpt["state"].u.copy()
            _vel_bend = _ckpt["time_vel"]
            _acc_bend = _ckpt["time_acc"]
            # ULアセンブラの完全状態復元（自工程保証:
            # coords_ref + R_ref + _u_total_accum の3点セットで参照配置を正確復元）
            if "ul_coords_ref" in _ckpt:
                if hasattr(assembler, "coords_ref"):
                    assembler.coords_ref[:] = _ckpt["ul_coords_ref"]
                if hasattr(assembler, "R_ref"):
                    assembler.R_ref[:] = _ckpt["ul_R_ref"]
                if hasattr(assembler, "_u_total_accum"):
                    assembler._u_total_accum[:] = _ckpt["ul_u_total_accum"]
                print("  [RESUME] ULアセンブラ完全復元: coords_ref + R_ref + u_accum")
            elif "ul_u_total_accum" in _ckpt:
                # 後方互換: coords_ref/R_ref なしの旧checkpoint
                _u_accum = _ckpt["ul_u_total_accum"]
                if hasattr(assembler, "_u_total_accum"):
                    assembler._u_total_accum[:] = _u_accum
                print(
                    f"  [RESUME] UL累積変位のみ復元（旧形式）: "
                    f"||u_accum||={np.linalg.norm(_u_accum):.4e}"
                )
            elif hasattr(assembler, "_u_total_accum"):
                assembler._u_total_accum[:] = _u_bend
            # 接触マネージャ状態の復元（自工程保証:
            # 各インクリメント完了時の保存状態で次のインクリメントをクリーンに開始）
            if "manager_pairs" in _ckpt:
                _restored_pairs = _ckpt["manager_pairs"]
                _restored_config = _ckpt.get("manager_config", contact_config)
                _restored_conn = _ckpt.get("connectivity", None)
                manager = _ContactManagerInput(
                    pairs=_restored_pairs,
                    config=_restored_config,
                    connectivity=_restored_conn,
                )
                contact_setup = ContactSetupData(
                    manager=manager,
                    k_pen=cfg.k_pen,
                    mu=cfg.mu,
                )
                _n_active = sum(
                    1 for p in _restored_pairs if hasattr(p, "state") and p.state.p_n > 0
                )
                print(
                    f"  [RESUME] 接触マネージャ復元: "
                    f"{len(_restored_pairs)} pairs ({_n_active} active)"
                )
            print(f"  [RESUME] 曲げcheckpointロード: ||u||={np.linalg.norm(_u_bend):.4e}")
            solver_result_bend = None
        else:
            # status-333: M-κ追跡用曲率関数
            # κ = θ / L（曲げフェーズ: θ = bending_angle * frac / frac_bend）
            _strand_length = strand_length
            _ba = bending_angle
            _fb = _frac_bend

            def _mk_curvature_func(frac: float) -> float:
                if _combined_prescribed_func is not None:
                    _theta = float(_combined_prescribed_func(frac)[0])
                else:
                    _theta = _ba * frac
                return _theta / _strand_length

            # 曲げフェーズ実行
            solver_input = ContactFrictionInputData(
                mesh=mesh,
                boundary=boundary,
                contact=contact_setup,
                callbacks=_callbacks,
                mass_matrix=M,
                dt_physical=t_total,
                rho_inf=cfg.rho_inf,
                max_nr_attempts=cfg.max_nr_attempts,
                tol_force=cfg.tol_force,
                max_increments=cfg.max_increments,
                tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
                kc_component_fd_diagnostic=cfg.kc_component_fd_diagnostic,
                du_norm_cap=cfg.du_norm_cap,
                penalty_exponent=cfg.penalty_exponent,
                checkpoint_path=cfg.checkpoint_path,
                checkpoint_frac=0.99,
                track_mk=cfg.track_contact_mk,
                mk_moment_dofs=tuple(prescribed_dofs_list),
                mk_curvature_func=_mk_curvature_func if cfg.track_contact_mk else None,
                track_contact_pairs=cfg.track_contact_pairs,
                # 接触 backtracking line search（status-362）
                contact_backtracking_enabled=cfg.contact_backtracking_enabled,
                contact_backtracking_max_steps=cfg.contact_backtracking_max_steps,
                contact_backtracking_active_flip_ratio=cfg.contact_backtracking_active_flip_ratio,
                contact_backtracking_active_flip_threshold=cfg.contact_backtracking_active_flip_threshold,
                contact_backtracking_residual_ratio=cfg.contact_backtracking_residual_ratio,
                contact_backtracking_alpha_decay=cfg.contact_backtracking_alpha_decay,
                contact_backtracking_min_alpha=cfg.contact_backtracking_min_alpha,
                contact_backtracking_mixed_only=cfg.contact_backtracking_mixed_only,
                contact_backtracking_rate_threshold=cfg.contact_backtracking_rate_threshold,
                # 接触法線減衰 escape hatch（status-366 Phase 2、候補 (e)）
                contact_damping_coefficient=cfg.contact_damping_coefficient,
                contact_damping_energy_budget_ratio=cfg.contact_damping_energy_budget_ratio,
                # チャタリング検知→接触凍結モード（status-368 候補 (d)）
                chattering_freeze_enabled=cfg.chattering_freeze_enabled,
                chattering_freeze_max_cycles=cfg.chattering_freeze_max_cycles,
                chattering_freeze_nr_max=cfg.chattering_freeze_nr_max,
                chattering_freeze_tol_factor=cfg.chattering_freeze_tol_factor,
                # active 履歴平滑化（status-371 候補 (g1)）
                active_ema_alpha=cfg.active_ema_alpha,
                # pair-wise relaxation（status-374/375 候補 (g3) Phase 2）
                pairwise_freeze_enabled=cfg.pairwise_freeze_enabled,
                pairwise_freeze_flip_threshold=cfg.pairwise_freeze_flip_threshold,
                pairwise_freeze_skip_type_d=cfg.pairwise_freeze_skip_type_d,
                # Augmented Lagrangian 外側ループ（status-376）
                al_outer_enabled=cfg.al_outer_enabled,
                al_n_uzawa_max=cfg.al_n_uzawa_max,
                # 陽的中央差分時間積分（status-378 Phase 2）
                solver_mode=cfg.solver_mode,
            )
            solver_result_bend = ContactFrictionProcess().process(solver_input)
            _u_bend = solver_result_bend.u
            # status-299: 速度・加速度を揺動フェーズに引き継ぎ（慣性力不整合防止）
            _vel_bend = solver_result_bend.final_velocity
            _acc_bend = solver_result_bend.final_acceleration
            _frac_bend = (
                solver_result_bend.load_history[-1] if solver_result_bend.load_history else 0.0
            )
            print(
                f"  曲げフェーズ完了: frac={_frac_bend:.4f}, "
                f"incr={solver_result_bend.n_increments}, "
                f"cutback={solver_result_bend.n_cutbacks}"
            )

            # status-299: 接触マネージャ状態を揺動フェーズに引き継ぎ（旧2フェーズ用）
            if (
                not _combined_mode
                and solver_result_bend.final_contact_manager is not None
                and cfg.n_oscillation_cycles > 0
            ):
                _bend_mgr = solver_result_bend.final_contact_manager
                contact_setup = ContactSetupData(
                    manager=_bend_mgr,
                    k_pen=cfg.k_pen,
                    mu=cfg.mu,
                )
                _n_active = sum(
                    1
                    for p in getattr(_bend_mgr, "pairs", [])
                    if hasattr(p, "state") and p.state.p_n > 0
                )
                print(
                    f"  接触マネージャ引き継ぎ: "
                    f"{len(getattr(_bend_mgr, 'pairs', []))} pairs ({_n_active} active)"
                )

        # ── 揺動フェーズ ──
        # 統合モード（_combined_mode）の場合、曲げ+揺動は prescribed_func で
        # 1回のソルバーで実行済み。旧2フェーズ方式はフォールバック。
        if _combined_mode:
            solver_result = solver_result_bend
        elif cfg.n_oscillation_cycles > 0:
            _n_osc = cfg.n_oscillation_cycles

            if cfg.oscillation_amplitude > 0.0:
                # status-299: 先端横変位揺動（u_z ±amplitude）
                # 曲げ完了位置を中心にsin波で往復。
                # prescribed_dofs: u_z（揺動）+ θ_y（曲げ完了値で固定）
                _uz_dofs = [n * 6 + 2 for n in right_nodes]
                _theta_dofs = [n * 6 + 4 for n in right_nodes]
                _osc_prescribed_dofs = np.array(_uz_dofs + _theta_dofs, dtype=int)
                _uz_at_bend = np.array([float(_u_bend[d]) for d in _uz_dofs])
                _theta_at_bend = np.array([float(_u_bend[d]) for d in _theta_dofs])
                _n_uz = len(_uz_dofs)
                _osc_amplitude = cfg.oscillation_amplitude
                # prescribed_values: u_z=曲げ完了値（frac=1.0でsin=0）, θ_y=曲げ完了値
                _osc_prescribed_values = np.concatenate([_uz_at_bend, _theta_at_bend])

                def _oscillation_func(frac: float) -> np.ndarray:
                    # prescribed_func は state.u[prescribed_dofs] に書き込む絶対値。
                    # u_z: 曲げ完了位置 + sin波変位
                    # θ_y: 曲げ完了値で固定（安定性確保）
                    delta_uz = _osc_amplitude * math.sin(2.0 * math.pi * _n_osc * frac)
                    uz_vals = _uz_at_bend + delta_uz
                    return np.concatenate([uz_vals, _theta_at_bend])

                print(
                    f"  揺動フェーズ: 先端u_z横変位±{_osc_amplitude:.1f}mm, "
                    f"{_n_osc}サイクル（θ_y固定）"
                )
                _uz_str = ", ".join(f"{uz:.2f}" for uz in _uz_at_bend[:3])
                _uz_suffix = "..." if len(_uz_at_bend) > 3 else ""
                print(f"  曲げ完了時u_z: {_uz_str}{_uz_suffix}")
            else:
                # 従来: θ_y回転cos波揺動（status-286互換）
                _osc_prescribed_dofs = prescribed_dofs_arr
                _osc_prescribed_values = prescribed_values_arr
                _theta_at_ckpt = float(_u_bend[prescribed_dofs_arr[0]])
                _theta_amplitude = _theta_at_ckpt

                def _oscillation_func(frac: float) -> np.ndarray:
                    # cos(0)=1 → Δθ=0, cos(π)=-1 → Δθ=-2θ_ckpt
                    delta_theta = _theta_amplitude * (math.cos(2.0 * math.pi * _n_osc * frac) - 1.0)
                    return np.full(len(_osc_prescribed_dofs), delta_theta)

            # 揺動フェーズの時間パラメータ
            t_osc = t_cycle * cfg.n_oscillation_cycles

            boundary_osc = BoundaryData(
                fixed_dofs=fixed_dofs_arr,
                prescribed_dofs=_osc_prescribed_dofs,
                prescribed_values=_osc_prescribed_values,
                f_ext_total=f_ext,
                mpc_transform=None,
                prescribed_func=_oscillation_func,
            )

            # status-299: 揺動フェーズ用に新しいアセンブラを作成。
            # 曲げ中のupdate_reference()で座標が変わったアセンブラを再利用すると
            # u_incr=0 → f_int=0 になり応力状態が消失する。
            # 新品アセンブラ（原点メッシュ）+ u0=_u_bend（全累積変位）で
            # f_int(u_bend) が正しく計算される。
            _beam_result_osc = ULCRBeamAssemblerProcess().process(
                ULCRBeamAssemblerInput(
                    node_coords=strand_coords,
                    connectivity=strand_conn,
                    E=cfg.E,
                    G=G,
                    A=sec["A"],
                    Iy=sec["Iy"],
                    Iz=sec["Iz"],
                    J=sec["J"],
                    kappa_y=sec["kappa"],
                    kappa_z=sec["kappa"],
                )
            )
            _asm_osc = _beam_result_osc.assembler
            _callbacks_osc = AssembleCallbacks(
                assemble_tangent=_asm_osc.assemble_tangent,
                assemble_internal_force=_asm_osc.assemble_internal_force,
                ul_assembler=_asm_osc,
            )

            # status-333: 揺動フェーズのM-κ曲率関数
            if cfg.track_contact_mk:
                if cfg.oscillation_amplitude > 0.0:
                    # u_z揺動: θ_y固定 → κ = θ_bend / L
                    _osc_kappa_base = bending_angle / strand_length

                    def _mk_curvature_func_osc(frac: float) -> float:
                        return _osc_kappa_base

                    _osc_mk_dofs = tuple(d for d in _osc_prescribed_dofs if d % 6 == 4)
                else:
                    # θ_y揺動: κ = θ(frac) / L
                    _sl = strand_length

                    def _mk_curvature_func_osc(frac: float) -> float:
                        return float(_oscillation_func(frac)[0]) / _sl

                    _osc_mk_dofs = tuple(int(d) for d in _osc_prescribed_dofs)
            else:
                _mk_curvature_func_osc = None
                _osc_mk_dofs = ()

            solver_input_osc = ContactFrictionInputData(
                mesh=mesh,
                boundary=boundary_osc,
                contact=contact_setup,
                callbacks=_callbacks_osc,
                u0=_u_bend,
                mass_matrix=M,
                dt_physical=t_osc,
                rho_inf=cfg.rho_inf,
                velocity=_vel_bend,
                acceleration=_acc_bend,
                max_nr_attempts=cfg.max_nr_attempts,
                tol_force=cfg.tol_force,
                max_increments=cfg.max_increments,
                tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
                kc_component_fd_diagnostic=cfg.kc_component_fd_diagnostic,
                du_norm_cap=cfg.du_norm_cap,
                penalty_exponent=cfg.penalty_exponent,
                skip_initial_detection=False,
                track_mk=cfg.track_contact_mk,
                mk_moment_dofs=_osc_mk_dofs,
                mk_curvature_func=_mk_curvature_func_osc,
                track_contact_pairs=cfg.track_contact_pairs,
                # 接触 backtracking line search（status-362）
                contact_backtracking_enabled=cfg.contact_backtracking_enabled,
                contact_backtracking_active_flip_ratio=cfg.contact_backtracking_active_flip_ratio,
                contact_backtracking_max_steps=cfg.contact_backtracking_max_steps,
                contact_backtracking_active_flip_threshold=cfg.contact_backtracking_active_flip_threshold,
                contact_backtracking_residual_ratio=cfg.contact_backtracking_residual_ratio,
                contact_backtracking_alpha_decay=cfg.contact_backtracking_alpha_decay,
                contact_backtracking_min_alpha=cfg.contact_backtracking_min_alpha,
                contact_backtracking_mixed_only=cfg.contact_backtracking_mixed_only,
                contact_backtracking_rate_threshold=cfg.contact_backtracking_rate_threshold,
                # 接触法線減衰 escape hatch（status-366 Phase 2、候補 (e)）
                contact_damping_coefficient=cfg.contact_damping_coefficient,
                contact_damping_energy_budget_ratio=cfg.contact_damping_energy_budget_ratio,
                # チャタリング検知→接触凍結モード（status-368 候補 (d)）
                chattering_freeze_enabled=cfg.chattering_freeze_enabled,
                chattering_freeze_max_cycles=cfg.chattering_freeze_max_cycles,
                chattering_freeze_nr_max=cfg.chattering_freeze_nr_max,
                chattering_freeze_tol_factor=cfg.chattering_freeze_tol_factor,
                # active 履歴平滑化（status-371 候補 (g1)）
                active_ema_alpha=cfg.active_ema_alpha,
                # pair-wise relaxation（status-374/375 候補 (g3) Phase 2）
                pairwise_freeze_enabled=cfg.pairwise_freeze_enabled,
                pairwise_freeze_flip_threshold=cfg.pairwise_freeze_flip_threshold,
                pairwise_freeze_skip_type_d=cfg.pairwise_freeze_skip_type_d,
                # Augmented Lagrangian 外側ループ（status-376）
                al_outer_enabled=cfg.al_outer_enabled,
                al_n_uzawa_max=cfg.al_n_uzawa_max,
                # 陽的中央差分時間積分（status-378 Phase 2）
                solver_mode=cfg.solver_mode,
            )
            solver_result = ContactFrictionProcess().process(solver_input_osc)
        else:
            solver_result = solver_result_bend

        return StrandBendingOscillationResult(
            solver_result=solver_result,
            mesh=mesh,
            n_ref_nodes=0,
            n_strand_nodes=n_strand_nodes,
            total_ndof=ndof,
            bending_angle=bending_angle,
        )

    def _process_fiber_beam(
        self,
        cfg: StrandBendingOscillationConfig,
    ) -> StrandBendingOscillationResult:
        """ファイバー梁モードで撚線曲げ揺動を実行.

        status-330 / Phase F5: 素線メッシュの代わりに1本のファイバー梁として解く。
        内部摩擦はセクションファイバー積分で処理。接触計算なし。

        パイプライン:
        1. 直線梁メッシュ生成（単一梁、n_elements 個の2ノード要素）
        2. ULCRFiberBeamAssemblerProcess でファイバー梁アセンブラ構築
        3. 左端固定、右端θ_y処方の境界条件
        4. _static_nr_solve で静的NR求解
        """
        # ── 1. 直線梁メッシュ生成 ──
        strand_length = cfg.pitch_length * cfg.n_pitches
        n_elems = int(cfg.n_elements_per_pitch * cfg.n_pitches)
        n_nodes = n_elems + 1

        # x軸方向の直線梁
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 0] = np.linspace(0.0, strand_length, n_nodes)
        connectivity = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])

        # メッシュデータ（radii=0: 接触なし）
        mesh = MeshData(
            node_coords=node_coords,
            connectivity=connectivity,
            radii=0.0,
            n_strands=1,
            strand_ids=np.zeros(n_elems, dtype=int),
        )

        # ── 2. ファイバー断面生成 ──
        diameter = cfg.wire_radius * 2.0
        if cfg.fiber_section_type == "polar":
            section = CircularFiberSection.polar(
                diameter, n_radial=cfg.fiber_n_fiber, n_theta=cfg.fiber_n_theta
            )
        else:
            section = CircularFiberSection.strip(diameter, n_fiber=cfg.fiber_n_fiber)

        # 材料則: 外部指定 or デフォルト弾性
        material = cfg.fiber_material if cfg.fiber_material is not None else Elastic1D(E=cfg.E)

        # ── 3. ファイバー梁アセンブラ構築 ──
        G = cfg.E / (2.0 * (1.0 + cfg.nu))
        sec = _circle_section(diameter, cfg.nu)

        beam_result = ULCRFiberBeamAssemblerProcess().process(
            ULCRFiberBeamAssemblerInput(
                node_coords=node_coords,
                connectivity=connectivity,
                section=section,
                material=material,
                G=G,
                J=sec["J"],
                kappa_y=sec["kappa"],
                kappa_z=sec["kappa"],
            )
        )
        assembler = beam_result.assembler
        ndof = n_nodes * 6

        # ── 4. 境界条件 ──
        # 左端（node 0）: 全6DOF固定
        fixed_dofs: set[int] = set()
        for k in range(6):
            fixed_dofs.add(k)

        # 右端（node n_nodes-1）: θ_x 固定
        right_node = n_nodes - 1
        fixed_dofs.add(right_node * 6 + 3)  # θ_x

        # x-z 面曲げ: 全ノードの u_y, θ_z を拘束
        # strip 断面では EI_z=0（z座標ゼロ）のため u_y/θ_z の剛性がゼロ。
        # polar 断面でも面外拘束が必要（x-z 面曲げ問題）。
        for n in range(n_nodes):
            fixed_dofs.add(n * 6 + 1)  # u_y
            fixed_dofs.add(n * 6 + 5)  # θ_z

        # 曲げ角度 = κ * L
        bending_angle = cfg.bending_curvature * strand_length

        # 処方変位: 右端の θ_y（x-z面曲げ回転）
        prescribed_dof = right_node * 6 + 4  # θ_y
        prescribed_dofs = np.array([prescribed_dof], dtype=int)
        prescribed_values = np.array([bending_angle])

        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

        # ── 5. 静的NRソルバー実行 ──
        # 非線形材料ではTL定式化（update_referenceを呼ばない）。
        # CR梁ULのf_int=0問題を回避（CLAUDE.md参照）。
        is_nonlinear = not isinstance(material, Elastic1D)

        # サイクル荷重パターン（status-331: 散逸エネルギー検証用）
        # n_cycles=1: 単調負荷 0→κ_max
        # n_cycles=2: 負荷→除荷 0→κ_max→0（三角波）
        # n_cycles>2: 負荷→除荷→反転... の多サイクル
        n_half_cycles = cfg.n_cycles
        n_increments = cfg.n_increments_per_cycle * n_half_cycles

        if n_half_cycles >= 2 and is_nonlinear:
            # サイクル荷重: prescribed_func で三角波パターンを定義
            def _prescribed_func(frac: float) -> np.ndarray:
                # 三角波: frac 0→1 で n_half_cycles 半サイクル
                # 半サイクル内の位相 [0, 1]
                phase = frac * n_half_cycles
                half_idx = int(phase)
                t = phase - half_idx
                if half_idx >= n_half_cycles:
                    half_idx = n_half_cycles - 1
                    t = 1.0
                # 偶数半サイクル: 0→κ_max（正方向）
                # 奇数半サイクル: κ_max→0（除荷）
                if half_idx % 2 == 0:
                    kappa_frac = t
                else:
                    kappa_frac = 1.0 - t
                return np.array([bending_angle * kappa_frac])

            def _curvature_func(frac: float) -> float:
                vals = _prescribed_func(frac)
                return float(vals[0]) / strand_length

            prescribed_func_arg = _prescribed_func
            mk_curvature_func_arg = _curvature_func
        else:
            # 単調負荷: prescribed_func なし（従来の frac * prescribed_values）
            prescribed_func_arg = None

            def _curvature_func_mono(frac: float) -> float:
                return cfg.bending_curvature * frac

            mk_curvature_func_arg = _curvature_func_mono

        # M-κ 追跡: 非線形材料時のみ有効化
        track_mk = is_nonlinear
        solver_result = _static_nr_solve(
            assembler=assembler,
            ndof=ndof,
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs,
            prescribed_values=prescribed_values,
            f_ext_total=np.zeros(ndof),
            n_increments=n_increments,
            max_nr=cfg.max_nr_attempts,
            tol=cfg.tol_force,
            use_ul=not is_nonlinear,
            prescribed_func=prescribed_func_arg,
            track_mk=track_mk,
            mk_curvature_func=mk_curvature_func_arg,
            mk_moment_dof=prescribed_dof,
        )

        return StrandBendingOscillationResult(
            solver_result=solver_result,
            mesh=mesh,
            n_ref_nodes=0,
            n_strand_nodes=n_nodes,
            total_ndof=ndof,
            bending_angle=bending_angle,
        )
