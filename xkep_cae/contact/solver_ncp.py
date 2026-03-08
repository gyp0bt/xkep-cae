"""Semi-smooth Newton ソルバー（NCP ベース接触） — Phase C6-L3.

Outer loop を廃止し、変位 u とラグランジュ乗数 λ を同時に更新する。

AL-NCP ハイブリッド + Alart-Curnier 鞍点定式化:
  接触力: p_n_i = max(0, λ_i + k_pen * (-g_i))  （AL 形式）
  NCP 条件 (Alart-Curnier):
    - Active  (λ + k_pen*(-g) > 0): C_i = k_pen * g_i  → g = 0 を強制
    - Inactive:                      C_i = λ_i          → λ = 0 を強制

鞍点系 (Saddle-point):
  [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
  [G_A      0    ] [Δλ_A]   [-g_active]

  K_eff = K_T + k_pen * G_A^T * G_A

Schur complement on constraint space (n_active × n_active) で解く。
k_pen が正則化項として機能し、K_eff を確実に正定値に保つ。

参考文献:
- Alart, Curnier (1991): CMAME
- Hüeber, Stadler, Wohlmuth (2008): "A primal-dual active set algorithm
  for three-dimensional contact problems with Coulomb friction"

設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §5
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.assembly import (
    _contact_dofs,
    _contact_shape_vector,
    _contact_tangent_shape_vector,
    compute_contact_force,
    compute_contact_stiffness,
)
from xkep_cae.contact.graph import ContactGraphHistory, snapshot_contact_graph
from xkep_cae.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
)
from xkep_cae.contact.mortar import (
    build_mortar_system,
    compute_mortar_contact_force,
    compute_mortar_p_n,
    identify_mortar_nodes,
)
from xkep_cae.contact.pair import ContactManager, ContactStatus


@dataclass
class ConvergenceDiagnostics:
    """収束失敗時の標準化診断情報.

    Newton反復の履歴情報を収集し、収束失敗の原因特定を支援する。

    Attributes:
        step: 荷重ステップ番号
        load_frac: 荷重分率
        res_history: 各反復の力残差ノルム比 ||R_u||/||f||
        ncp_history: 各反復のNCP残差ノルム ||C_n||
        ncp_t_history: 各反復の摩擦NCP残差ノルム ||C_t||
        n_active_history: 各反復のNCP活性ペア数
        du_norm_history: 各反復の変位増分ノルム ||du||
        max_du_dof_history: 各反復で最大の変位増分を持つDOFインデックス
        condition_number: 最終反復の条件数推定（計算した場合）
    """

    step: int = 0
    load_frac: float = 0.0
    res_history: list[float] = field(default_factory=list)
    ncp_history: list[float] = field(default_factory=list)
    ncp_t_history: list[float] = field(default_factory=list)
    n_active_history: list[int] = field(default_factory=list)
    du_norm_history: list[float] = field(default_factory=list)
    max_du_dof_history: list[int] = field(default_factory=list)
    condition_number: float | None = None

    def format_report(self, max_iter: int = 50) -> str:
        """診断レポートの文字列を生成する.

        Returns:
            人間が読みやすいフォーマットの診断レポート
        """
        lines = [
            "=" * 60,
            "  NCP Solver Convergence Diagnostics",
            "=" * 60,
            f"  Step: {self.step}, Load fraction: {self.load_frac:.6f}",
            f"  Iterations: {len(self.res_history)} / {max_iter}",
        ]

        if self.condition_number is not None:
            lines.append(f"  Condition number (est.): {self.condition_number:.2e}")

        # 残差推移テーブル
        n_iter = len(self.res_history)
        if n_iter > 0:
            lines.append("")
            lines.append(
                "  iter  ||R||/||f||   ||C_n||     ||C_t||     n_active  ||du||       max_du_dof"
            )
            lines.append("  " + "-" * 76)
            for i in range(n_iter):
                res = self.res_history[i] if i < len(self.res_history) else 0.0
                ncp = self.ncp_history[i] if i < len(self.ncp_history) else 0.0
                ncp_t = self.ncp_t_history[i] if i < len(self.ncp_t_history) else 0.0
                n_act = self.n_active_history[i] if i < len(self.n_active_history) else 0
                du = self.du_norm_history[i] if i < len(self.du_norm_history) else 0.0
                dof = self.max_du_dof_history[i] if i < len(self.max_du_dof_history) else -1
                lines.append(
                    f"  {i:4d}  {res:11.3e}  {ncp:11.3e}  {ncp_t:11.3e}  "
                    f"{n_act:8d}  {du:11.3e}  {dof:10d}"
                )

            # 収束率の推定
            if n_iter >= 3:
                ratios = []
                for i in range(2, n_iter):
                    prev = self.res_history[i - 1]
                    pprev = self.res_history[i - 2]
                    if pprev > 1e-30 and prev > 1e-30:
                        import math

                        log_denom = math.log(prev / pprev)
                        if abs(log_denom) > 1e-30:
                            r = math.log(self.res_history[i] / prev) / log_denom
                            ratios.append(r)
                if ratios:
                    avg_rate = sum(ratios) / len(ratios)
                    lines.append(f"\n  Convergence order (avg): {avg_rate:.2f}")
                    if avg_rate < 1.5:
                        lines.append(
                            "  WARNING: Sub-quadratic convergence — "
                            "tangent stiffness may be inaccurate"
                        )

            # 残差増大の検出
            if n_iter >= 2:
                max_growth = max(
                    self.res_history[i] / max(self.res_history[i - 1], 1e-30)
                    for i in range(1, n_iter)
                )
                if max_growth > 10.0:
                    lines.append(
                        f"  WARNING: Max residual growth ratio = {max_growth:.1f}x — "
                        "possible divergence or Newton overshoot"
                    )

            # アクティブセットの変動検出
            if n_iter >= 2 and any(n > 0 for n in self.n_active_history):
                changes = sum(
                    1
                    for i in range(1, len(self.n_active_history))
                    if self.n_active_history[i] != self.n_active_history[i - 1]
                )
                if changes > n_iter * 0.5:
                    lines.append(
                        f"  WARNING: Active set changed in {changes}/{n_iter - 1} iterations — "
                        "possible chattering"
                    )

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class NCPSolverInput:
    """NCP ベース接触解析の入力パラメータ.

    newton_raphson_contact_ncp の入力を構造化するデータクラス。
    全てのフィールドを明示的に設定するか、solver_input.to_kwargs() で
    既存の関数シグネチャに変換して使用する。

    用語:
        - increment: 荷重増分（adaptive_timestepping で自動制御）
        - iteration: Newton 反復（各 increment 内での非線形反復）

    Attributes:
        f_ext_total: (ndof,) 最終外荷重ベクトル
        fixed_dofs: 拘束DOFのインデックス配列
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径（スカラーまたは配列）
        max_iter: Newton 最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_ncp: NCP 残差の収束判定
        dt_initial_fraction: 初期荷重増分の分率（0=自動, >0 で明示指定）
        ul_assembler: Updated Lagrangian アセンブラ（大回転問題用）
    """

    f_ext_total: np.ndarray
    fixed_dofs: np.ndarray
    assemble_tangent: Callable
    assemble_internal_force: Callable
    manager: ContactManager
    node_coords_ref: np.ndarray
    connectivity: np.ndarray
    radii: np.ndarray | float
    max_iter: int = 50
    tol_force: float = 1e-8
    tol_disp: float = 1e-8
    tol_ncp: float = 1e-8
    dt_initial_fraction: float = 0.0
    show_progress: bool = True
    u0: np.ndarray | None = None
    ndof_per_node: int = 6
    broadphase_margin: float = 0.0
    broadphase_cell_size: float | None = None
    ncp_type: str = "fb"
    ncp_reg: float = 1e-12
    k_pen: float = 0.0
    f_ext_base: np.ndarray | None = None
    use_friction: bool = False
    mu: float | None = None
    mu_ramp_steps: int | None = None
    line_contact: bool = False
    n_gauss: int | None = None
    use_mortar: bool = False
    use_line_search: bool = True
    line_search_max_steps: int = 5
    modified_nr_threshold: int = 0
    prescribed_dofs: np.ndarray | None = None
    prescribed_values: np.ndarray | None = None
    adaptive_omega: bool = False
    omega_init: float = 1.0
    omega_min: float = 0.05
    omega_max: float = 1.0
    omega_shrink: float = 0.5
    omega_growth: float = 1.2
    du_norm_cap: float = 0.0
    dt_grow_iter_threshold: int = 0
    ul_assembler: object | None = None

    def solve(self) -> NCPSolveResult:
        """このパラメータで NCP ソルバーを実行する."""
        return newton_raphson_contact_ncp(**self.to_kwargs())

    def to_kwargs(self) -> dict:
        """既存の newton_raphson_contact_ncp 関数に渡す kwargs 辞書を生成."""
        import dataclasses

        kw = dataclasses.asdict(self)
        # Callable はシリアル化できないので直接設定
        kw["assemble_tangent"] = self.assemble_tangent
        kw["assemble_internal_force"] = self.assemble_internal_force
        kw["manager"] = self.manager
        kw["ul_assembler"] = self.ul_assembler
        kw["u0"] = self.u0
        kw["prescribed_dofs"] = self.prescribed_dofs
        kw["prescribed_values"] = self.prescribed_values
        kw["f_ext_base"] = self.f_ext_base
        kw["broadphase_cell_size"] = self.broadphase_cell_size
        kw["mu"] = self.mu
        kw["mu_ramp_steps"] = self.mu_ramp_steps
        kw["n_gauss"] = self.n_gauss
        # adaptive_timestepping は常に True（ソルバー統一後のデフォルト）
        kw["adaptive_timestepping"] = True
        # n_load_steps は deprecated だが内部計算用に 1 をセット
        kw["n_load_steps"] = None
        # deprecated params
        kw["max_step_cuts"] = 0
        kw["bisection_max_depth"] = 0
        kw["active_set_update_interval"] = 1
        return kw


@dataclass
class NCPSolveResult:
    """NCP ベース接触解析の結果.

    Attributes:
        u: (ndof,) 最終変位ベクトル
        lambdas: (n_pairs,) ラグランジュ乗数（全ペア、非活性は 0）
        converged: 収束したかどうか
        n_increments: 解析に使用された荷重増分数
        total_newton_iterations: 全増分の合計 Newton 反復回数
        n_active_final: 最終的な ACTIVE ペア数
        load_history: 各増分の荷重係数
        displacement_history: 各増分の変位
        contact_force_history: 各増分の接触力ノルム
        graph_history: 接触グラフの時系列
        diagnostics: 収束失敗時の診断情報（収束時はNone）
    """

    u: np.ndarray
    lambdas: np.ndarray
    converged: bool
    n_increments: int
    total_newton_iterations: int
    n_active_final: int
    load_history: list[float] = field(default_factory=list)
    displacement_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)
    graph_history: ContactGraphHistory = field(default_factory=ContactGraphHistory)
    diagnostics: ConvergenceDiagnostics | None = None

    @property
    def n_load_steps(self) -> int:
        """後方互換性のためのエイリアス。n_increments を参照。"""
        return self.n_increments


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


def _compute_contact_force_from_lambdas(
    manager: ContactManager,
    lambdas: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
    k_pen: float = 0.0,
) -> np.ndarray:
    """AL-NCP 接触力ベクトルを計算する.

    p_n_i = max(0, λ_i + k_pen * (-g_i))
    f_c = Σ p_n_i * g_shape_i

    Args:
        manager: 接触マネージャ
        lambdas: (n_pairs,) ラグランジュ乗数
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ正則化パラメータ

    Returns:
        f_c: (ndof_total,) 接触内力ベクトル
    """
    f_c = np.zeros(ndof_total)

    for i, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        lam_i = lambdas[i] if i < len(lambdas) else 0.0
        g_i = pair.state.gap
        p_n = max(0.0, lam_i + k_pen * (-g_i))

        if p_n <= 0.0:
            continue

        g_shape = _contact_shape_vector(pair)
        dofs = _contact_dofs(pair, ndof_per_node)
        for k in range(4):
            for d in range(3):
                local_idx = k * 3 + d
                global_idx = dofs[k * ndof_per_node + d]
                f_c[global_idx] += p_n * g_shape[local_idx]

    return f_c


def _compute_friction_forces_ncp(
    manager: ContactManager,
    lambdas: np.ndarray,
    u: np.ndarray,
    u_ref: np.ndarray,
    node_coords_ref: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
    k_pen: float = 0.0,
    mu: float = 0.3,
    mu_ramp_counter: int = 0,
    mu_ramp_steps: int = 0,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """NCP ソルバー用の摩擦力計算.

    各 ACTIVE ペアで:
    1. NCP 法線力 p_n = max(0, λ + k_pen*(-g)) を設定
    2. 接線変位を計算
    3. Coulomb return mapping を実行
    4. 摩擦力ベクトルをアセンブリ

    Args:
        manager: 接触マネージャ
        lambdas: (n_pairs,) ラグランジュ乗数
        u: (ndof,) 現在の変位
        u_ref: (ndof,) 参照変位（前ステップ収束解）
        node_coords_ref: (n_nodes, 3) 参照座標
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ正則化パラメータ
        mu: 摩擦係数
        mu_ramp_counter: μランプカウンタ
        mu_ramp_steps: μランプ総ステップ数

    Returns:
        f_friction: (ndof_total,) 摩擦内力ベクトル
        friction_tangents: {pair_idx: D_t (2,2)} 摩擦接線剛性マップ
    """
    mu_eff = compute_mu_effective(mu, mu_ramp_counter, mu_ramp_steps)
    f_friction = np.zeros(ndof_total)
    friction_tangents: dict[int, np.ndarray] = {}

    for i, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        lam_i = lambdas[i] if i < len(lambdas) else 0.0
        g_i = pair.state.gap
        p_n = max(0.0, lam_i + k_pen * (-g_i))
        pair.state.p_n = p_n

        if p_n <= 0.0 or mu_eff <= 0.0:
            continue

        # ペナルティ剛性の初期化（未設定時）
        if pair.state.k_pen <= 0.0:
            pair.state.k_pen = k_pen
            pair.state.k_t = k_pen * manager.config.k_t_ratio

        # 接線変位
        delta_ut = compute_tangential_displacement(pair, u, u_ref, node_coords_ref, ndof_per_node)

        # Coulomb return mapping
        q = friction_return_mapping(pair, delta_ut, mu_eff)

        if float(np.linalg.norm(q)) < 1e-30:
            continue

        # 摩擦力アセンブリ
        dofs = _contact_dofs(pair, ndof_per_node)
        for axis in range(2):
            if abs(q[axis]) < 1e-30:
                continue
            g_t = _contact_tangent_shape_vector(pair, axis)
            for k in range(4):
                for d in range(3):
                    f_friction[dofs[k * ndof_per_node + d]] += q[axis] * g_t[k * 3 + d]

        # 摩擦接線剛性
        D_t = friction_tangent_2x2(pair, mu_eff)
        friction_tangents[i] = D_t

    return f_friction, friction_tangents


def _build_friction_stiffness(
    manager: ContactManager,
    friction_tangents: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """摩擦接線剛性行列のみを組み立てる（法線剛性を含まない）.

    NCP ソルバーでは法線剛性は鞍点系 (k_pen * G_A^T G_A) で処理されるため、
    compute_contact_stiffness を使うと法線剛性が二重カウントされる。
    この関数は摩擦接線剛性のみを構築する。

    K_f = Σ D_t[a1,a2] * g_t[a1] * g_t[a2]^T
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        if pair_idx not in friction_tangents:
            continue

        D_t = friction_tangents[pair_idx]
        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for k, node in enumerate(nodes):
            for d in range(3):
                gdofs[k * 3 + d] = node * ndof_per_node + d

        g_t = [
            _contact_tangent_shape_vector(pair, 0),
            _contact_tangent_shape_vector(pair, 1),
        ]
        for a1 in range(2):
            for a2 in range(2):
                d_val = D_t[a1, a2]
                if abs(d_val) < 1e-30:
                    continue
                for i in range(12):
                    for j in range(12):
                        val = d_val * g_t[a1][i] * g_t[a2][j]
                        if abs(val) > 1e-30:
                            rows.append(gdofs[i])
                            cols.append(gdofs[j])
                            data.append(val)

    if len(data) == 0:
        return sp.csr_matrix((ndof_total, ndof_total))

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(ndof_total, ndof_total),
    ).tocsr()


def _build_constraint_jacobian(
    manager: ContactManager,
    ndof_total: int,
    ndof_per_node: int = 6,
) -> tuple[sp.csr_matrix, list[int]]:
    """制約ヤコビアン G = ∂g_n/∂u を構築する."""
    active_indices = []
    rows = []
    cols = []
    vals = []

    row_idx = 0
    for i, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        active_indices.append(i)
        s = pair.state.s
        t = pair.state.t
        normal = pair.state.normal

        dofs = _contact_dofs(pair, ndof_per_node)
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]
        for k in range(4):
            for d in range(3):
                global_dof = dofs[k * ndof_per_node + d]
                val = coeffs[k] * normal[d]
                if abs(val) > 1e-30:
                    rows.append(row_idx)
                    cols.append(global_dof)
                    vals.append(val)

        row_idx += 1

    n_active = len(active_indices)
    if n_active == 0:
        return sp.csr_matrix((0, ndof_total)), active_indices

    G = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_active, ndof_total),
    ).tocsr()
    return G, active_indices


def _build_tangential_constraint_jacobian(
    manager: ContactManager,
    active_indices: list[int],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """接線方向の制約ヤコビアン G_t を構築する.

    G_t は (2*n_active, ndof) の行列で、各アクティブペアに対して
    2行（t1方向, t2方向）を持つ。

    G_t[2*j,   :] = 接線変位の t1 成分の u 微分
    G_t[2*j+1, :] = 接線変位の t2 成分の u 微分

    接線相対変位: Δu_t = (u_B - u_A) · [t1, t2]
    → coeffs = [-(1-s), -s, (1-t), t]  (B-A方向)

    Args:
        manager: 接触マネージャ
        active_indices: アクティブペアインデックスリスト
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        G_t: (2*n_active, ndof_total) 接線制約ヤコビアン
    """
    n_active = len(active_indices)
    if n_active == 0:
        return sp.csr_matrix((0, ndof_total))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for j, pair_idx in enumerate(active_indices):
        pair = manager.pairs[pair_idx]
        s = pair.state.s
        t = pair.state.t
        t1 = pair.state.tangent1
        t2 = pair.state.tangent2

        dofs = _contact_dofs(pair, ndof_per_node)
        # B - A 方向: coeffs = [-(1-s), -s, (1-t), t]
        coeffs = [-(1.0 - s), -s, (1.0 - t), t]

        for axis, ti in enumerate([t1, t2]):
            row = 2 * j + axis
            for k in range(4):
                for d in range(3):
                    global_dof = dofs[k * ndof_per_node + d]
                    val = coeffs[k] * ti[d]
                    if abs(val) > 1e-30:
                        rows.append(row)
                        cols.append(global_dof)
                        vals.append(val)

    G_t = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(2 * n_active, ndof_total),
    ).tocsr()
    return G_t


def _solve_linear_system(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    mode: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
    gmres_dof_threshold: int = 2000,
) -> np.ndarray:
    """線形連立方程式を解く.

    mode="auto" では DOF 閾値に基づいて直接法と反復法を自動選択する:
      - DOF < gmres_dof_threshold: 直接法（spsolve）
      - DOF >= gmres_dof_threshold: 反復法（GMRES + ILU 前処理）
    """
    import warnings

    import scipy.sparse.linalg as spla

    if mode == "direct":
        return spla.spsolve(K, rhs)

    if mode == "iterative":
        K_csc = K.tocsc()
        try:
            ilu = spla.spilu(K_csc, drop_tol=ilu_drop_tol)
            M = spla.LinearOperator(K.shape, ilu.solve)
        except RuntimeError:
            M = None
        _restart_k = min(max(30, K.shape[0] // 10), 200)
        x, info = spla.gmres(
            K, rhs, M=M, atol=iterative_tol, restart=_restart_k, maxiter=max(500, K.shape[0])
        )
        if info != 0:
            x = spla.spsolve(K, rhs)
        return x

    # auto: DOF 閾値ベースで選択
    n = K.shape[0]
    if n >= gmres_dof_threshold:
        return _solve_linear_system(
            K,
            rhs,
            mode="iterative",
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )

    # 小規模: direct → iterative fallback
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = spla.spsolve(K, rhs)
    for w in caught:
        if "MatrixRankWarning" in str(w.category.__name__) or "singular" in str(w.message).lower():
            return _solve_linear_system(
                K, rhs, mode="iterative", iterative_tol=iterative_tol, ilu_drop_tol=ilu_drop_tol
            )
    if not np.all(np.isfinite(x)):
        return _solve_linear_system(
            K, rhs, mode="iterative", iterative_tol=iterative_tol, ilu_drop_tol=ilu_drop_tol
        )
    return x


def _apply_bc(K_lil, rhs, fixed_dofs):
    """境界条件を適用する（in-place 変更）.

    LIL/CSR/CSC いずれの形式でも動作する。
    """
    for dof in fixed_dofs:
        K_lil[dof, :] = 0.0
        K_lil[:, dof] = 0.0
        K_lil[dof, dof] = 1.0
        rhs[dof] = 0.0


def _apply_bc_csr(K, rhs, fixed_dofs):
    """境界条件を適用する（CSR/CSC ベース高速版、コピーを返す）."""
    from xkep_cae.contact.bc_utils import apply_bc_fast

    return apply_bc_fast(K, rhs, fixed_dofs)


def _solve_saddle_point_gmres(
    K_T: sp.csr_matrix,
    G_A: sp.csr_matrix,
    k_pen: float,
    R_u: np.ndarray,
    g_active: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
    use_amg: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系をブロック前処理付き GMRES で解く.

    拡大系:
      [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
      [G_A      0    ] [Δλ_A]   [-g_active]

    ブロック対角前処理:
      P = [K_eff^{-1}   0      ]    K_eff^{-1} ≈ ILU
          [0            S_d^{-1}]    S_d ≈ diag(G_A * K_eff^{-1} * G_A^T)

    n_active が大きい場合に直接 Schur complement より効率的。
    K_eff^{-1} の (1 + n_active) 回の求解を GMRES 反復に置き換える。

    設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §6

    Args:
        K_T: 構造接線剛性
        G_A: (n_active, ndof) NCP アクティブ制約ヤコビアン
        k_pen: ペナルティ正則化パラメータ
        R_u: (ndof,) 力残差
        g_active: (n_active,) アクティブペアのギャップ
        fixed_dofs: 拘束 DOF
        iterative_tol: GMRES 収束判定
        ilu_drop_tol: ILU drop tolerance

    Returns:
        (du, dlam_A): 変位増分, アクティブ乗数増分
    """
    import scipy.sparse.linalg as spla

    ndof = K_T.shape[0]
    n_active = G_A.shape[0]
    n_total = ndof + n_active

    # K_eff = K_T + k_pen * G_A^T * G_A
    K_eff = K_T + k_pen * (G_A.T @ G_A)

    # BC 適用（高速版）
    rhs_u = -R_u.copy()
    K_bc_csr, rhs_u = _apply_bc_csr(K_eff, rhs_u, fixed_dofs)

    # G_A の BC 処理（拘束 DOF 列をゼロに: indptr/data 直接操作）
    G_A_csc = G_A.tocsc().copy()
    for dof in fixed_dofs:
        start = G_A_csc.indptr[dof]
        end = G_A_csc.indptr[dof + 1]
        G_A_csc.data[start:end] = 0.0
    G_A_bc_csr = G_A_csc.tocsr()
    G_A_bc_T = G_A_bc_csr.T.tocsr()

    # --- 前処理構築（AMG or ILU） ---
    ilu = None
    amg_solver = None
    if use_amg:
        try:
            import pyamg

            amg_solver = pyamg.smoothed_aggregation_solver(
                K_bc_csr,
                max_coarse=50,
                max_levels=10,
            )
        except Exception:
            pass  # AMG失敗時はILUにフォールバック

    if amg_solver is None:
        # ILU 前処理構築（適応 drop_tol）
        _ilu_tol = ilu_drop_tol
        for _ilu_attempt in range(4):
            try:
                ilu = spla.spilu(K_bc_csr.tocsc(), drop_tol=_ilu_tol)
                break
            except RuntimeError:
                _ilu_tol *= 10.0  # drop_tol を緩和して再試行

    # --- K_eff^{-1} の近似ソルバー ---
    def _k_eff_solve(b):
        """K_eff の近似逆行列適用（AMG or ILU）."""
        if amg_solver is not None:
            return amg_solver.solve(b, tol=1e-8, maxiter=20)
        elif ilu is not None:
            return ilu.solve(b)
        else:
            return b

    # --- Schur 補集合の対角近似: S_ii = G_A[i,:] * K_eff^{-1} * G_A[i,:]^T ---
    _schur_reg = 1e-12
    s_diag = np.ones(n_active) * _schur_reg
    if amg_solver is not None or ilu is not None:
        G_A_bc_dense = G_A_bc_csr.toarray()  # (n_active, ndof)
        try:
            if amg_solver is not None:
                V = np.zeros((ndof, n_active))
                for i in range(n_active):
                    V[:, i] = amg_solver.solve(G_A_bc_dense[i, :], tol=1e-8, maxiter=20)
            else:
                # バッチソルブ: (ndof, n_active)
                V = ilu.solve(G_A_bc_dense.T)
            s_diag = np.einsum("ij,ji->i", G_A_bc_dense, V)
            # 対角成分が負または微小な場合の安全下限（最大値の1e-10）
            _s_max = np.max(np.abs(s_diag)) if n_active > 0 else 1.0
            _safe_floor = max(1e-12, _s_max * 1e-10)
            s_diag = np.maximum(s_diag, _safe_floor)
        except Exception:
            # フォールバック: 行ごとのソルブ
            for i in range(n_active):
                g_row = G_A_bc_dense[i, :]
                v = _k_eff_solve(g_row)
                s_diag[i] = max(g_row @ v, 1e-12)
    else:
        # 前処理なし: 粗い近似
        s_diag[:] = 1.0 / max(k_pen, 1e-12)

    # --- 拡大系行列-ベクトル積 ---
    def matvec(x):
        x_u = x[:ndof]
        x_lam = x[ndof:]
        y = np.zeros(n_total)
        y[:ndof] = K_bc_csr @ x_u - G_A_bc_T @ x_lam
        y[ndof:] = G_A_bc_csr @ x_u
        # BC 行: 拘束 DOF は identity
        for dof in fixed_dofs:
            y[dof] = x_u[dof]
        return y

    A_op = spla.LinearOperator((n_total, n_total), matvec=matvec)

    # --- ブロック対角前処理演算子 ---
    def precond(x):
        y = np.zeros(n_total)
        y[:ndof] = _k_eff_solve(x[:ndof])
        y[ndof:] = x[ndof:] / s_diag
        return y

    M = spla.LinearOperator((n_total, n_total), matvec=precond)

    # --- RHS ---
    rhs = np.zeros(n_total)
    rhs[:ndof] = rhs_u
    rhs[ndof:] = -g_active

    # --- GMRES（restart 適応チューニング） ---
    # restart: 小〜中規模は ndof/10、大規模でも最大200で打ち切り
    _restart = min(max(30, n_total // 10), 200)
    x, info = spla.gmres(
        A_op,
        rhs,
        M=M,
        atol=iterative_tol,
        restart=_restart,
        maxiter=max(500, n_total),
    )

    if info != 0:
        # フォールバック: 直接 Schur complement
        return _solve_saddle_point_direct(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            linear_solver="auto",
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )

    du = x[:ndof]
    dlam_A = x[ndof:]
    return du, dlam_A


def _solve_saddle_point_direct(
    K_T: sp.csr_matrix,
    G_A: sp.csr_matrix,
    k_pen: float,
    R_u: np.ndarray,
    g_active: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    linear_solver: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系を直接 Schur complement で解く.

    鞍点系:
      [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
      [G_A      0    ] [Δλ_A]   [-g_active]

    K_eff = K_T + k_pen * G_A^T * G_A

    制約空間 Schur complement (n_active × n_active) で解く:
      1. v0 = K_eff^{-1} * (-R_u)        (非制約解)
      2. V  = K_eff^{-1} * G_A^T         (制約感度)
      3. S  = G_A * V                     (制約 Schur 補集合)
      4. Δλ = S^{-1} * (-g - G_A * v0)
      5. Δu = v0 + V * Δλ

    Args:
        K_T: 構造接線剛性
        G_A: (n_active, ndof) NCP アクティブ制約ヤコビアン
        k_pen: ペナルティ正則化パラメータ
        R_u: (ndof,) 力残差
        g_active: (n_active,) アクティブペアのギャップ
        fixed_dofs: 拘束 DOF
        linear_solver: 線形ソルバーモード
        iterative_tol: 反復ソルバー許容値
        ilu_drop_tol: ILU drop tolerance

    Returns:
        (du, dlam_A): 変位増分, アクティブ乗数増分
    """
    ndof = K_T.shape[0]
    n_active = G_A.shape[0]
    solve_kw = dict(mode=linear_solver, iterative_tol=iterative_tol, ilu_drop_tol=ilu_drop_tol)

    # K_eff = K_T + k_pen * G_A^T * G_A （ペナルティ正則化）
    K_eff = K_T + k_pen * (G_A.T @ G_A)

    # BC 適用（高速版）
    rhs_u = -R_u.copy()
    K_bc_csr, rhs_u = _apply_bc_csr(K_eff, rhs_u, fixed_dofs)

    # Step 1: v0 = K_eff^{-1} * (-R_u)
    v0 = _solve_linear_system(K_bc_csr, rhs_u, **solve_kw)

    # Step 2: V = K_eff^{-1} * G_A^T  (ndof × n_active)
    G_A_T_dense = G_A.T.toarray()  # ndof × n_active
    G_A_T_bc = G_A_T_dense.copy()
    for dof in fixed_dofs:
        G_A_T_bc[dof, :] = 0.0

    V = np.zeros((ndof, n_active))
    if n_active > 4:
        # 並列ソルブ（ThreadPoolExecutor）: 各 RHS は独立
        from concurrent.futures import ThreadPoolExecutor

        def _solve_col(j):
            return j, _solve_linear_system(K_bc_csr, G_A_T_bc[:, j], **solve_kw)

        with ThreadPoolExecutor() as pool:
            for j, vj in pool.map(lambda j: _solve_col(j), range(n_active)):
                V[:, j] = vj
    else:
        for j in range(n_active):
            V[:, j] = _solve_linear_system(K_bc_csr, G_A_T_bc[:, j], **solve_kw)

    # Step 3: S = G_A * V  (n_active × n_active — 制約 Schur 補集合)
    G_A_dense = G_A.toarray()
    S = G_A_dense @ V

    # 適応正則化（S の対角が微小な場合のみ補強）
    S_diag = np.diag(S).copy()
    _s_diag_max = np.max(np.abs(S_diag)) if n_active > 0 else 1.0
    _reg_floor = max(1e-12, _s_diag_max * 1e-10)
    reg_needed = np.maximum(_reg_floor - S_diag, 0.0)
    S += np.diag(reg_needed + 1e-12)

    # Step 4: rhs_S = -g_active - G_A * v0
    rhs_S = -g_active - G_A_dense @ v0

    # Δλ_A を解く（n_active × n_active 密行列、通常は極めて小さい）
    dlam_A = np.linalg.solve(S, rhs_S)

    # Step 5: Δu = v0 + V * Δλ_A
    du = v0 + V @ dlam_A

    return du, dlam_A


def _solve_augmented_friction_system(
    K_T: sp.csr_matrix,
    G_n: sp.csr_matrix,
    G_t: sp.csr_matrix,
    k_pen: float,
    k_t: float,
    R_u: np.ndarray,
    C_n: np.ndarray,
    C_t: np.ndarray,
    J_blocks: dict,
    fixed_dofs: np.ndarray,
    n_ncp_active: int,
    ncp_active_mask: np.ndarray,
    active_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alart-Curnier 摩擦拡大鞍点系を解く.

    拡大系:
      [K_eff    -G_n_A^T  -G_t_A^T] [Δu    ]   [-R_u]
      [J_n_u    J_n_n      0      ] [Δλ_n_A] = [-C_n]
      [J_t_u    J_t_n      J_t_t  ] [Δλ_t_A]   [-C_t]

    K_eff = K_T + k_pen * G_n_A^T G_n_A

    各ペアの J ブロックは状態（active/inactive, stick/slip）に応じて異なる。

    Args:
        K_T: 構造接線剛性
        G_n: (n_geom_active, ndof) 法線制約ヤコビアン（全幾何アクティブ）
        G_t: (2*n_geom_active, ndof) 接線制約ヤコビアン（全幾何アクティブ）
        k_pen: 法線ペナルティ剛性
        k_t: 接線ペナルティ剛性
        R_u: (ndof,) 力残差
        C_n: (n_geom_active,) 法線 NCP 残差
        C_t: (2*n_geom_active,) 接線 NCP 残差
        J_blocks: 各ペアの Jacobian ブロック辞書
        fixed_dofs: 拘束 DOF
        n_ncp_active: NCP アクティブペア数
        ncp_active_mask: NCP アクティブマスク
        active_indices: アクティブペアインデックス

    Returns:
        (du, dlam_n, dlam_t): 変位増分, 法線乗数増分, 接線乗数増分
    """
    import scipy.sparse.linalg as spla

    ndof = K_T.shape[0]

    # NCP アクティブ行のみ抽出
    if n_ncp_active > 0:
        active_rows_n = np.where(ncp_active_mask)[0]
        G_n_A = G_n[active_rows_n, :]
    else:
        G_n_A = sp.csr_matrix((0, ndof))
        active_rows_n = np.array([], dtype=int)

    # 接線行: NCP アクティブに対応する 2*n 行
    if n_ncp_active > 0:
        t_rows = []
        for r in active_rows_n:
            t_rows.extend([2 * r, 2 * r + 1])
        t_rows = np.array(t_rows, dtype=int)
        G_t_A = G_t[t_rows, :]
    else:
        G_t_A = sp.csr_matrix((0, ndof))
        t_rows = np.array([], dtype=int)

    n_n = n_ncp_active
    n_t = 2 * n_ncp_active
    n_total = ndof + n_n + n_t

    # K_eff = K_T + k_pen * G_n_A^T G_n_A
    if n_n > 0:
        K_eff = K_T + k_pen * (G_n_A.T @ G_n_A)
    else:
        K_eff = K_T.copy()

    # --- 拡大系をブロック行列で構築 ---
    # J_n_u, J_n_n, J_t_u, J_t_n, J_t_t を密行列で構築
    J_n_u = np.zeros((n_n, ndof))
    J_n_n = np.zeros((n_n, n_n))
    J_t_u = np.zeros((n_t, ndof))
    J_t_n = np.zeros((n_t, n_n))
    J_t_t = np.zeros((n_t, n_t))

    G_n_A_dense = G_n_A.toarray() if n_n > 0 else np.zeros((0, ndof))
    G_t_A_dense = G_t_A.toarray() if n_t > 0 else np.zeros((0, ndof))

    for j_local in range(n_ncp_active):
        j_geom = int(active_rows_n[j_local])
        pair_idx = active_indices[j_geom]
        jb = J_blocks.get(pair_idx)
        if jb is None:
            # デフォルト: 法線 active, stick
            # C_n = k_pen * g → J_n_u = k_pen * G_n
            J_n_u[j_local, :] = k_pen * G_n_A_dense[j_local, :]
            # C_t = -k_t * Δu_t → J_t_u = -k_t * G_t, J_t_t = 0
            J_t_u[2 * j_local, :] = -k_t * G_t_A_dense[2 * j_local, :]
            J_t_u[2 * j_local + 1, :] = -k_t * G_t_A_dense[2 * j_local + 1, :]
            continue

        mode = jb["mode"]  # "active_stick", "active_slip", "inactive"

        if mode == "inactive":
            # C_n = λ_n → J_n_n = 1, C_t = λ_t → J_t_t = I
            J_n_n[j_local, j_local] = 1.0
            J_t_t[2 * j_local, 2 * j_local] = 1.0
            J_t_t[2 * j_local + 1, 2 * j_local + 1] = 1.0

        elif mode == "active_stick":
            # C_n = k_pen * g → J_n_u = k_pen * G_n
            J_n_u[j_local, :] = k_pen * G_n_A_dense[j_local, :]
            # C_t = -k_t * Δu_t → J_t_u = -k_t * G_t, J_t_t = 0
            J_t_u[2 * j_local, :] = -k_t * G_t_A_dense[2 * j_local, :]
            J_t_u[2 * j_local + 1, :] = -k_t * G_t_A_dense[2 * j_local + 1, :]

        elif mode == "active_slip":
            # 法線: C_n = k_pen * g → J_n_u = k_pen * G_n
            J_n_u[j_local, :] = k_pen * G_n_A_dense[j_local, :]

            # 接線 slip Jacobians
            # J_t_t = I - (μ*p_n/||λ̂_t||) * (I - q̂⊗q̂)
            J_t_t_local = jb["J_t_t"]  # (2,2)
            J_t_t[2 * j_local : 2 * j_local + 2, 2 * j_local : 2 * j_local + 2] = J_t_t_local

            # J_t_n = -μ * q̂  (2,1) — critical coupling term
            J_t_n_local = jb["J_t_n"]  # (2,)
            J_t_n[2 * j_local, j_local] = J_t_n_local[0]
            J_t_n[2 * j_local + 1, j_local] = J_t_n_local[1]

            # J_t_u: slip の変位微分
            # ∂C_t/∂u = μ*k_pen * outer(q̂, G_n) - μ*p_n*k_t/||λ̂_t|| * (I-q̂⊗q̂) @ G_t
            if "J_t_u" in jb:
                J_t_u[2 * j_local : 2 * j_local + 2, :] = jb["J_t_u"]  # (2, ndof)

    # C_n, C_t を NCP アクティブ行のみに絞る
    C_n_A = C_n[active_rows_n] if n_n > 0 else np.array([])
    C_t_A = C_t[t_rows] if n_t > 0 else np.array([])

    # 拡大系組立: sp.bmat で構築
    # [K_eff, -G_n_A^T, -G_t_A^T]   [-R_u ]
    # [J_n_u,  J_n_n,    0       ] = [-C_n_A]
    # [J_t_u,  J_t_n,    J_t_t   ]   [-C_t_A]
    # BC を K_eff に適用（高速版）
    rhs_u = -R_u.copy()
    K_eff_bc, rhs_u = _apply_bc_csr(K_eff, rhs_u, fixed_dofs)

    # G の BC 処理（拘束 DOF 列をゼロに）
    if n_n > 0:
        G_n_A_bc = G_n_A_dense.copy()
        for dof in fixed_dofs:
            G_n_A_bc[:, dof] = 0.0
            J_n_u[:, dof] = 0.0
    if n_t > 0:
        G_t_A_bc = G_t_A_dense.copy()
        for dof in fixed_dofs:
            G_t_A_bc[:, dof] = 0.0
            J_t_u[:, dof] = 0.0

    # 拡大行列を構築
    if n_n == 0 and n_t == 0:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                du = spla.spsolve(K_eff_bc, rhs_u)
            except spla.MatrixRankWarning:
                # 行列特異時: 対角正則化を追加
                K_reg = K_eff_bc + 1e-10 * sp.eye(K_eff_bc.shape[0], format="csc")
                du = spla.spsolve(K_reg, rhs_u)
        if not np.all(np.isfinite(du)):
            du = np.zeros(ndof)
        return du, np.array([]), np.array([])

    blocks = [[K_eff_bc, None, None], [None, None, None], [None, None, None]]

    if n_n > 0:
        blocks[0][1] = sp.csr_matrix(-G_n_A_bc.T)
        blocks[1][0] = sp.csr_matrix(J_n_u)
        blocks[1][1] = sp.csr_matrix(J_n_n)
    if n_t > 0:
        blocks[0][2] = sp.csr_matrix(-G_t_A_bc.T) if n_t > 0 else None
        blocks[2][0] = sp.csr_matrix(J_t_u)
        blocks[2][2] = sp.csr_matrix(J_t_t)
    if n_n > 0 and n_t > 0:
        blocks[1][2] = sp.csr_matrix((n_n, n_t))
        blocks[2][1] = sp.csr_matrix(J_t_n)

    # None ブロックをゼロ行列で埋める
    sizes = [ndof, n_n, n_t]
    for i_b in range(3):
        for j_b in range(3):
            if blocks[i_b][j_b] is None:
                if sizes[i_b] > 0 and sizes[j_b] > 0:
                    blocks[i_b][j_b] = sp.csr_matrix((sizes[i_b], sizes[j_b]))

    # サイズ 0 のブロック行/列を除外
    active_block_rows = [i_b for i_b in range(3) if sizes[i_b] > 0]
    filtered_blocks = []
    for i_b in active_block_rows:
        row_blocks = []
        for j_b in active_block_rows:
            row_blocks.append(blocks[i_b][j_b])
        filtered_blocks.append(row_blocks)

    A = sp.bmat(filtered_blocks, format="csc")

    # RHS
    rhs_parts = []
    if ndof > 0:
        rhs_parts.append(rhs_u)
    if n_n > 0:
        rhs_parts.append(-C_n_A)
    if n_t > 0:
        rhs_parts.append(-C_t_A)
    rhs = np.concatenate(rhs_parts)

    # 解く
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        try:
            x = spla.spsolve(A, rhs)
        except spla.MatrixRankWarning:
            # 行列特異時: 対角正則化を追加してリトライ
            A_reg = A + 1e-8 * sp.eye(A.shape[0], format="csc")
            x = spla.spsolve(A_reg, rhs)
    if not np.all(np.isfinite(x)):
        # フォールバック: GMRES
        x_gm, info = spla.gmres(A.tocsr(), rhs, atol=1e-10, maxiter=max(500, n_total))
        if info == 0 and np.all(np.isfinite(x_gm)):
            x = x_gm

    du = x[:ndof]
    offset = ndof
    dlam_n = x[offset : offset + n_n] if n_n > 0 else np.array([])
    offset += n_n
    dlam_t = x[offset : offset + n_t] if n_t > 0 else np.array([])

    return du, dlam_n, dlam_t


def _solve_saddle_point_contact(
    K_T: sp.csr_matrix,
    G_A: sp.csr_matrix,
    k_pen: float,
    R_u: np.ndarray,
    g_active: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    linear_solver: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
    use_block_preconditioner: bool = False,
    gmres_dof_threshold: int = 2000,
    use_amg: bool = False,
    residual_scaling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系を解く（直接法 or ブロック前処理 GMRES）.

    use_block_preconditioner=False (デフォルト):
      制約空間 Schur complement で直接解法。n_active が小さい場合に高速。

    use_block_preconditioner=True:
      ブロック対角前処理付き GMRES。n_active が大きい場合に効率的。
      K_eff^{-1} を ILU/AMG で近似し、Schur 補集合の対角近似を前処理に使用。

    linear_solver="auto" かつ ndof >= gmres_dof_threshold の場合、
    ブロック前処理 GMRES を自動選択する。

    Args:
        K_T: 構造接線剛性
        G_A: (n_active, ndof) NCP アクティブ制約ヤコビアン
        k_pen: ペナルティ正則化パラメータ
        R_u: (ndof,) 力残差
        g_active: (n_active,) アクティブペアのギャップ
        fixed_dofs: 拘束 DOF
        linear_solver: 線形ソルバーモード
        iterative_tol: 反復ソルバー許容値
        ilu_drop_tol: ILU drop tolerance
        use_block_preconditioner: ブロック前処理 GMRES の使用（Phase C6-L4）
        gmres_dof_threshold: DOF 閾値。この値以上で自動的に反復法を選択
        use_amg: PyAMG SA前処理を使用（S3改良7）

    Returns:
        (du, dlam_A): 変位増分, アクティブ乗数増分
    """
    n_active = G_A.shape[0]
    ndof = K_T.shape[0]

    # auto モードで DOF 閾値を超えた場合、ブロック前処理 GMRES を自動選択
    auto_block = linear_solver == "auto" and ndof >= gmres_dof_threshold and n_active > 0

    if n_active == 0:
        rhs = -R_u.copy()
        K_bc_csr, rhs = _apply_bc_csr(K_T, rhs, fixed_dofs)
        solve_kw = dict(
            mode=linear_solver,
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
            gmres_dof_threshold=gmres_dof_threshold,
        )
        du = _solve_linear_system(K_bc_csr, rhs, **solve_kw)
        return du, np.array([])

    # --- S3改良10: 対角スケーリング前処理 ---
    D_scale = None
    if residual_scaling and ndof > 0:
        K_diag = np.abs(K_T.diagonal())
        K_diag = np.maximum(K_diag, 1e-30)
        D_scale = 1.0 / np.sqrt(K_diag)
        # K̃ = D K D, R̃ = D R, G̃ = G D
        D_sp = sp.diags(D_scale)
        K_T = D_sp @ K_T @ D_sp
        R_u = D_scale * R_u
        G_A = G_A @ D_sp

    if use_block_preconditioner or auto_block:
        du, dlam = _solve_saddle_point_gmres(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
            use_amg=use_amg,
        )
    else:
        du, dlam = _solve_saddle_point_direct(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            linear_solver=linear_solver,
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )

    # スケーリングの逆変換
    if D_scale is not None:
        du = D_scale * du

    return du, dlam


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


def newton_raphson_contact_ncp(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    core_radii: np.ndarray | float | None = None,
    n_load_steps: int | None = None,
    max_iter: int = 50,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_ncp: float = 1e-8,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
    ncp_type: str = "fb",
    ncp_reg: float = 1e-12,
    k_pen: float = 0.0,
    f_ext_base: np.ndarray | None = None,
    use_friction: bool = False,
    mu: float | None = None,
    mu_ramp_steps: int | None = None,
    line_contact: bool = False,
    n_gauss: int | None = None,
    use_mortar: bool = False,
    # --- 収束安定化パラメータ（S3 統合版） ---
    use_line_search: bool = True,
    line_search_max_steps: int = 5,
    max_step_cuts: int = 0,
    modified_nr_threshold: int = 0,
    prescribed_dofs: np.ndarray | None = None,
    prescribed_values: np.ndarray | None = None,
    # --- レガシー互換パラメータ ---
    adaptive_omega: bool = False,
    omega_init: float = 1.0,
    omega_min: float = 0.05,
    omega_max: float = 1.0,
    omega_shrink: float = 0.5,
    omega_growth: float = 1.2,
    bisection_max_depth: int = 0,
    active_set_update_interval: int = 1,
    du_norm_cap: float = 0.0,
    adaptive_timestepping: bool = True,
    dt_initial_fraction: float = 0.0,
    dt_grow_iter_threshold: int = 0,
    # --- Updated Lagrangian 統合 ---
    ul_assembler: object | None = None,
) -> NCPSolveResult:
    """Semi-smooth Newton 法による接触解析（AL-NCP + 鞍点定式化）.

    Outer loop 不要。u と λ を同時に更新。
    摩擦有効時は法線 NCP + 摩擦 return mapping のハイブリッド方式。
    line_contact 有効時は Gauss 積分で接触力・剛性を評価。

    Alart-Curnier NCP + 鞍点系:
      Active set: A = {i : λ_i + k_pen*(-g_i) > 0}
      Active pairs → g = 0 を制約として鞍点系で解く
      Inactive pairs → λ = 0 に設定

    摩擦（use_friction=True 時）:
      各 Newton 反復で Coulomb return mapping を実行し、
      摩擦力を残差に、摩擦接線剛性を K_T に加算する。
      法線方向は NCP で Outer loop 不要、摩擦は return mapping で処理。

    line contact（line_contact=True 時）:
      法線力と法線剛性を Gauss 積分で評価する。
      NCP 活性セット判定は代表点ギャップ（PtP）を使用し、
      力・剛性評価のみ line-to-line を使用するハイブリッド方式。

    Args:
        f_ext_total: (ndof,) 最終外荷重
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径（被膜込み）
        core_radii: 芯線半径（被膜なし）。None の場合 radii と同一（被膜なし）。
            被膜接触モデル（coating_stiffness > 0）使用時に指定する。
        n_load_steps: [DEPRECATED] dt_initial_fraction を使用。
            指定時は dt_initial_fraction = 1/n_load_steps に自動変換される。
        max_iter: Newton 最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_ncp: NCP 残差の収束判定
        show_progress: 進捗表示
        u0: 初期変位
        ndof_per_node: 1節点あたりの DOF 数
        broadphase_margin: broadphase 探索マージン
        broadphase_cell_size: broadphase セルサイズ
        ncp_type: NCP 関数の種類（"fb" or "min"）— 収束モニタリング用
        ncp_reg: FB 関数の正則化パラメータ
        k_pen: ペナルティ正則化パラメータ。0 の場合は
            manager.config.k_pen_scale を使用。
        f_ext_base: (ndof,) ベース外荷重
        use_friction: Coulomb 摩擦の有効化（NCP + return mapping ハイブリッド）
        mu: 摩擦係数（None なら config.mu）
        mu_ramp_steps: μランプ（None なら config.mu_ramp_steps）
        line_contact: Line-to-line Gauss 積分の有効化（None なら config.line_contact）
        n_gauss: Gauss 積分点数（None なら config.n_gauss）
        use_line_search: バックトラッキング line search の有効化
        line_search_max_steps: line search の最大ステップ数
        max_step_cuts: [DEPRECATED] adaptive_timestepping を使用してください。
            非ゼロ指定時は自動的に adaptive_timestepping=True に変換されます。
        modified_nr_threshold: Modified NR 切替閾値。N>0 の場合、
            N 反復後に接線剛性を凍結（CR梁の振動発散抑制）。0=常に Full NR。
        prescribed_dofs: 処方変位DOFのインデックス配列（変位制御用）。
            各ステップで u[prescribed_dofs] = load_frac * prescribed_values を設定。
        prescribed_values: 処方変位の目標値（load_frac=1.0 での値）。
        adaptive_omega: 適応的緩和係数の有効化（レガシー）
        bisection_max_depth: [DEPRECATED] adaptive_timestepping を使用してください。

    Returns:
        NCPSolveResult
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    # --- n_load_steps → dt_initial_fraction 自動変換 ---
    # n_load_steps は deprecated。adaptive_timestepping の初期増分として変換する。
    if n_load_steps is not None:
        import warnings

        warnings.warn(
            "n_load_steps は非推奨です。"
            "dt_initial_fraction を使用してください。"
            "n_load_steps は dt_initial_fraction = 1.0 / n_load_steps に自動変換されます。",
            DeprecationWarning,
            stacklevel=2,
        )
        if dt_initial_fraction <= 0.0 and n_load_steps > 0:
            dt_initial_fraction = 1.0 / n_load_steps
    else:
        n_load_steps = 1  # 内部計算用のデフォルト

    # 処方変位の設定
    _prescribed_dofs = (
        np.asarray(prescribed_dofs, dtype=int)
        if prescribed_dofs is not None
        else np.array([], dtype=int)
    )
    _prescribed_values = (
        np.asarray(prescribed_values, dtype=float)
        if prescribed_values is not None
        else np.array([])
    )
    has_prescribed = len(_prescribed_dofs) > 0
    if has_prescribed:
        fixed_dofs = np.unique(np.concatenate([fixed_dofs, _prescribed_dofs]))

    # bisection_max_depth / max_step_cuts → adaptive_timestepping へ統合
    # DEPRECATED: ステップ二分法は適応時間増分制御に統合済み
    _max_step_cuts = max(max_step_cuts, bisection_max_depth)
    if _max_step_cuts > 0 and not adaptive_timestepping:
        import warnings

        warnings.warn(
            "max_step_cuts/bisection_max_depth は非推奨です。"
            "adaptive_timestepping=True を使用してください。"
            "自動的に adaptive_timestepping を有効化します。",
            DeprecationWarning,
            stacklevel=2,
        )
        adaptive_timestepping = True

    # k_pen の決定（S3改良9: beam_ei/ea_l 自動推定をNCPソルバーに導入）
    if k_pen <= 0.0:
        if manager.config.k_pen_mode in ("beam_ei", "ea_l"):
            # 代表要素長の推定
            _L_elems = []
            for elem in connectivity:
                n1, n2 = int(elem[0]), int(elem[1])
                dxyz = node_coords_ref[n2] - node_coords_ref[n1]
                _L_elems.append(float(np.linalg.norm(dxyz)))
            _L_avg = float(np.mean(_L_elems)) if _L_elems else 1.0
            _L_avg = max(_L_avg, 1e-30)

            if manager.config.k_pen_mode == "beam_ei":
                from xkep_cae.contact.law_normal import auto_beam_penalty_stiffness

                k_pen = auto_beam_penalty_stiffness(
                    manager.config.beam_E,
                    manager.config.beam_I,
                    _L_avg,
                    n_contact_pairs=max(1, manager.n_pairs),
                    scale=manager.config.k_pen_scale,
                    scaling=manager.config.k_pen_scaling,
                )
            else:
                from xkep_cae.contact.law_normal import auto_penalty_stiffness

                k_pen = auto_penalty_stiffness(
                    manager.config.beam_E,
                    manager.config.beam_A,
                    _L_avg,
                    scale=manager.config.k_pen_scale,
                )
            if show_progress:
                print(
                    f"  NCP auto k_pen ({manager.config.k_pen_mode}): "
                    f"k_pen={k_pen:.2e}, L_avg={_L_avg:.4f}"
                )
        else:
            k_pen = manager.config.k_pen_scale

    # --- k_pen continuation（S3改良8）---
    _k_pen_target = k_pen
    _k_pen_cont = manager.config.k_pen_continuation
    _k_pen_cont_steps = manager.config.k_pen_continuation_steps
    if _k_pen_cont:
        k_pen = _k_pen_target * manager.config.k_pen_continuation_start

    # 摩擦設定の解決
    _use_friction = use_friction or manager.config.use_friction
    _mu = mu if mu is not None else manager.config.mu
    _mu_ramp_steps = mu_ramp_steps if mu_ramp_steps is not None else manager.config.mu_ramp_steps

    # line contact 設定の解決
    _line_contact = line_contact or manager.config.line_contact
    _n_gauss = n_gauss if n_gauss is not None else manager.config.n_gauss

    # Mortar 設定の解決（line contact 必須）
    _use_mortar = (use_mortar or manager.config.use_mortar) and _line_contact

    # λ の初期化
    n_pairs = manager.n_pairs
    lam_all = np.zeros(n_pairs)
    # 接線乗数 λ_t（Alart-Curnier 摩擦用: 各ペア 2 成分）
    lam_t_all = np.zeros((n_pairs, 2))

    # Mortar 乗数（Mortar 有効時のみ使用）
    lam_mortar: np.ndarray | None = None
    mortar_nodes: list[int] = []
    g_mortar: np.ndarray | None = None

    # 線形ソルバー設定
    linear_solver_mode = manager.config.linear_solver
    iterative_tol_cfg = manager.config.iterative_tol
    ilu_drop_tol_cfg = manager.config.ilu_drop_tol
    use_block_preconditioner = manager.config.ncp_block_preconditioner
    gmres_dof_threshold_cfg = manager.config.gmres_dof_threshold

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    graph_history = ContactGraphHistory()
    total_newton = 0

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    # 変位制御時（f_ext=0）: 初回残差から動的にスケーリング
    _dynamic_ref = f_ext_ref_norm < 1e-30
    if _dynamic_ref:
        f_ext_ref_norm = 1.0  # 初回反復まで暫定値

    _f_ext_base = f_ext_base if f_ext_base is not None else np.zeros(ndof)

    # 摩擦用: 参照変位（前ステップ収束解）
    u_ref = u.copy()

    from collections import deque

    # --- 適応時間増分制御（S3改良6） ---
    _adaptive_dt = adaptive_timestepping or manager.config.adaptive_timestepping
    _dt_grow = manager.config.dt_grow_factor
    _dt_shrink = manager.config.dt_shrink_factor
    _dt_grow_iter = (
        dt_grow_iter_threshold
        if dt_grow_iter_threshold > 0
        else manager.config.dt_grow_iter_threshold
    )
    _dt_shrink_iter = manager.config.dt_shrink_iter_threshold
    _dt_contact_change_thr = manager.config.dt_contact_change_threshold
    _dt_min_frac = manager.config.dt_min_fraction
    _dt_max_frac = manager.config.dt_max_fraction

    # dt_initial_fraction: 初期ステップ幅の明示指定（n_load_steps=1 + adaptive の場合に有用）
    _dt_initial = dt_initial_fraction if dt_initial_fraction > 0.0 else 0.0

    # dt_min/max の自動計算: dt_initial_fraction 指定時はそれをベースにする
    _effective_n = max(n_load_steps, int(1.0 / _dt_initial) if _dt_initial > 0 else n_load_steps)
    if _dt_min_frac <= 0.0:
        _dt_min_frac = 1.0 / (_effective_n * 32)
    if _dt_max_frac <= 0.0:
        _dt_max_frac = min(8.0 / _effective_n, 1.0)

    # --- S3改良10: 残差スケーリング ---
    _residual_scaling = manager.config.residual_scaling

    # --- S3改良11: 接触力ランプ ---
    _contact_force_ramp = manager.config.contact_force_ramp
    _contact_force_ramp_iters = manager.config.contact_force_ramp_iters

    step_queue: deque[float] = deque()
    if _adaptive_dt:
        # 適応モード: 最初の目標分率のみ設定し、以降は動的に生成
        if _dt_initial > 0.0:
            _base_delta = _dt_initial
        else:
            _base_delta = 1.0 / n_load_steps
        step_queue.append(min(_base_delta, 1.0))
    else:
        for _s in range(1, n_load_steps + 1):
            step_queue.append(_s / n_load_steps)
    load_frac_prev = 0.0
    step_display = 0  # 表示用ステップカウンタ
    _prev_n_active = 0  # 前ステップのactive set数（適応Δt用）
    _consecutive_good = 0  # 連続良好ステップ数（安定化検出用）

    # --- Updated Lagrangian 統合 ---
    _ul = ul_assembler is not None
    _ul_frac_base = 0.0  # UL リセット時の荷重分率基準

    # チェックポイント（適応時間増分の不収束時ロールバック用）
    u_ckpt = u.copy()
    lam_ckpt = lam_all.copy()
    lam_t_ckpt = lam_t_all.copy()
    lam_mortar_ckpt = lam_mortar.copy() if lam_mortar is not None else None
    mortar_nodes_ckpt = list(mortar_nodes)
    u_ref_ckpt = u_ref.copy()
    _ul_frac_base_ckpt = _ul_frac_base
    if _ul:
        ul_assembler.checkpoint()

    # 接線予測子用: 前ステップの変位増分を記録
    u_prev_converged = u.copy()
    delta_frac_prev = 0.0  # 前ステップの荷重分率増分

    # --- 初期貫入チェック・位置調整 ---
    # 参照座標で初回ペア検出し、初期貫入の有無を確認する。
    manager.detect_candidates(
        node_coords_ref,
        connectivity,
        radii,
        margin=broadphase_margin,
        cell_size=broadphase_cell_size,
        core_radii=core_radii,
    )
    _pos_tol = manager.config.position_tolerance
    _adjust = manager.config.adjust_initial_penetration
    _use_coating = manager.config.coating_stiffness > 0.0

    # position_tolerance > 0: 小ギャップペアを接触位置に移動
    if _adjust and _pos_tol > 0.0:
        node_coords_ref, n_pen_fixed, n_gap_closed = manager.adjust_initial_positions(
            node_coords_ref, _pos_tol
        )
        if show_progress and (n_pen_fixed + n_gap_closed) > 0:
            print(
                f"  初期位置調整(adjust=yes, tol={_pos_tol * 1000:.3f}mm): "
                f"ギャップ閉鎖={n_gap_closed}ペア"
            )
        # 再検出
        manager.detect_candidates(
            node_coords_ref,
            connectivity,
            radii,
            margin=broadphase_margin,
            cell_size=broadphase_cell_size,
            core_radii=core_radii,
        )

    # 初期貫入チェック（status-137: gap_offset 廃止 → メッシュ側で防止）
    n_initial_pen = manager.check_initial_penetration(node_coords_ref)
    if n_initial_pen > 0 and not _use_coating:
        # 被膜モデル無効時: 初期貫入はメッシュ不備
        if not _adjust:
            raise ValueError(
                f"初期貫入が検出されました: {n_initial_pen}ペア。"
                f"adjust_initial_penetration=True で位置調整するか、"
                f"メッシュ生成時に coating_thickness を指定して "
                f"被膜厚分のgapを確保してください。"
            )
        # adjust=True かつ position_tolerance が設定済みの場合は
        # 上流の adjust_initial_positions() で処理済み。
        # 残存する微小貫入は弦近似誤差（16要素/ピッチで<2%）。
        if show_progress:
            print(f"  初期貫入検出: {n_initial_pen}ペア（弦近似誤差範囲内）")
    elif n_initial_pen > 0 and _use_coating:
        if show_progress:
            print(
                f"  被膜接触モデル有効: 初期被膜圧縮ペア={n_initial_pen}"
                f"（被膜弾性スプリングで処理）"
            )
    # λベクトルをペア数に合わせる
    if len(lam_all) < manager.n_pairs:
        lam_new = np.zeros(manager.n_pairs)
        lam_new[: len(lam_all)] = lam_all
        lam_all = lam_new
        lam_t_new = np.zeros((manager.n_pairs, 2))
        lam_t_new[: len(lam_t_all)] = lam_t_all
        lam_t_all = lam_t_new

    while step_queue:
        # 適応Δt成長で追い越された旧ターゲットをスキップ
        load_frac = step_queue[0]
        if load_frac <= load_frac_prev + 1e-15:
            step_queue.popleft()
            continue
        step_display += 1
        f_ext = _f_ext_base + load_frac * f_ext_total

        step_converged = False
        energy_ref = None  # エネルギー収束基準値

        # --- 適応的緩和係数の初期化 ---
        _omega = omega_init if adaptive_omega else 1.0
        _prev_merit = float("inf")
        _omega_stall_count = 0  # omega回復メカニズム用カウンタ

        # --- 接線予測子（前ステップの変位増分から外挿） ---
        delta_frac = load_frac - load_frac_prev
        if delta_frac_prev > 1e-30 and delta_frac > 1e-30:
            du_prev = u - u_prev_converged
            du_prev_norm = float(np.linalg.norm(du_prev))
            if du_prev_norm > 1e-30:
                ratio = min(delta_frac / delta_frac_prev, 2.0)
                u = u + ratio * du_prev

        # --- 処方変位の適用 ---
        if has_prescribed:
            u[_prescribed_dofs] = (load_frac - _ul_frac_base) * _prescribed_values

        # --- ステップ開始時: 候補検出 ---
        coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)
        manager.detect_candidates(
            coords_def,
            connectivity,
            radii,
            margin=broadphase_margin,
            cell_size=broadphase_cell_size,
            core_radii=core_radii,
        )

        # --- 段階的接触アクティベーション ---
        if manager.config.staged_activation_steps > 0:
            # n_load_steps は adaptive では不定のため、推定値を使用
            _est_total = max(n_load_steps, int(1.0 / max(_dt_min_frac, 0.01)))
            max_layer = manager.compute_active_layer_for_step(step_display, _est_total)
            manager.filter_pairs_by_layer(max_layer)

        manager.update_geometry(coords_def)

        # ペア数拡張（λウォームスタート対応）
        if len(lam_all) < manager.n_pairs:
            old_n = len(lam_all)
            lam_new = np.zeros(manager.n_pairs)
            lam_new[:old_n] = lam_all
            lam_t_new = np.zeros((manager.n_pairs, 2))
            lam_t_new[:old_n] = lam_t_all

            # 近傍ペアからλ初期推定（S3改良4）
            if manager.config.lambda_warmstart_neighbor and old_n > 0:
                # アクティブペアの平均λを新ペアの初期値にする
                active_lams = lam_new[:old_n][lam_new[:old_n] > 0.0]
                if len(active_lams) > 0:
                    lam_init = float(np.median(active_lams))
                    for idx in range(old_n, manager.n_pairs):
                        pair = manager.pairs[idx]
                        if pair.state.status != ContactStatus.INACTIVE and pair.state.gap < 0.0:
                            lam_new[idx] = lam_init

            lam_all = lam_new
            lam_t_all = lam_t_new

        K_T = None  # Modified NR 用: 接線凍結時に再利用

        # --- Active-set freeze 用の前回値 ---
        _frozen_ncp_active_mask: np.ndarray | None = None
        _frozen_ncp_mortar_active: np.ndarray | None = None

        # --- Active-set チャタリング抑制（S3改良5: 時間方向畳み込み） ---
        _chattering_window = manager.config.chattering_window
        _active_history: list[np.ndarray] = []  # 直近N反復のNCP active mask履歴

        # --- 収束診断データ収集 ---
        _diag = ConvergenceDiagnostics(step=step_display, load_frac=load_frac)

        for it in range(max_iter):
            total_newton += 1

            # 1. 幾何更新（s,t,normal を毎反復更新）
            coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)
            manager.update_geometry(coords_def, freeze_active_set=True)

            # 2. 制約ヤコビアン（幾何的 ACTIVE ペア）
            G_mat, active_indices = _build_constraint_jacobian(manager, ndof, ndof_per_node)
            n_geom_active = len(active_indices)

            gaps = np.array([manager.pairs[i].state.gap for i in active_indices])
            lams = np.array([lam_all[i] for i in active_indices])

            # 2m. Mortar 系の構築（Mortar 有効時）
            if _use_mortar and n_geom_active > 0:
                new_mortar_nodes, _node_to_pairs = identify_mortar_nodes(manager, active_indices)
                n_mortar = len(new_mortar_nodes)

                # Mortar 節点セット変更時の λ リマッピング
                if lam_mortar is None or new_mortar_nodes != mortar_nodes:
                    old_map = (
                        {node: idx for idx, node in enumerate(mortar_nodes)} if mortar_nodes else {}
                    )
                    new_lam = np.zeros(n_mortar)
                    if lam_mortar is not None:
                        for new_idx, node in enumerate(new_mortar_nodes):
                            old_idx = old_map.get(node)
                            if old_idx is not None and old_idx < len(lam_mortar):
                                new_lam[new_idx] = lam_mortar[old_idx]
                    lam_mortar = new_lam
                    mortar_nodes = new_mortar_nodes

                G_mortar, g_mortar = build_mortar_system(
                    manager,
                    active_indices,
                    mortar_nodes,
                    coords_def,
                    ndof,
                    ndof_per_node,
                    _n_gauss,
                    k_pen,
                )

            # 3. NCP アクティブセット判定（frozen active-set 対応）
            _update_active = (
                it % active_set_update_interval == 0
                or _frozen_ncp_active_mask is None
                or (
                    _frozen_ncp_active_mask is not None
                    and len(_frozen_ncp_active_mask) != n_geom_active
                )
            )
            if _update_active:
                if _use_mortar and n_geom_active > 0 and len(mortar_nodes) > 0:
                    p_n_mortar_arr = compute_mortar_p_n(mortar_nodes, lam_mortar, g_mortar, k_pen)
                    ncp_mortar_active = p_n_mortar_arr > 0.0
                else:
                    ncp_mortar_active = np.array([], dtype=bool)
                p_n_arr = np.maximum(0.0, lams + k_pen * (-gaps))
                ncp_active_mask_raw = p_n_arr > 0.0

                # チャタリング抑制: 直近N反復の過半数投票（S3改良5）
                if _chattering_window > 1 and n_geom_active > 0:
                    _active_history.append(ncp_active_mask_raw.copy())
                    if len(_active_history) > _chattering_window:
                        _active_history.pop(0)
                    # 全履歴を同じ長さに揃える（ペア数変動対応）
                    valid_hist = [h for h in _active_history if len(h) == n_geom_active]
                    if len(valid_hist) >= 2:
                        vote = np.mean(valid_hist, axis=0)
                        ncp_active_mask = vote > 0.5
                    else:
                        ncp_active_mask = ncp_active_mask_raw
                else:
                    ncp_active_mask = ncp_active_mask_raw

                _frozen_ncp_active_mask = ncp_active_mask.copy()
                _frozen_ncp_mortar_active = (
                    ncp_mortar_active.copy() if len(ncp_mortar_active) > 0 else None
                )
            else:
                # Frozen: p_n_arr のみ更新、active mask は前回値を使用
                p_n_arr = np.maximum(0.0, lams + k_pen * (-gaps))
                ncp_active_mask = _frozen_ncp_active_mask
                if _frozen_ncp_mortar_active is not None:
                    ncp_mortar_active = _frozen_ncp_mortar_active
                else:
                    ncp_mortar_active = np.array([], dtype=bool)

            # 4. NCP 法線力を pair.state に同期（assembly 関数が参照するため）
            for idx_p, pair in enumerate(manager.pairs):
                if pair.state.status == ContactStatus.INACTIVE:
                    continue
                lam_i = lam_all[idx_p] if idx_p < len(lam_all) else 0.0
                p_n = max(0.0, lam_i + k_pen * (-pair.state.gap))
                pair.state.lambda_n = lam_i
                pair.state.p_n = p_n
                if pair.state.k_pen <= 0.0:
                    pair.state.k_pen = k_pen
                    pair.state.k_t = k_pen * manager.config.k_t_ratio

            # 4a. 接触力ベクトル（法線）
            if _use_mortar and n_geom_active > 0 and len(mortar_nodes) > 0:
                # Mortar 接触力
                f_c = compute_mortar_contact_force(
                    manager,
                    active_indices,
                    mortar_nodes,
                    lam_mortar,
                    coords_def,
                    ndof,
                    ndof_per_node,
                    _n_gauss,
                    k_pen,
                )
            elif _line_contact:
                # Line-to-line Gauss 積分（assembly の既存インフラを利用）
                # line_contact フラグを一時的に有効化
                _orig_lc = manager.config.line_contact
                _orig_ng = manager.config.n_gauss
                manager.config.line_contact = True
                manager.config.n_gauss = _n_gauss
                f_c = compute_contact_force(
                    manager, ndof, ndof_per_node=ndof_per_node, node_coords=coords_def
                )
                manager.config.line_contact = _orig_lc
                manager.config.n_gauss = _orig_ng
            else:
                f_c = _compute_contact_force_from_lambdas(
                    manager, lam_all, ndof, ndof_per_node, k_pen=k_pen
                )

            # 4b. 摩擦力ベクトル（Alart-Curnier 接線乗数方式）
            if _use_friction:
                mu_eff = compute_mu_effective(_mu, step_display, _mu_ramp_steps)
                # 接線乗数から摩擦力を計算: f_fric = -G_t^T * λ_t
                f_friction = np.zeros(ndof)
                for _j, pair_idx in enumerate(active_indices):
                    pair = manager.pairs[pair_idx]
                    if pair.state.status == ContactStatus.INACTIVE:
                        continue
                    lam_t_j = lam_t_all[pair_idx]
                    if float(np.linalg.norm(lam_t_j)) < 1e-30:
                        continue
                    dofs = _contact_dofs(pair, ndof_per_node)
                    for axis in range(2):
                        if abs(lam_t_j[axis]) < 1e-30:
                            continue
                        g_t = _contact_tangent_shape_vector(pair, axis)
                        for kk in range(4):
                            for d in range(3):
                                # g_t_shape の符号は -(G_t^T) に対応
                                # f = -G_t^T * λ_t, g_t_shape は assembly 規約に従う
                                f_friction[dofs[kk * ndof_per_node + d]] += (
                                    lam_t_j[axis] * g_t[kk * 3 + d]
                                )
                f_c = f_c + f_friction

            # 4c. 接触力ランプ（S3改良11）: 初期反復で接触力を段階的に増大
            if _contact_force_ramp and it < _contact_force_ramp_iters:
                ramp_factor = (it + 1) / _contact_force_ramp_iters
                f_c = f_c * ramp_factor

            # 4d. 被膜スプリング力（status-137: 被膜層を陽にモデル化）
            if _use_coating:
                f_coat = manager.compute_coating_forces(coords_def)
                f_c = f_c + f_coat

            # 5. 力残差
            f_int = assemble_internal_force(u)
            R_u = f_int + f_c - f_ext
            R_u[fixed_dofs] = 0.0

            # 6. NCP 残差（Alart-Curnier 方式）
            # 6-mortar: Mortar NCP 残差
            if _use_mortar and len(mortar_nodes) > 0:
                n_mortar = len(mortar_nodes)
                C_mortar = np.empty(n_mortar)
                for k in range(n_mortar):
                    if ncp_mortar_active[k]:
                        C_mortar[k] = k_pen * g_mortar[k]
                    else:
                        C_mortar[k] = lam_mortar[k]
            else:
                C_mortar = np.array([])

            # 6-perpair: Per-pair NCP 残差
            #    Active: C_i = k_pen * g_i  (g → 0 を目標)
            #    Inactive: C_i = λ_i        (λ → 0 を目標)
            C_ac = np.empty(n_geom_active) if n_geom_active > 0 else np.array([])
            for j in range(n_geom_active):
                if ncp_active_mask[j]:
                    C_ac[j] = k_pen * gaps[j]
                else:
                    C_ac[j] = lams[j]

            # 6b. 接線 NCP 残差 + J ブロック構築（Alart-Curnier 摩擦）
            C_t_ac = np.zeros(2 * n_geom_active)
            J_blocks: dict = {}
            if _use_friction and n_geom_active > 0:
                _k_t = k_pen * manager.config.k_t_ratio
                for j in range(n_geom_active):
                    pair_idx = active_indices[j]
                    pair = manager.pairs[pair_idx]
                    lam_t_j = lam_t_all[pair_idx]  # (2,)
                    p_n_j = p_n_arr[j] if ncp_active_mask[j] else 0.0

                    if not ncp_active_mask[j]:
                        # 法線非アクティブ → 接線も非アクティブ
                        C_t_ac[2 * j] = lam_t_j[0]
                        C_t_ac[2 * j + 1] = lam_t_j[1]
                        J_blocks[pair_idx] = {"mode": "inactive"}
                        continue

                    # 接線変位増分 Δu_t
                    delta_ut = compute_tangential_displacement(
                        pair, u, u_ref, node_coords_ref, ndof_per_node
                    )

                    # 増強接線乗数: λ̂_t = λ_t + k_t * Δu_t
                    lam_t_hat = lam_t_j + _k_t * delta_ut
                    lam_t_hat_norm = float(np.linalg.norm(lam_t_hat))

                    # Stick/slip 判定
                    if lam_t_hat_norm <= mu_eff * p_n_j or lam_t_hat_norm < 1e-30:
                        # Stick: C_t = λ_t - λ̂_t = -k_t * Δu_t
                        C_t_ac[2 * j : 2 * j + 2] = lam_t_j - lam_t_hat
                        J_blocks[pair_idx] = {"mode": "active_stick"}
                    else:
                        # Slip: C_t = λ_t - μ*p_n * q̂
                        q_hat = lam_t_hat / lam_t_hat_norm
                        C_t_ac[2 * j : 2 * j + 2] = lam_t_j - mu_eff * p_n_j * q_hat

                        # J_t_t = I - (μ*p_n/||λ̂_t||) * (I - q̂⊗q̂)
                        ratio = mu_eff * p_n_j / lam_t_hat_norm
                        I2 = np.eye(2)
                        J_t_t_local = I2 - ratio * (I2 - np.outer(q_hat, q_hat))
                        # 正則化: J_t_t の固有値は 1（q̂方向）と 1-ratio（垂直方向）
                        # ratio > 1 で負固有値→行列特異化を防ぐため正則化項を追加
                        if ratio > 1.0:
                            reg_eps = (ratio - 1.0) + 1e-4
                            J_t_t_local += reg_eps * I2

                        # J_t_n = -μ * q̂  (∂f_fric/∂λ_n coupling!)
                        J_t_n_local = -mu_eff * q_hat

                        # J_t_u: slip の変位微分
                        # ∂C_t/∂u = μ*k_pen * outer(q̂, G_n_row) - μ*p_n*k_t/||λ̂_t|| * (I-q̂⊗q̂) @ G_t_rows
                        G_n_row_j = G_mat[j, :].toarray().ravel()  # (ndof,)
                        G_t_rows_j = np.zeros((2, ndof))
                        dofs_j = _contact_dofs(pair, ndof_per_node)
                        for ax in range(2):
                            g_t_sv = _contact_tangent_shape_vector(pair, ax)
                            # G_t の符号: coeffs = [-(1-s), -s, (1-t), t]
                            # g_t_shape の符号: [(1-s), s, -(1-t), -t]
                            # G_t = -g_t_shape を DOF 配置
                            for kk in range(4):
                                for d in range(3):
                                    G_t_rows_j[ax, dofs_j[kk * ndof_per_node + d]] = -g_t_sv[
                                        kk * 3 + d
                                    ]

                        proj = np.eye(2) - np.outer(q_hat, q_hat)
                        J_t_u_local = (
                            mu_eff * k_pen * np.outer(q_hat, G_n_row_j)
                            - mu_eff * p_n_j * _k_t / lam_t_hat_norm * proj @ G_t_rows_j
                        )

                        J_blocks[pair_idx] = {
                            "mode": "active_slip",
                            "J_t_t": J_t_t_local,
                            "J_t_n": J_t_n_local,
                            "J_t_u": J_t_u_local,
                        }

            # 7. 収束判定
            res_u_norm = float(np.linalg.norm(R_u))
            # 変位制御時: 初回反復の残差を基準ノルムに設定
            if _dynamic_ref and it == 0 and res_u_norm > 1e-30:
                f_ext_ref_norm = res_u_norm
            if _use_mortar and len(C_mortar) > 0:
                ncp_norm = float(np.linalg.norm(C_mortar))
                n_ncp_active = int(np.sum(ncp_mortar_active))
            else:
                ncp_norm = float(np.linalg.norm(C_ac)) if n_geom_active > 0 else 0.0
                n_ncp_active = int(np.sum(ncp_active_mask))
            ncp_t_norm = (
                float(np.linalg.norm(C_t_ac)) if (_use_friction and n_geom_active > 0) else 0.0
            )

            # 診断データ収集
            _diag.res_history.append(res_u_norm / f_ext_ref_norm)
            _diag.ncp_history.append(ncp_norm)
            _diag.ncp_t_history.append(ncp_t_norm)
            _diag.n_active_history.append(n_ncp_active)

            force_conv = res_u_norm / f_ext_ref_norm < tol_force
            ncp_conv = ncp_norm < tol_ncp
            ncp_t_conv = ncp_t_norm < tol_ncp if _use_friction else True
            all_conv = force_conv and ncp_conv and ncp_t_conv

            if all_conv:
                step_converged = True
                if show_progress:
                    msg = (
                        f"  Incr {step_display} (frac={load_frac:.4f}), iter {it}, "
                        f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                        f"||C_n|| = {ncp_norm:.3e}"
                    )
                    if _use_friction:
                        msg += f", ||C_t|| = {ncp_t_norm:.3e}"
                    msg += f" (converged, {n_ncp_active} active)"
                    print(msg)
                break

            if show_progress and it % 5 == 0:
                msg = (
                    f"  Incr {step_display} (frac={load_frac:.4f}), iter {it}, "
                    f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                    f"||C_n|| = {ncp_norm:.3e}"
                )
                if _use_friction:
                    msg += f", ||C_t|| = {ncp_t_norm:.3e}"
                msg += f", active={n_ncp_active}/{n_geom_active}"
                if adaptive_omega:
                    msg += f", ω={_omega:.3f}"
                print(msg)

            # 8. Modified NR 判定 + line search 係数
            _mnr_refresh = 5  # Modified NR の接線リフレッシュ周期
            _in_mnr = modified_nr_threshold > 0 and it >= modified_nr_threshold
            _mnr_refresh_iter = _in_mnr and (it - modified_nr_threshold) % _mnr_refresh == 0

            # Full NR / MNRリフレッシュ: 残差3倍まで許容（初期過渡対応）
            # MNR凍結中: 単調減少要求
            _ls_diverge = 1.0 if (_in_mnr and not _mnr_refresh_iter) else 3.0

            # 9. 構造接線剛性（Modified NR 凍結時はスキップ）
            if _in_mnr and not _mnr_refresh_iter and K_T is not None:
                pass  # K_T を再利用（Modified NR 凍結中）
            else:
                K_T = assemble_tangent(u)

            # 8b. line contact 法線剛性を加算（Gauss 積分）
            # Mortar 使用時はスキップ: Mortar 鞍点系の k_pen * G^T G が接触剛性を提供
            # Per-pair K_line と Mortar 剛性の二重カウントを防止
            if _line_contact and not _use_mortar:
                _orig_lc = manager.config.line_contact
                _orig_ng = manager.config.n_gauss
                manager.config.line_contact = True
                manager.config.n_gauss = _n_gauss
                K_line = compute_contact_stiffness(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    use_geometric_stiffness=manager.config.use_geometric_stiffness,
                    node_coords=coords_def,
                )
                manager.config.line_contact = _orig_lc
                manager.config.n_gauss = _orig_ng
                K_T = K_T + K_line

            # 8c. 摩擦は拡大鞍点系で処理（K_T への加算不要）

            # 9-10. 鞍点系で解く（Mortar / 摩擦 / 法線のみ で分岐）
            if _use_mortar and len(mortar_nodes) > 0 and n_ncp_active > 0:
                # Mortar 鞍点系
                active_mortar_rows = np.where(ncp_mortar_active)[0]
                G_mortar_A = G_mortar[active_mortar_rows, :]
                g_mortar_A = g_mortar[active_mortar_rows]
                du, dlam_mortar_A = _solve_saddle_point_contact(
                    K_T,
                    G_mortar_A,
                    k_pen,
                    R_u,
                    g_mortar_A,
                    fixed_dofs,
                    linear_solver=linear_solver_mode,
                    iterative_tol=iterative_tol_cfg,
                    ilu_drop_tol=ilu_drop_tol_cfg,
                    use_block_preconditioner=use_block_preconditioner,
                    gmres_dof_threshold=gmres_dof_threshold_cfg,
                    use_amg=manager.config.use_amg_preconditioner,
                    residual_scaling=_residual_scaling,
                )

                # 11m. Line search + Mortar 乗数更新
                if use_line_search:
                    alpha = _ncp_line_search(
                        u,
                        du,
                        f_ext,
                        fixed_dofs,
                        assemble_internal_force,
                        res_u_norm,
                        max_steps=line_search_max_steps,
                        f_c=f_c,
                        diverge_factor=_ls_diverge,
                    )
                    du = alpha * du
                elif du_norm_cap > 0.0:
                    _du_n = float(np.linalg.norm(du))
                    _u_ref_n = max(float(np.linalg.norm(u)), 1.0)
                    if _du_n > du_norm_cap * _u_ref_n:
                        du *= du_norm_cap * _u_ref_n / _du_n
                u += _omega * du
                for j_local, row in enumerate(active_mortar_rows):
                    lam_mortar[row] += _omega * dlam_mortar_A[j_local]
                for k in range(len(mortar_nodes)):
                    if not ncp_mortar_active[k]:
                        lam_mortar[k] = 0.0
                lam_mortar = np.maximum(lam_mortar, 0.0)

            elif _use_friction and n_geom_active > 0:
                # Alart-Curnier 摩擦拡大鞍点系
                G_t_mat = _build_tangential_constraint_jacobian(
                    manager, active_indices, ndof, ndof_per_node
                )
                _k_t = k_pen * manager.config.k_t_ratio

                du, dlam_n, dlam_t = _solve_augmented_friction_system(
                    K_T,
                    G_mat,
                    G_t_mat,
                    k_pen,
                    _k_t,
                    R_u,
                    C_ac,
                    C_t_ac,
                    J_blocks,
                    fixed_dofs,
                    n_ncp_active,
                    ncp_active_mask,
                    active_indices,
                )

                # 11. Line search + 更新
                if use_line_search:
                    alpha = _ncp_line_search(
                        u,
                        du,
                        f_ext,
                        fixed_dofs,
                        assemble_internal_force,
                        res_u_norm,
                        max_steps=line_search_max_steps,
                        f_c=f_c,
                        diverge_factor=_ls_diverge,
                    )
                    du = alpha * du
                elif du_norm_cap > 0.0:
                    _du_n = float(np.linalg.norm(du))
                    _u_ref_n = max(float(np.linalg.norm(u)), 1.0)
                    if _du_n > du_norm_cap * _u_ref_n:
                        du *= du_norm_cap * _u_ref_n / _du_n
                u += _omega * du

                # 法線乗数の更新
                if n_ncp_active > 0:
                    active_rows_n = np.where(ncp_active_mask)[0]
                    for j_local in range(n_ncp_active):
                        j_geom = int(active_rows_n[j_local])
                        pair_idx = active_indices[j_geom]
                        lam_all[pair_idx] += _omega * dlam_n[j_local]

                    # 接線乗数の更新
                    for j_local in range(n_ncp_active):
                        j_geom = int(active_rows_n[j_local])
                        pair_idx = active_indices[j_geom]
                        lam_t_all[pair_idx] += _omega * dlam_t[2 * j_local : 2 * j_local + 2]

            else:
                # 従来の法線のみ鞍点系
                if n_ncp_active > 0:
                    active_row_indices = np.where(ncp_active_mask)[0]
                    G_A = G_mat[active_row_indices, :]
                    g_A = gaps[active_row_indices]
                else:
                    G_A = sp.csr_matrix((0, ndof))
                    g_A = np.array([])

                du, dlam_A = _solve_saddle_point_contact(
                    K_T,
                    G_A,
                    k_pen,
                    R_u,
                    g_A,
                    fixed_dofs,
                    linear_solver=linear_solver_mode,
                    iterative_tol=iterative_tol_cfg,
                    ilu_drop_tol=ilu_drop_tol_cfg,
                    use_block_preconditioner=use_block_preconditioner,
                    gmres_dof_threshold=gmres_dof_threshold_cfg,
                    use_amg=manager.config.use_amg_preconditioner,
                    residual_scaling=_residual_scaling,
                )

                # 11. Line search + 更新
                if use_line_search:
                    alpha = _ncp_line_search(
                        u,
                        du,
                        f_ext,
                        fixed_dofs,
                        assemble_internal_force,
                        res_u_norm,
                        max_steps=line_search_max_steps,
                        f_c=f_c,
                        diverge_factor=_ls_diverge,
                    )
                    du = alpha * du
                elif du_norm_cap > 0.0:
                    _du_n = float(np.linalg.norm(du))
                    _u_ref_n = max(float(np.linalg.norm(u)), 1.0)
                    if _du_n > du_norm_cap * _u_ref_n:
                        du *= du_norm_cap * _u_ref_n / _du_n
                u += _omega * du

                # NCP アクティブ乗数の更新
                if n_ncp_active > 0:
                    active_pair_indices = [
                        active_indices[j] for j in range(n_geom_active) if ncp_active_mask[j]
                    ]
                    for j_local, pair_idx in enumerate(active_pair_indices):
                        lam_all[pair_idx] += _omega * dlam_A[j_local]

            # NCP 非アクティブ乗数をゼロに（Mortar 時はスキップ）
            if not _use_mortar:
                for j in range(n_geom_active):
                    if not ncp_active_mask[j]:
                        lam_all[active_indices[j]] = 0.0
                        lam_t_all[active_indices[j]] = 0.0
                lam_all = np.maximum(lam_all, 0.0)

            # --- adaptive omega: メリット関数による緩和係数更新 ---
            if adaptive_omega:
                _merit = res_u_norm + ncp_norm
                if _merit < _prev_merit:
                    _omega = min(_omega * omega_growth, omega_max)
                else:
                    _omega = max(_omega * omega_shrink, omega_min)
                _prev_merit = _merit

                # omega回復メカニズム: 最小値に張り付いた場合の脱出
                if _omega <= omega_min + 1e-15:
                    _omega_stall_count += 1
                    if _omega_stall_count >= 20:
                        _omega = omega_init
                        _prev_merit = float("inf")
                        _omega_stall_count = 0
                else:
                    _omega_stall_count = 0

            # 変位ノルム判定
            u_norm = float(np.linalg.norm(u))
            du_norm_val = float(np.linalg.norm(du))

            # 診断データ: du情報を収集
            _diag.du_norm_history.append(du_norm_val)
            _max_du_idx = int(np.argmax(np.abs(du))) if du_norm_val > 0 else -1
            _diag.max_du_dof_history.append(_max_du_idx)

            if u_norm > 1e-30 and du_norm_val / u_norm < tol_disp and ncp_conv:
                step_converged = True
                if show_progress:
                    print(
                        f"  Incr {step_display} (frac={load_frac:.4f}), iter {it}, "
                        f"||du||/||u|| = {du_norm_val / u_norm:.3e} "
                        f"(disp converged, {n_ncp_active} active)"
                    )
                break

            # エネルギー収束判定: |du · R_u| / |du_0 · R_0| < tol_energy
            energy = abs(float(np.dot(du, R_u)))
            if energy_ref is None:
                energy_ref = energy if energy > 1e-30 else 1.0
            if energy_ref > 1e-30 and energy / energy_ref < 1e-10 and ncp_conv:
                step_converged = True
                if show_progress:
                    print(
                        f"  Incr {step_display} (frac={load_frac:.4f}), iter {it}, "
                        f"energy = {energy:.3e} "
                        f"(energy converged, {n_ncp_active} active)"
                    )
                break

        if not step_converged:
            # --- 適応時間増分: 不収束時のステップ縮小リトライ ---
            delta = load_frac - load_frac_prev
            if _adaptive_dt and delta > _dt_min_frac + 1e-15:
                # 状態をチェックポイントに復元
                u = u_ckpt.copy()
                lam_all = lam_ckpt.copy()
                lam_t_all = lam_t_ckpt.copy()
                if lam_mortar_ckpt is not None:
                    lam_mortar = lam_mortar_ckpt.copy()
                mortar_nodes = list(mortar_nodes_ckpt)
                u_ref = u_ref_ckpt.copy()
                _ul_frac_base = _ul_frac_base_ckpt
                if _ul:
                    ul_assembler.rollback()
                    node_coords_ref = ul_assembler.coords_ref
                # 予測子もリセット（ロールバック後は外挿しない）
                delta_frac_prev = 0.0
                _consecutive_good = 0  # カットバック発生→積極成長モードに復帰
                # 中間点を挿入（適応縮小）
                mid_frac = load_frac_prev + delta * _dt_shrink
                step_queue.appendleft(load_frac)  # 元のターゲットを戻す
                step_queue.appendleft(mid_frac)  # 縮小ステップを先頭に
                step_display -= 1  # カウンタを巻き戻し
                if show_progress:
                    print(
                        f"  Adaptive dt retry: frac {load_frac:.4f} → "
                        f"sub-steps [{mid_frac:.4f}, {load_frac:.4f}] "
                        f"(delta {delta:.4f} → {delta * _dt_shrink:.4f})"
                    )
                continue
            else:
                if show_progress:
                    print(
                        f"  WARNING: Step {step_display} (frac={load_frac:.4f}) "
                        f"did not converge in {max_iter} iterations."
                    )
                    print(_diag.format_report(max_iter=max_iter))
                _u_fail = ul_assembler.u_total_accum + u if _ul else u
                return NCPSolveResult(
                    u=_u_fail,
                    lambdas=lam_all,
                    converged=False,
                    n_increments=step_display,
                    total_newton_iterations=total_newton,
                    n_active_final=manager.n_active,
                    load_history=load_history,
                    displacement_history=disp_history,
                    contact_force_history=contact_force_history,
                    graph_history=graph_history,
                    diagnostics=_diag,
                )

        # ステップ完了 — キューから消費
        step_queue.popleft()

        # --- Updated Lagrangian: 参照配置更新 & 変位リセット ---
        if _ul:
            ul_assembler.update_reference(u)
            node_coords_ref = ul_assembler.coords_ref
            _ul_frac_base = load_frac
            # u リセット（参照配置に吸収済み）
            u = np.zeros(ndof)
            u_ref = np.zeros(ndof)

        # --- 適応時間増分制御（S3改良6）: 次ステップ幅の動的決定 ---
        if _adaptive_dt and load_frac < 1.0 - 1e-12:
            current_delta = load_frac - load_frac_prev
            next_delta = current_delta  # デフォルトは同じ幅

            # (a) 収束反復数に基づく拡大/縮小（安定化検出付き成長戦略）
            step_iters = it + 1  # このステップの反復数
            if step_iters <= _dt_grow_iter:
                _consecutive_good += 1
                if _consecutive_good <= 2:
                    # 成長フェーズ（初期 or カットバック後の回復）
                    next_delta = current_delta * _dt_grow
                else:
                    # 安定フェーズ: 増加率を漸減（オーバーシュート防止）
                    _damp = max(0.1, 1.0 / _consecutive_good)
                    next_delta = current_delta * (1.0 + (_dt_grow - 1.0) * _damp)
            elif step_iters >= _dt_shrink_iter:
                next_delta = current_delta * _dt_shrink
                _consecutive_good = 0
            else:
                # 中間: 変更なし
                _consecutive_good = 0

            # (b) 接触状態変化率に基づく縮小
            _current_n_active = manager.n_active
            if _prev_n_active > 0:
                change_rate = abs(_current_n_active - _prev_n_active) / max(_prev_n_active, 1)
                if change_rate > _dt_contact_change_thr:
                    next_delta = min(next_delta, current_delta * _dt_shrink)

            # 分率制限
            next_delta = max(next_delta, _dt_min_frac)
            next_delta = min(next_delta, _dt_max_frac)

            # 次のターゲット分率
            next_frac = min(load_frac + next_delta, 1.0)
            # 残り区間が小さすぎる場合は1.0に揃える
            if 1.0 - next_frac < _dt_min_frac * 0.5:
                next_frac = 1.0
            step_queue.appendleft(next_frac)

            if show_progress and abs(next_delta - current_delta) > 1e-10:
                print(
                    f"  Adaptive dt: delta {current_delta:.4f} → {next_delta:.4f} "
                    f"(iters={step_iters}, n_active={_current_n_active})"
                )

        _prev_n_active = manager.n_active

        # --- k_pen continuation（S3改良8）: ステップ進行で段階的にk_penを目標値に近づける ---
        if _k_pen_cont and k_pen < _k_pen_target - 1e-30:
            # 対数スケールで等分割
            _k_pen_ratio = _k_pen_target / max(k_pen, 1e-30)
            _k_pen_step_factor = _k_pen_ratio ** (1.0 / max(_k_pen_cont_steps, 1))
            k_pen = min(k_pen * _k_pen_step_factor, _k_pen_target)
            if show_progress:
                print(f"  k_pen continuation: k_pen → {k_pen:.2e} (target={_k_pen_target:.2e})")

        # 接線予測子用: 前ステップの変位増分と荷重増分を記録
        delta_frac_prev = load_frac - load_frac_prev
        u_prev_converged = u.copy()  # UL 時は u=0 が正しい基準
        load_frac_prev = load_frac
        for i, pair in enumerate(manager.pairs):
            if i < len(lam_all):
                pair.state.lambda_n = lam_all[i]
                pair.state.p_n = max(0.0, lam_all[i] + k_pen * (-pair.state.gap))

        # --- Mortar 適応ペナルティ増大 ---
        # Mortar 重み付きギャップから最大貫入率を推定し、k_pen を自動増大
        if _use_mortar and lam_mortar is not None and len(mortar_nodes) > 0:
            _tol_pen = manager.config.tol_penetration_ratio
            _pen_growth = manager.config.penalty_growth_factor
            _k_pen_max = manager.config.k_pen_max
            if _tol_pen > 0.0 and g_mortar is not None:
                # 法線力が正（接触活性）の Mortar 節点について貫入チェック
                p_n_m = compute_mortar_p_n(mortar_nodes, lam_mortar, g_mortar, k_pen)
                for mk in range(len(mortar_nodes)):
                    if p_n_m[mk] > 0.0 and g_mortar[mk] < 0.0:
                        # Mortar ギャップは重み付き → 代表半径で正規化
                        pen_abs = abs(g_mortar[mk])
                        # 代表半径の推定: アクティブペアの平均 search_radius
                        sr_avg = (
                            np.mean(
                                [
                                    p.search_radius
                                    for p in manager.pairs
                                    if p.state.status != ContactStatus.INACTIVE
                                    and p.search_radius > 1e-30
                                ]
                            )
                            if manager.n_active > 0
                            else 1.0
                        )
                        pen_ratio = pen_abs / max(sr_avg, 1e-30)
                        if pen_ratio > _tol_pen and k_pen < _k_pen_max:
                            k_pen = min(k_pen * _pen_growth, _k_pen_max)
                            if show_progress:
                                print(
                                    f"  Mortar adaptive k_pen: "
                                    f"pen_ratio={pen_ratio:.3f} > tol={_tol_pen:.3f}, "
                                    f"k_pen → {k_pen:.2e}"
                                )
                            break  # 1回の増大で次ステップへ

        # 摩擦用: 参照変位を更新（次ステップ用）
        u_ref = u.copy()

        # チェックポイント保存（二分法ロールバック用）
        u_ckpt = u.copy()
        lam_ckpt = lam_all.copy()
        lam_t_ckpt = lam_t_all.copy()
        lam_mortar_ckpt = lam_mortar.copy() if lam_mortar is not None else None
        mortar_nodes_ckpt = list(mortar_nodes)
        u_ref_ckpt = u_ref.copy()
        _ul_frac_base_ckpt = _ul_frac_base
        if _ul:
            ul_assembler.checkpoint()

        load_history.append(load_frac)
        _u_hist = ul_assembler.u_total_accum + u if _ul else u.copy()
        disp_history.append(_u_hist.copy() if _ul else _u_hist)
        f_c_norm = float(np.linalg.norm(f_c))
        contact_force_history.append(f_c_norm)

        try:
            graph = snapshot_contact_graph(manager, step_index=step_display - 1)
            graph_history.add_snapshot(graph)
        except Exception:
            pass

    # UL モード: 累積変位を返す（u はリセット済みのため）
    _u_out = ul_assembler.u_total_accum + u if _ul else u
    return NCPSolveResult(
        u=_u_out,
        lambdas=lam_all,
        converged=True,
        n_increments=step_display,
        total_newton_iterations=total_newton,
        n_active_final=manager.n_active,
        load_history=load_history,
        displacement_history=disp_history,
        contact_force_history=contact_force_history,
        graph_history=graph_history,
    )
