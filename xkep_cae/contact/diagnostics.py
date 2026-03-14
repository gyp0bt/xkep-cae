"""収束診断・結果データクラス.

solver_ncp.py / solver_smooth_penalty.py 共通のデータ構造。

- ConvergenceDiagnostics: 収束失敗時の診断情報
- NCPSolveResult: ソルバー結果
- NCPSolverInput: ソルバー入力（構造化パラメータ）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from xkep_cae.contact.graph import ContactGraphHistory
from xkep_cae.contact.pair import ContactManager


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
    # 動的解析結果（dynamics=True 時のみ有効）
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None

    @property
    def n_load_steps(self) -> int:
        """後方互換性のためのエイリアス。n_increments を参照。"""
        return self.n_increments


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
    contact_stabilization: float = 0.0
    lambda_decay: float = 1.0
    lambda_init: np.ndarray | None = None

    def solve(self) -> NCPSolveResult:
        """このパラメータで NCP ソルバーを実行する."""
        from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp

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
