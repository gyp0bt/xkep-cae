"""Newton ループ内サブプロセス群.

動的 Newton ソルバーから呼び出される共通サブプロセス。
各ステップを独立した SolverProcess として実装。

status-222 で Uzawa ループ・準静的ソルバーを削除。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._manager_process import (
    UpdateGeometryInput,
    UpdateGeometryProcess,
)
from xkep_cae.contact.solver._utils import (
    DeformedCoordsInput,
    DeformedCoordsProcess,
    NCPLineSearchInput,
    NCPLineSearchProcess,
)
from xkep_cae.core import ProcessMeta, SolverProcess

# ================================================================
# 1. ContactForceAssemblyProcess（ステップ 2〜5: 接触力+摩擦+被膜+残差）
# ================================================================


@dataclass(frozen=True)
class ContactForceAssemblyInput:
    """接触力アセンブリの入力."""

    u: np.ndarray
    f_ext: np.ndarray
    fixed_dofs: np.ndarray
    manager: object
    node_coords_ref: np.ndarray
    contact_force_strategy: object
    friction_strategy: object
    coating_strategy: object | None
    k_pen: float
    mu: float
    u_ref: np.ndarray
    load_frac: float
    load_frac_prev: float
    increment_display: int
    ndof_per_node: int
    use_coating: bool
    assemble_internal_force: object
    connectivity: np.ndarray | None = None  # Hermite 中心線補間用


@dataclass(frozen=True)
class ContactForceAssemblyOutput:
    """接触力アセンブリの出力."""

    f_c: np.ndarray
    R_u: np.ndarray
    coords_def: np.ndarray


class ContactForceAssemblyProcess(
    SolverProcess[ContactForceAssemblyInput, ContactForceAssemblyOutput],
):
    """接触力・摩擦力・被膜力のアセンブリ + 力残差計算（ステップ 2〜5）."""

    meta = ProcessMeta(
        name="ContactForceAssembly",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_solver.md",
    )
    uses = [DeformedCoordsProcess, UpdateGeometryProcess]

    def process(self, inp: ContactForceAssemblyInput) -> ContactForceAssemblyOutput:
        u = inp.u

        # 1. 幾何更新
        _dc_out = DeformedCoordsProcess().process(
            DeformedCoordsInput(
                node_coords_ref=inp.node_coords_ref,
                u=u,
                ndof_per_node=inp.ndof_per_node,
            )
        )
        coords_def = _dc_out.coords
        _freeze_st = (
            hasattr(inp.manager, "config")
            and hasattr(inp.manager.config, "freeze_geometry_in_nr")
            and inp.manager.config.freeze_geometry_in_nr
        )
        _st_relax = 1.0
        if hasattr(inp.manager, "config") and hasattr(inp.manager.config, "st_relaxation"):
            _st_relax = inp.manager.config.st_relaxation
        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=inp.manager,
                node_coords=coords_def,
                connectivity=inp.connectivity,
                freeze_active_set=True,
                freeze_st=_freeze_st,
                st_relaxation=_st_relax,
            )
        )
        inp.manager.pairs[:] = _ug_out.manager.pairs

        # 2. 接触力
        f_c, _ = inp.contact_force_strategy.evaluate(u, inp.manager, inp.k_pen)

        # 3. 摩擦力
        if hasattr(inp.friction_strategy, "set_mu_ramp_counter"):
            inp.friction_strategy.set_mu_ramp_counter(inp.increment_display)
        f_friction, _ = inp.friction_strategy.evaluate(
            u,
            inp.manager.pairs,
            inp.mu,
            u_ref=inp.u_ref,
        )
        f_c = f_c + f_friction

        # 4. 被膜力
        if inp.use_coating and inp.coating_strategy is not None:
            coat_dt = max(inp.load_frac - inp.load_frac_prev, 1e-15)
            f_coat = inp.coating_strategy.forces(
                inp.manager.pairs, coords_def, inp.manager.config, coat_dt
            )
            f_c = f_c + f_coat
            if inp.manager.config.coating_mu > 0.0:
                f_coat_fric = inp.coating_strategy.friction_forces(
                    inp.manager.pairs, coords_def, inp.manager.config, u, inp.u_ref
                )
                f_c = f_c + f_coat_fric

        # 5. 力残差
        # f_c のアセンブリは g_shape 規約で「ワイヤ→上向き」が正。
        # 物理的には接触力はワイヤを下に押す外力なので符号反転。
        f_c = -f_c
        f_int = inp.assemble_internal_force(u)
        R_u = f_int + f_c - inp.f_ext
        R_u[inp.fixed_dofs] = 0.0

        return ContactForceAssemblyOutput(
            f_c=f_c,
            R_u=R_u,
            coords_def=coords_def,
        )


# ================================================================
# 2. ConvergenceCheckProcess（収束判定）
# ================================================================


class ConvergenceType(Enum):
    """収束種別."""

    NOT_CONVERGED = "not_converged"
    FORCE = "force"
    DISPLACEMENT = "displacement"
    ENERGY = "energy"


@dataclass(frozen=True)
class ConvergenceCheckInput:
    """収束判定の入力."""

    R_u: np.ndarray
    du: np.ndarray | None
    u: np.ndarray
    f_ext_ref_norm: float
    tol_force: float
    tol_disp: float
    dynamic_ref: bool
    is_first_attempt: bool
    energy_ref: float | None
    manager: object
    ndof_per_node: int = 6
    char_length: float = 0.0  # 代表長さ [mm]（回転残差の正規化用、status-241）
    mpc_transform: object | None = None  # MPCEliminationResult（status-255: 縮退系残差判定）
    atol_force: float = 0.0  # 絶対許容値 [N]（status-297: 微小dt時の収束保証）


@dataclass(frozen=True)
class ConvergenceCheckOutput:
    """収束判定の出力."""

    converged: bool
    convergence_type: ConvergenceType
    res_u_norm: float
    f_ref: float
    n_active: int
    du_norm: float
    energy: float
    energy_ref: float | None
    res_trans_norm: float = 0.0
    res_rot_norm: float = 0.0
    res_weighted_norm: float = 0.0  # 重み付き統合ノルム [N]（status-241）


class ConvergenceCheckProcess(
    SolverProcess[ConvergenceCheckInput, ConvergenceCheckOutput],
):
    """Newton イテレーションの収束判定（力・変位・エネルギー）."""

    meta = ProcessMeta(
        name="ConvergenceCheck",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_solver.md",
    )

    def process(self, inp: ConvergenceCheckInput) -> ConvergenceCheckOutput:
        # MPC縮退系での残差判定（status-255）:
        # R_red = T^T R_u とすればslave DOFの制約反力が消去され、
        # 独立DOFのみの残差で収束判定できる。
        _mpc = inp.mpc_transform
        if _mpc is not None:
            _R_red = _mpc.T.T @ inp.R_u
            # _R_red は縮退系（ndof_reduced次元）
            # 独立DOFのndof_per_nodeは元と同じだが、
            # slave DOFが除外された不均一配列なので全体ノルムで評価
            R_eval = np.asarray(_R_red).ravel()
        else:
            R_eval = inp.R_u

        res_u_norm = float(np.linalg.norm(R_eval))

        # 力/モーメント分離: 並進 DOF と回転 DOF のノルムを個別計算
        ndpn = inp.ndof_per_node
        if _mpc is not None:
            # MPC縮退系: independent_dofs から並進/回転を分離
            _indep = _mpc.independent_dofs
            _trans_mask = (_indep % ndpn) < 3
            _rot_mask = (_indep % ndpn) >= 3
            res_trans_norm = float(np.linalg.norm(R_eval[_trans_mask]))
            res_rot_norm = float(np.linalg.norm(R_eval[_rot_mask]))
        elif ndpn >= 6 and len(inp.R_u) >= ndpn:
            n_nodes = len(inp.R_u) // ndpn
            # 並進 DOF: 各節点の先頭3成分
            trans_idx = np.array(
                [i * ndpn + d for i in range(n_nodes) for d in range(3)], dtype=int
            )
            # 回転 DOF: 各節点の後半3成分
            rot_idx = np.array(
                [i * ndpn + d for i in range(n_nodes) for d in range(3, 6)], dtype=int
            )
            res_trans_norm = float(np.linalg.norm(inp.R_u[trans_idx]))
            res_rot_norm = float(np.linalg.norm(inp.R_u[rot_idx]))
        else:
            res_trans_norm = res_u_norm
            res_rot_norm = 0.0

        # 重み付き統合ノルム: 回転残差を代表長さで除して力単位に正規化
        # R_rot [N·mm] / L_char [mm] → [N] として並進残差と統合
        _L = inp.char_length
        if _L > 0 and res_rot_norm > 0:
            res_weighted_norm = float(np.sqrt(res_trans_norm**2 + (res_rot_norm / _L) ** 2))
        else:
            res_weighted_norm = res_trans_norm

        f_ref = inp.f_ext_ref_norm
        if inp.dynamic_ref and inp.is_first_attempt and res_trans_norm > 1e-30:
            # 参照値も並進残差のみで設定
            f_ref = res_trans_norm

        n_active = sum(1 for p in inp.manager.pairs if hasattr(p, "state") and p.state.p_n > 0.0)

        # 力収束: 並進残差のみで判定（接触力は並進 DOF のみにアセンブリ）
        # atol_force: 絶対許容値。微小dtでf_refが極小でも、
        # 絶対残差が通常スケールの力収束水準以下なら収束を認める（status-297）。
        _force_converged = res_trans_norm / f_ref < inp.tol_force
        if not _force_converged and inp.atol_force > 0:
            _force_converged = res_trans_norm < inp.atol_force
        if _force_converged:
            return ConvergenceCheckOutput(
                converged=True,
                convergence_type=ConvergenceType.FORCE,
                res_u_norm=res_u_norm,
                f_ref=f_ref,
                n_active=n_active,
                du_norm=0.0,
                energy=0.0,
                energy_ref=inp.energy_ref,
                res_trans_norm=res_trans_norm,
                res_rot_norm=res_rot_norm,
                res_weighted_norm=res_weighted_norm,
            )

        # 変位・エネルギー収束は du が必要
        du_norm = 0.0
        energy = 0.0
        energy_ref = inp.energy_ref
        conv_type = ConvergenceType.NOT_CONVERGED

        if inp.du is not None:
            du_norm = float(np.linalg.norm(inp.du))
            u_norm = float(np.linalg.norm(inp.u))

            # status-278: 変位収束を並進DOFのみで判定。
            # lumped質量行列の回転慣性が極小（~10^-8）のため、
            # 動的残差の回転成分がNRで解消されず du の回転成分が支配的になる。
            # 並進DOFのみで判定すれば、回転残差に引きずられず収束を達成できる。
            ndpn = inp.ndof_per_node
            if ndpn >= 6 and len(inp.du) >= ndpn:
                _n_nd = len(inp.du) // ndpn
                _trans_idx = np.array(
                    [i * ndpn + d for i in range(_n_nd) for d in range(3)], dtype=int
                )
                _du_trans = float(np.linalg.norm(inp.du[_trans_idx]))
                _u_trans = float(np.linalg.norm(inp.u[_trans_idx]))
            else:
                _du_trans = du_norm
                _u_trans = u_norm

            if _u_trans > 1e-30 and _du_trans / _u_trans < inp.tol_disp:
                conv_type = ConvergenceType.DISPLACEMENT
            else:
                energy = abs(float(np.dot(inp.du, inp.R_u)))
                if energy_ref is None:
                    energy_ref = energy if energy > 1e-30 else 1.0
                if energy_ref > 1e-30 and energy / energy_ref < 1e-10:
                    conv_type = ConvergenceType.ENERGY

        return ConvergenceCheckOutput(
            converged=conv_type != ConvergenceType.NOT_CONVERGED,
            convergence_type=conv_type,
            res_u_norm=res_u_norm,
            f_ref=f_ref,
            n_active=n_active,
            du_norm=du_norm,
            energy=energy,
            energy_ref=energy_ref,
            res_trans_norm=res_trans_norm,
            res_rot_norm=res_rot_norm,
            res_weighted_norm=res_weighted_norm,
        )


# ================================================================
# 3. TangentAssemblyProcess（ステップ 7: 接線剛性組立）
# ================================================================


@dataclass(frozen=True)
class TangentAssemblyInput:
    """接線剛性アセンブリの入力."""

    u: np.ndarray
    manager: object
    contact_force_strategy: object
    friction_strategy: object
    coating_strategy: object | None
    assemble_tangent: object
    k_pen: float
    mu: float
    ndof: int
    coords_def: np.ndarray
    load_frac: float
    load_frac_prev: float
    use_coating: bool
    contact_tangent_scale: float = 1.0  # 接触剛性スケーリング（status-247 リラクゼーション用）


@dataclass(frozen=True)
class TangentAssemblyOutput:
    """接線剛性アセンブリの出力."""

    K_T: sp.spmatrix


class TangentAssemblyProcess(
    SolverProcess[TangentAssemblyInput, TangentAssemblyOutput],
):
    """接線剛性行列の組立（構造 + 接触 + 摩擦 + 被膜）."""

    meta = ProcessMeta(
        name="TangentAssembly",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_solver.md",
    )

    def process(self, inp: TangentAssemblyInput) -> TangentAssemblyOutput:
        K_T = inp.assemble_tangent(inp.u)

        K_c = inp.contact_force_strategy.tangent(
            inp.u,
            inp.manager,
            inp.k_pen,
            node_coords=inp.coords_def,
        )
        # リラクゼーション中は接触剛性をスケーリング（残差との整合性、status-247）
        _cts = inp.contact_tangent_scale
        K_T = K_T + (K_c * _cts if _cts < 1.0 else K_c)

        # 被膜剛性
        if inp.use_coating and inp.coating_strategy is not None:
            coat_dt = max(inp.load_frac - inp.load_frac_prev, 1e-15)
            K_coat = inp.coating_strategy.stiffness(
                inp.manager.pairs, inp.coords_def, inp.manager.config, inp.ndof, coat_dt
            )
            K_T = K_T + K_coat
            if inp.manager.config.coating_mu > 0.0:
                K_coat_fric = inp.coating_strategy.friction_stiffness(
                    inp.manager.pairs, inp.coords_def, inp.manager.config, inp.ndof
                )
                K_T = K_T + K_coat_fric

        # 摩擦剛性（符号反転: f_c=-f_c_raw なので dR/du には -d(f_fric)/du が必要）
        if inp.friction_strategy.friction_tangents:
            _use_st = (
                hasattr(inp.manager, "config")
                and hasattr(inp.manager.config, "consistent_st_tangent")
                and inp.manager.config.consistent_st_tangent
            )
            # Hermite隣接ノード拡張パラメータ（status-274）
            # status-324: gap culling threshold を摩擦 K_st にも適用
            _fric_kw: dict = {}
            if hasattr(inp.contact_force_strategy, "compute_gap_cull_threshold"):
                _fric_kw["gap_cull_threshold"] = (
                    inp.contact_force_strategy.compute_gap_cull_threshold(inp.k_pen)
                )
            if _use_st:
                _fric_use_hermite = (
                    hasattr(inp.manager, "config")
                    and hasattr(inp.manager.config, "use_hermite_centerline")
                    and inp.manager.config.use_hermite_centerline
                )
                if _fric_use_hermite:
                    _fric_conn = getattr(inp.manager, "connectivity", None)
                    if _fric_conn is not None:
                        from xkep_cae.contact.geometry._compute import (
                            _compute_adj_node_map,
                            _compute_node_counts,
                            _compute_node_tangents,
                        )

                        _fric_kw["use_hermite"] = True
                        _fric_kw["node_tangents"] = _compute_node_tangents(
                            inp.coords_def, _fric_conn
                        )
                        _max_node = int(np.max(_fric_conn)) + 1
                        _fric_kw["node_counts"] = _compute_node_counts(_max_node, _fric_conn)
                        _fric_kw["adj_node_map"] = _compute_adj_node_map(_fric_conn)
            K_fric = inp.friction_strategy.tangent(
                inp.u,
                inp.manager.pairs,
                inp.mu,
                node_coords=inp.coords_def if _use_st else None,
                consistent_st_tangent=_use_st,
                **_fric_kw,
            )
            K_T = K_T - K_fric

        return TangentAssemblyOutput(K_T=K_T)


# ================================================================
# 4. LinearSolveProcess（ステップ 8: 線形ソルブ）
# ================================================================


@dataclass(frozen=True)
class LinearSolveInput:
    """線形ソルブの入力."""

    K_T: sp.spmatrix
    R_u: np.ndarray
    fixed_dofs: np.ndarray
    mpc_transform: object | None = None  # MPCEliminationResult


@dataclass(frozen=True)
class LinearSolveOutput:
    """線形ソルブの出力."""

    du: np.ndarray | None
    success: bool


def _sparse_solve(K_csc: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """スパース線形ソルブ: pypardiso → scipy fallback（status-311）."""
    try:
        import pypardiso

        return pypardiso.spsolve(K_csc, rhs)
    except (ImportError, RuntimeError):
        pass
    from scipy.sparse.linalg import spsolve

    return spsolve(K_csc, rhs)


def _zero_sparse_rows(K: sp.spmatrix, dofs: np.ndarray) -> None:
    """CSR行列の指定行をゼロ化（ベクトル化版、status-312）.

    forループ `for d in dofs: data[indptr[d]:indptr[d+1]] = 0` を
    NumPy配列演算で置換。固定DOFが数千〜数万の大規模問題で高速。
    """
    if len(dofs) == 0:
        return
    indptr = K.indptr
    starts = indptr[dofs]
    ends = indptr[dofs + 1]
    lengths = ends - starts
    total = lengths.sum()
    if total == 0:
        return
    offsets = np.repeat(starts, lengths)
    within = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    K.data[offsets + within] = 0.0


class LinearSolveProcess(
    SolverProcess[LinearSolveInput, LinearSolveOutput],
):
    """境界条件適用 + 線形ソルブ."""

    meta = ProcessMeta(
        name="LinearSolve",
        module="solve",
        version="1.1.0",
        document_path="docs/newton_solver.md",
    )

    def process(self, inp: LinearSolveInput) -> LinearSolveOutput:
        _mpc = inp.mpc_transform

        if _mpc is not None:
            # MPC DOF消去: T^T K T の縮退系を求解
            return self._solve_with_mpc(inp, _mpc)

        return self._solve_standard(inp)

    def _solve_standard(self, inp: LinearSolveInput) -> LinearSolveOutput:
        """標準BC適用 + 線形ソルブ.

        status-311: tolil() + forループを排除し、CSR直接操作に置換。
        status-312: forループをNumPyベクトル化（_zero_sparse_rows）。
        """
        _rhs = -inp.R_u.copy()
        fixed = inp.fixed_dofs

        K_csc = inp.K_T.tocsc()
        if len(fixed) > 0:
            K_csr = K_csc.tocsr()
            _zero_sparse_rows(K_csr, fixed)
            K_csc = K_csr.tocsc()
            _zero_sparse_rows(K_csc, fixed)  # CSC列 = CSCの「行」
            K_csr = K_csc.tocsr()
            diag_vals = K_csr.diagonal()
            diag_vals[fixed] = 1.0
            K_csr.setdiag(diag_vals)
            K_csc = K_csr.tocsc()
            K_csc.eliminate_zeros()
            _rhs[fixed] = 0.0

        try:
            du = _sparse_solve(K_csc, _rhs)
            return LinearSolveOutput(du=du, success=True)
        except Exception:
            return LinearSolveOutput(du=None, success=False)

    def _solve_with_mpc(self, inp: LinearSolveInput, mpc: object) -> LinearSolveOutput:
        """MPC DOF消去による縮退系ソルブ.

        K_red = T^T K T, f_red = T^T f
        du_red = solve(K_red, f_red)
        du_full = T @ du_red

        固定DOFはMPC変換後の縮退系で適用する。
        """
        T = mpc.T  # (ndof_total, ndof_reduced)
        K_full = inp.K_T.tocsc()
        _rhs_full = -inp.R_u.copy()

        # 縮退系に変換
        K_red = T.T @ K_full @ T
        _rhs_red = T.T @ _rhs_full

        # 固定DOFを縮退系のindexに変換（status-312: forループ排除）
        indep = np.asarray(mpc.independent_dofs)
        _indep_to_reduced = np.full(K_full.shape[0], -1, dtype=int)
        _indep_to_reduced[indep] = np.arange(len(indep))
        rd = _indep_to_reduced[inp.fixed_dofs]
        fixed_reduced = rd[rd >= 0]

        # 縮退系にBC適用（status-312: _zero_sparse_rowsベクトル化）
        K_red_csc = K_red.tocsc()
        if len(fixed_reduced) > 0:
            K_red_csr = K_red_csc.tocsr()
            _zero_sparse_rows(K_red_csr, fixed_reduced)
            K_red_csc = K_red_csr.tocsc()
            _zero_sparse_rows(K_red_csc, fixed_reduced)
            K_red_csr = K_red_csc.tocsr()
            diag_vals = K_red_csr.diagonal()
            diag_vals[fixed_reduced] = 1.0
            K_red_csr.setdiag(diag_vals)
            K_red_csc = K_red_csr.tocsc()
            K_red_csc.eliminate_zeros()
            _rhs_red[fixed_reduced] = 0.0

        try:
            du_red = _sparse_solve(K_red_csc, _rhs_red)
            # 全体系に復元
            du_full = (T @ du_red).A.ravel() if sp.issparse(T @ du_red) else T @ du_red
            return LinearSolveOutput(du=du_full, success=True)
        except Exception:
            return LinearSolveOutput(du=None, success=False)


# ================================================================
# 5. LineSearchUpdateProcess（ステップ 9: Line search + 変位更新）
# ================================================================


@dataclass(frozen=True)
class LineSearchUpdateInput:
    """Line search + 変位更新の入力."""

    u: np.ndarray
    du: np.ndarray
    f_ext: np.ndarray
    fixed_dofs: np.ndarray
    assemble_internal_force: object
    res_u_norm: float
    f_c: np.ndarray
    use_line_search: bool
    line_search_max_steps: int
    du_norm_cap: float


@dataclass(frozen=True)
class LineSearchUpdateOutput:
    """Line search + 変位更新の出力."""

    du_scaled: np.ndarray
    scale_factor: float


class LineSearchUpdateProcess(
    SolverProcess[LineSearchUpdateInput, LineSearchUpdateOutput],
):
    """Line search によるステップスケーリング + du_norm_cap 適用."""

    meta = ProcessMeta(
        name="LineSearchUpdate",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_solver.md",
    )
    uses = [NCPLineSearchProcess]

    def process(self, inp: LineSearchUpdateInput) -> LineSearchUpdateOutput:
        scale_factor = 1.0
        if inp.use_line_search:
            _ls_out = NCPLineSearchProcess().process(
                NCPLineSearchInput(
                    u=inp.u,
                    du=inp.du,
                    f_ext=inp.f_ext,
                    fixed_dofs=inp.fixed_dofs,
                    assemble_internal_force=inp.assemble_internal_force,
                    res_u_norm=inp.res_u_norm,
                    max_steps=inp.line_search_max_steps,
                    f_c=inp.f_c,
                    diverge_factor=3.0,
                )
            )
            scale_factor *= _ls_out.alpha
        if inp.du_norm_cap > 0.0:
            _du_n = float(np.linalg.norm(scale_factor * inp.du))
            _u_ref_n = max(float(np.linalg.norm(inp.u)), 1.0)
            if _du_n > inp.du_norm_cap * _u_ref_n:
                scale_factor *= inp.du_norm_cap * _u_ref_n / _du_n
        du_scaled = scale_factor * inp.du
        return LineSearchUpdateOutput(du_scaled=du_scaled, scale_factor=scale_factor)


# UzawaUpdateProcess は status-222 で削除。復元手順は status-222.md 参照。


# ================================================================
# 6. TangentFDDiagnosticProcess（接線剛性FD方向診断: status-256）
# ================================================================


@dataclass(frozen=True)
class TangentFDDiagnosticInput:
    """接線剛性FD方向診断の入力."""

    u: np.ndarray
    du: np.ndarray
    R_u: np.ndarray
    K_T: sp.spmatrix
    mpc_transform: sp.spmatrix | None = None
    fixed_dofs: np.ndarray | None = None
    eps: float = 1e-7
    compute_residual: object | None = None  # u → R_u を計算する callable
    active_contact_dofs: np.ndarray | None = None  # gap<0 ペアの関連DOF（status-260）
    # f_c単独FD検証（status-290: K_c精度の直接評価）
    compute_contact_force: object | None = None  # u → f_c を計算する callable
    K_c: sp.spmatrix | None = None  # 接触接線剛性行列（K_matのみ or K_mat+K_geo+K_st）


@dataclass(frozen=True)
class TangentFDDiagnosticOutput:
    """接線剛性FD方向診断の出力."""

    directional_ratio: float  # ||R(u+eps*du)||/||R(u)||: <1なら方向有効
    fd_directional_deriv: float  # (R(u+eps*du) - R(u))/eps · du/||du||
    analytical_directional_deriv: float  # K_T @ du · du / ||du||
    deriv_agreement: float  # |fd - analytical| / max(|fd|, |analytical|)
    active_dof_rel_err: float = -1.0  # 活性DOFのみの相対誤差（-1=未計算, status-260）
    fc_rel_err: float = -1.0  # f_c単独のFD相対誤差（-1=未計算, status-290）
    report: str = ""  # 診断レポート文字列


class TangentFDDiagnosticProcess(
    SolverProcess[TangentFDDiagnosticInput, TangentFDDiagnosticOutput],
):
    """接線剛性の方向有効性を有限差分で検証する診断 Process.

    status-256: MPC+接触でNR方向が残差を減少させない問題を診断。

    1. 方向有効性: ||R(u + eps*du)|| vs ||R(u)||
       → <1 なら du 方向に残差が減少（Newton方向として有効）
    2. 方向微分整合性: K_T @ du vs (R(u+eps*du) - R(u)) / eps
       → 解析的接線と実際の残差変化の整合性を検証
    3. MPC系での検証: T^T K T @ du_red vs T^T (R(u+eps*du) - R(u)) / eps
    """

    meta = ProcessMeta(
        name="TangentFDDiagnostic",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_solver.md",
    )

    def process(self, inp: TangentFDDiagnosticInput) -> TangentFDDiagnosticOutput:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("接線剛性FD方向診断（status-256）")
        lines.append("=" * 60)

        du_norm = float(np.linalg.norm(inp.du))
        R_norm = float(np.linalg.norm(inp.R_u))
        lines.append(f"  ||du|| = {du_norm:.4e}")
        lines.append(f"  ||R_u|| = {R_norm:.4e}")

        if du_norm < 1e-30 or R_norm < 1e-30:
            return TangentFDDiagnosticOutput(
                directional_ratio=1.0,
                fd_directional_deriv=0.0,
                analytical_directional_deriv=0.0,
                deriv_agreement=0.0,
                report="\n".join(lines) + "\n  du or R_u is zero — skip",
            )

        # 解析的方向微分: dR/du @ du ≈ K_T @ du（符号注意: R = f_int - f_ext, K_T = dR/du）
        K_du = inp.K_T @ inp.du
        analytical_dd = float(np.dot(K_du, inp.du)) / du_norm

        # MPC系での検証
        if inp.mpc_transform is not None:
            T = inp.mpc_transform
            R_red = T.T @ inp.R_u
            K_red_du = T.T @ K_du
            lines.append(f"  ||R_red|| = {float(np.linalg.norm(R_red)):.4e}")
            lines.append(f"  ||K_red @ du|| = {float(np.linalg.norm(K_red_du)):.4e}")

            # R_red と K_red_du の方向整合性
            R_red_norm = float(np.linalg.norm(R_red))
            K_red_du_norm = float(np.linalg.norm(K_red_du))
            if R_red_norm > 1e-30 and K_red_du_norm > 1e-30:
                cos_angle = float(np.dot(R_red, K_red_du)) / (R_red_norm * K_red_du_norm)
                lines.append(f"  cos(R_red, K_red@du) = {cos_angle:.4f}")
                # NR では K_T du = -R_u なので cos ≈ -1 が期待値
                if cos_angle > -0.5:
                    lines.append(f"  ⚠ 方向不整合: cos={cos_angle:.4f} (期待: ≈-1.0)")

        # FD方向検証（compute_residual がある場合）
        fd_dd = 0.0
        directional_ratio = 1.0
        active_dof_rel_err = -1.0
        if inp.compute_residual is not None:
            eps = inp.eps
            # 一貫性のため R(u) も compute_residual で再計算（status-257）
            # NRループ内のリラクゼーション有無に関わらず正確なFD比較を保証
            R_base = inp.compute_residual(inp.u)
            R_base_norm = float(np.linalg.norm(R_base))
            u_pert = inp.u + eps * inp.du
            R_pert = inp.compute_residual(u_pert)
            R_pert_norm = float(np.linalg.norm(R_pert))
            directional_ratio = R_pert_norm / R_base_norm if R_base_norm > 1e-30 else 1.0

            # FD方向微分（一貫した基準点）
            dR = (R_pert - R_base) / eps
            fd_dd = float(np.dot(dR, inp.du)) / du_norm

            # R_u（NR残差）と compute_residual(u) の乖離を報告
            _R_gap = float(np.linalg.norm(R_base - inp.R_u))
            if _R_gap > 1e-10 * max(R_norm, R_base_norm, 1e-30):
                lines.append(
                    f"  ⚠ R_u vs compute_residual(u) 乖離: {_R_gap:.4e}"
                    f" (リラクゼーション/動的項の差異)"
                )

            lines.append(f"  ||R_base|| = {R_base_norm:.4e}")
            lines.append(f"  ||R(u+eps*du)|| = {R_pert_norm:.4e}")
            lines.append(f"  方向有効性: ||R(u+eps*du)||/||R(u)|| = {directional_ratio:.6f}")
            if directional_ratio >= 1.0:
                lines.append("  ⚠ du方向で残差が増加 — Newton方向が無効")
            else:
                lines.append("  ✓ du方向で残差が減少 — Newton方向は有効")

            lines.append(f"  FD方向微分 = {fd_dd:.4e}")
            lines.append(f"  解析方向微分 = {analytical_dd:.4e}")

            # 全体系でのFD vs 解析比較（MPC変換前）
            dR_arr = np.asarray(dR).ravel()
            K_du_arr = np.asarray(K_du).ravel()
            _full_diff = dR_arr - K_du_arr
            _full_diff_norm = float(np.linalg.norm(_full_diff))
            _full_ref = max(
                float(np.linalg.norm(dR_arr)),
                float(np.linalg.norm(K_du_arr)),
                1e-30,
            )
            _full_rel_err = _full_diff_norm / _full_ref
            lines.append(f"  全体系 K@du: FD vs 解析 相対誤差 = {_full_rel_err:.4e}")
            if _full_rel_err > 0.1:
                lines.append("  ⚠ 全体系で接線剛性不整合 — K_c自体が不正確")
                # 全体系DOF別エラートップ5
                _full_abs_diff = np.abs(_full_diff)
                _top5_full = np.argsort(_full_abs_diff)[::-1][:5]
                _ndpn = 6  # ndof_per_node（デフォルト6DOF）
                lines.append("  全体系不整合DOF上位5件:")
                for idx in _top5_full:
                    if _full_abs_diff[idx] > 1e-30:
                        lines.append(
                            f"    dof={idx} (node={idx // _ndpn}, "
                            f"comp={idx % _ndpn}): "
                            f"FD={dR_arr[idx]:.4e}, "
                            f"analytical={K_du_arr[idx]:.4e}, "
                            f"diff={_full_diff[idx]:.4e}"
                        )
                # ── status-290: comp別不整合分布 ──
                # 各DOF成分(0=x,1=y,2=z,3=θx,4=θy,5=θz)ごとの不整合量を集計
                _n_nodes_full = len(dR_arr) // _ndpn
                _comp_err = np.zeros(_ndpn)
                for c in range(_ndpn):
                    _cdofs = np.arange(c, len(dR_arr), _ndpn)
                    _comp_err[c] = float(np.linalg.norm(_full_diff[_cdofs]))
                _comp_total = float(np.linalg.norm(_full_diff))
                _comp_names = ["x", "y", "z", "θx", "θy", "θz"]
                _comp_str = ", ".join(
                    f"{_comp_names[c]}:{_comp_err[c]:.2e}({100 * _comp_err[c] / max(_comp_total, 1e-30):.0f}%)"
                    for c in range(_ndpn)
                )
                lines.append(f"  comp別不整合: [{_comp_str}]")
                # ── 接触DOF vs 構造DOF分離 ──
                if inp.active_contact_dofs is not None and len(inp.active_contact_dofs) > 0:
                    _ac = inp.active_contact_dofs
                    _non_ac = np.setdiff1d(np.arange(len(dR_arr)), _ac)
                    _ac_err = float(np.linalg.norm(_full_diff[_ac]))
                    _non_ac_err = float(np.linalg.norm(_full_diff[_non_ac]))
                    lines.append(
                        f"  接触DOF不整合: {_ac_err:.2e} ({len(_ac)} DOFs), "
                        f"構造DOF不整合: {_non_ac_err:.2e} ({len(_non_ac)} DOFs)"
                    )

            # 活性DOFのみの精度評価（status-260: 活性集合変化を除外）
            if inp.active_contact_dofs is not None and len(inp.active_contact_dofs) > 0:
                _ac_dofs = inp.active_contact_dofs
                _ac_fd = dR_arr[_ac_dofs]
                _ac_an = K_du_arr[_ac_dofs]
                _ac_diff = _ac_fd - _ac_an
                _ac_diff_norm = float(np.linalg.norm(_ac_diff))
                _ac_ref = max(
                    float(np.linalg.norm(_ac_fd)),
                    float(np.linalg.norm(_ac_an)),
                    1e-30,
                )
                active_dof_rel_err = _ac_diff_norm / _ac_ref
                lines.append(
                    f"  活性DOF K@du: FD vs 解析 相対誤差 = {active_dof_rel_err:.4e}"
                    f" ({len(_ac_dofs)} DOFs)"
                )
                if active_dof_rel_err > 0.1:
                    lines.append("  ⚠ 活性DOFで接線剛性不整合")
                else:
                    lines.append("  ✓ 活性DOFの接線剛性は整合")

            # MPC系でのFD検証
            if inp.mpc_transform is not None:
                T = inp.mpc_transform
                dR_red = T.T @ dR
                fd_K_red_du = dR_red  # ≈ K_red @ du （eps除算済み）
                K_red_du_analytical = T.T @ K_du
                diff = fd_K_red_du - K_red_du_analytical
                diff_norm = float(np.linalg.norm(diff))
                ref_norm = max(
                    float(np.linalg.norm(fd_K_red_du)),
                    float(np.linalg.norm(K_red_du_analytical)),
                    1e-30,
                )
                rel_err = diff_norm / ref_norm
                lines.append(f"  MPC縮退系 K_red@du: FD vs 解析 相対誤差 = {rel_err:.4e}")
                if rel_err > 0.1:
                    lines.append("  ⚠ MPC縮退系で接線剛性不整合 (rel_err > 10%)")

                    # DOF別エラーランキング（上位5件）
                    abs_diff = np.abs(np.asarray(diff).ravel())
                    top5 = np.argsort(abs_diff)[::-1][:5]
                    lines.append("  不整合DOF上位5件（縮退系index）:")
                    for idx in top5:
                        _fd_v = float(np.asarray(fd_K_red_du).ravel()[idx])
                        _an_v = float(np.asarray(K_red_du_analytical).ravel()[idx])
                        _df_v = float(abs_diff[idx])
                        if _df_v > 1e-30:
                            lines.append(
                                f"    red_dof={idx}: "
                                f"FD={_fd_v:.4e}, "
                                f"analytical={_an_v:.4e}, "
                                f"diff={_df_v:.4e}"
                            )
        else:
            lines.append("  (compute_residual未提供 — FD検証スキップ)")
            lines.append(f"  解析方向微分 = {analytical_dd:.4e}")

        # 整合度
        max_dd = max(abs(fd_dd), abs(analytical_dd), 1e-30)
        deriv_agreement = abs(fd_dd - analytical_dd) / max_dd

        lines.append(f"  方向微分整合度: |FD-解析|/max = {deriv_agreement:.4e}")

        # ── f_c 単独FD検証（status-290: K_c精度の直接評価） ──
        fc_rel_err = -1.0
        if inp.compute_contact_force is not None and inp.K_c is not None:
            lines.append("-" * 60)
            lines.append("  f_c 単独FD検証（K_c精度）")
            eps = inp.eps
            fc_base = inp.compute_contact_force(inp.u)
            fc_pert = inp.compute_contact_force(inp.u + eps * inp.du)
            dfc = (fc_pert - fc_base) / eps  # FD: ∂f_c/∂u @ du
            kc_du = np.asarray(inp.K_c @ inp.du).ravel()  # 解析: K_c @ du
            # 符号注意: f_c は R_u = f_int + f_c - f_ext で使われる。
            # tangent() は ∂f_c/∂u を返す。evaluate() の f_c は符号反転済み(-f_c_raw)。
            # FD: ((-f_c_raw_pert) - (-f_c_raw_base))/eps = -(f_c_raw_pert - f_c_raw_base)/eps
            # K_c @ du は ∂(-f_c_raw)/∂u @ du = K_c @ du
            # → dfc と kc_du を直接比較可能
            _fc_diff = dfc - kc_du
            _fc_diff_norm = float(np.linalg.norm(_fc_diff))
            _fc_ref = max(float(np.linalg.norm(dfc)), float(np.linalg.norm(kc_du)), 1e-30)
            fc_rel_err = _fc_diff_norm / _fc_ref
            lines.append(f"  ||∂f_c/∂u·du (FD)|| = {float(np.linalg.norm(dfc)):.4e}")
            lines.append(f"  ||K_c @ du|| = {float(np.linalg.norm(kc_du)):.4e}")
            lines.append(f"  f_c FD相対誤差 = {fc_rel_err:.4e}")
            if fc_rel_err > 0.1:
                lines.append("  ⚠ K_c自体が不正確（f_c単独で10%超の不整合）")
            else:
                lines.append("  ✓ K_cは整合（問題はK_struct側にある可能性）")
            # comp別分解
            _ndpn = 6
            if len(dfc) >= _ndpn:
                _comp_names = ["x", "y", "z", "θx", "θy", "θz"]
                _fc_comp_parts = []
                _fc_comp_total = max(_fc_diff_norm, 1e-30)
                for c in range(min(_ndpn, 6)):
                    _cdofs = np.arange(c, len(dfc), _ndpn)
                    _ce = float(np.linalg.norm(_fc_diff[_cdofs]))
                    _fc_comp_parts.append(
                        f"{_comp_names[c]}={_ce:.1e}/{100 * _ce / _fc_comp_total:.0f}%"
                    )
                lines.append(f"  f_c comp別不整合: [{', '.join(_fc_comp_parts)}]")
            # DOF上位5件
            _fc_abs_diff = np.abs(_fc_diff)
            _fc_top5 = np.argsort(_fc_abs_diff)[::-1][:5]
            lines.append("  f_c不整合DOF上位5件:")
            for idx in _fc_top5:
                if _fc_abs_diff[idx] > 1e-30:
                    lines.append(
                        f"    dof={idx} (node={idx // _ndpn}, comp={idx % _ndpn}): "
                        f"FD={dfc[idx]:.4e}, K_c={kc_du[idx]:.4e}, diff={_fc_diff[idx]:.4e}"
                    )

        lines.append("=" * 60)

        return TangentFDDiagnosticOutput(
            directional_ratio=directional_ratio,
            fd_directional_deriv=fd_dd,
            analytical_directional_deriv=analytical_dd,
            deriv_agreement=deriv_agreement,
            active_dof_rel_err=active_dof_rel_err,
            fc_rel_err=fc_rel_err,
            report="\n".join(lines),
        )
