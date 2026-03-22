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
        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(manager=inp.manager, node_coords=coords_def, freeze_active_set=True)
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
        res_u_norm = float(np.linalg.norm(inp.R_u))
        f_ref = inp.f_ext_ref_norm
        if inp.dynamic_ref and inp.is_first_attempt and res_u_norm > 1e-30:
            f_ref = res_u_norm

        n_active = sum(1 for p in inp.manager.pairs if hasattr(p, "state") and p.state.p_n > 0.0)

        # 力収束
        if res_u_norm / f_ref < inp.tol_force:
            return ConvergenceCheckOutput(
                converged=True,
                convergence_type=ConvergenceType.FORCE,
                res_u_norm=res_u_norm,
                f_ref=f_ref,
                n_active=n_active,
                du_norm=0.0,
                energy=0.0,
                energy_ref=inp.energy_ref,
            )

        # 変位・エネルギー収束は du が必要
        du_norm = 0.0
        energy = 0.0
        energy_ref = inp.energy_ref
        conv_type = ConvergenceType.NOT_CONVERGED

        if inp.du is not None:
            du_norm = float(np.linalg.norm(inp.du))
            u_norm = float(np.linalg.norm(inp.u))

            if u_norm > 1e-30 and du_norm / u_norm < inp.tol_disp:
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

        K_c = inp.contact_force_strategy.tangent(inp.u, inp.manager, inp.k_pen)
        K_T = K_T + K_c

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

        # 摩擦剛性
        if inp.friction_strategy.friction_tangents:
            K_fric = inp.friction_strategy.tangent(inp.u, inp.manager.pairs, inp.mu)
            K_T = K_T + K_fric

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


@dataclass(frozen=True)
class LinearSolveOutput:
    """線形ソルブの出力."""

    du: np.ndarray | None
    success: bool


class LinearSolveProcess(
    SolverProcess[LinearSolveInput, LinearSolveOutput],
):
    """境界条件適用 + 線形ソルブ."""

    meta = ProcessMeta(
        name="LinearSolve",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_solver.md",
    )

    def process(self, inp: LinearSolveInput) -> LinearSolveOutput:
        K_eff = inp.K_T.tocsc()
        _rhs = -inp.R_u.copy()
        for d in inp.fixed_dofs:
            K_eff[d, :] = 0.0
            K_eff[:, d] = 0.0
            K_eff[d, d] = 1.0
            _rhs[d] = 0.0
        K_eff.eliminate_zeros()

        try:
            from scipy.sparse.linalg import spsolve

            du = spsolve(K_eff.tocsc(), _rhs)
            return LinearSolveOutput(du=du, success=True)
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
