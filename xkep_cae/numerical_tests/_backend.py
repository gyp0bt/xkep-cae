"""数値試験バックエンド — コールバック注入レジストリ.

xkep_cae 内から deprecated パッケージへの直接 import を回避する（C14 準拠）。
テスト conftest.py 等でバックエンドを注入してから数値試験を実行する。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

from xkep_cae.numerical_tests.core import NumericalTestConfig


# ---------------------------------------------------------------------------
# Protocol 定義
# ---------------------------------------------------------------------------
class DirichletApplier(Protocol):
    """Dirichlet 境界条件の適用."""

    def __call__(
        self,
        K: sp.spmatrix,
        f: np.ndarray,
        fixed_dofs: np.ndarray,
    ) -> tuple[sp.spmatrix, np.ndarray]: ...


class LinearSolver(Protocol):
    """線形連立方程式ソルバー."""

    def __call__(
        self,
        K: sp.spmatrix,
        f: np.ndarray,
        *,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, dict]: ...


class KeFunc(Protocol):
    """要素剛性行列関数: coords (2, ndim) → Ke (edof, edof)."""

    def __call__(self, coords: np.ndarray) -> np.ndarray: ...


class MassMatrixFunc(Protocol):
    """要素質量行列関数: coords (2, ndim) → Me (edof, edof)."""

    def __call__(self, coords: np.ndarray, *args: Any) -> np.ndarray: ...


KeFuncFactory = Callable[[Any, dict], KeFunc]
SectionForceComputer = Callable[
    [NumericalTestConfig, dict, np.ndarray, np.ndarray, np.ndarray], list
]
MassAssembler2D = Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]
MassAssembler3D = Callable[[np.ndarray, np.ndarray, float, float, float, float], np.ndarray]
NLAssemblerFactory = Callable[
    [np.ndarray, np.ndarray, float, float, float, float, float, float, float, float],
    tuple[Callable, Callable],
]
TransientSolver = Callable[..., Any]


# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------
class BackendRegistry:
    """要素剛性・BC・ソルバー・質量行列・動的ソルバーのバックエンド登録.

    テスト conftest.py 等で configure() を呼び出してバックエンドを注入する。

    使用例::

        from xkep_cae.numerical_tests._backend import backend
        backend.configure(
            apply_dirichlet=...,
            solve=...,
            ke_func_factory=...,
            section_force_computer=...,
        )

    """

    def __init__(self) -> None:
        # 静的試験用（必須）
        self._apply_dirichlet: DirichletApplier | None = None
        self._solve: LinearSolver | None = None
        self._ke_func_factory: KeFuncFactory | None = None
        self._section_force_computer: SectionForceComputer | None = None
        # 周波数応答用（オプション）
        self._mass_2d_consistent: MassAssembler2D | None = None
        self._mass_3d_consistent: MassAssembler3D | None = None
        self._mass_2d_lumped: MassAssembler2D | None = None
        self._mass_3d_lumped: MassAssembler3D | None = None
        self._beam2d_lumped_mass_local: Callable | None = None
        self._beam3d_lumped_mass_local: Callable | None = None
        self._beam2d_mass_global: Callable | None = None
        self._beam3d_mass_global: Callable | None = None
        self._beam3d_length_and_direction: Callable | None = None
        # 動的試験用（オプション）
        self._transient_config_class: type | None = None
        self._transient_solver: TransientSolver | None = None
        self._cr_assembler_factory: NLAssemblerFactory | None = None
        self._cosserat_nl_assembler_factory: NLAssemblerFactory | None = None

    def configure(
        self,
        *,
        apply_dirichlet: DirichletApplier,
        solve: LinearSolver,
        ke_func_factory: KeFuncFactory,
        section_force_computer: SectionForceComputer,
    ) -> None:
        """静的試験バックエンドを設定する."""
        self._apply_dirichlet = apply_dirichlet
        self._solve = solve
        self._ke_func_factory = ke_func_factory
        self._section_force_computer = section_force_computer

    def configure_frequency(
        self,
        *,
        mass_2d_consistent: MassAssembler2D | None = None,
        mass_3d_consistent: MassAssembler3D | None = None,
        mass_2d_lumped: MassAssembler2D | None = None,
        mass_3d_lumped: MassAssembler3D | None = None,
        beam2d_lumped_mass_local: Callable | None = None,
        beam3d_lumped_mass_local: Callable | None = None,
        beam2d_mass_global: Callable | None = None,
        beam3d_mass_global: Callable | None = None,
        beam3d_length_and_direction: Callable | None = None,
    ) -> None:
        """周波数応答バックエンドを設定する."""
        self._mass_2d_consistent = mass_2d_consistent
        self._mass_3d_consistent = mass_3d_consistent
        self._mass_2d_lumped = mass_2d_lumped
        self._mass_3d_lumped = mass_3d_lumped
        self._beam2d_lumped_mass_local = beam2d_lumped_mass_local
        self._beam3d_lumped_mass_local = beam3d_lumped_mass_local
        self._beam2d_mass_global = beam2d_mass_global
        self._beam3d_mass_global = beam3d_mass_global
        self._beam3d_length_and_direction = beam3d_length_and_direction

    def configure_dynamic(
        self,
        *,
        transient_config_class: type,
        transient_solver: TransientSolver,
        cr_assembler_factory: NLAssemblerFactory | None = None,
        cosserat_nl_assembler_factory: NLAssemblerFactory | None = None,
    ) -> None:
        """動的試験バックエンドを設定する."""
        self._transient_config_class = transient_config_class
        self._transient_solver = transient_solver
        self._cr_assembler_factory = cr_assembler_factory
        self._cosserat_nl_assembler_factory = cosserat_nl_assembler_factory

    def reset(self) -> None:
        """全バックエンドをリセットする."""
        self.__init__()  # type: ignore[misc]

    # --- 静的試験用プロパティ ---

    @property
    def is_configured(self) -> bool:
        return all(
            [
                self._apply_dirichlet is not None,
                self._solve is not None,
                self._ke_func_factory is not None,
                self._section_force_computer is not None,
            ]
        )

    @property
    def apply_dirichlet(self) -> DirichletApplier:
        if self._apply_dirichlet is None:
            raise RuntimeError(
                "Backend 未設定。backend.configure() で "
                "apply_dirichlet/solve/ke_func_factory/section_force_computer を注入してください。"
            )
        return self._apply_dirichlet

    @property
    def solve(self) -> LinearSolver:
        if self._solve is None:
            raise RuntimeError("Backend 未設定（solve）。")
        return self._solve

    @property
    def ke_func_factory(self) -> KeFuncFactory:
        if self._ke_func_factory is None:
            raise RuntimeError("Backend 未設定（ke_func_factory）。")
        return self._ke_func_factory

    @property
    def section_force_computer(self) -> SectionForceComputer:
        if self._section_force_computer is None:
            raise RuntimeError("Backend 未設定（section_force_computer）。")
        return self._section_force_computer

    # --- 周波数応答用プロパティ ---

    @property
    def mass_2d_consistent(self) -> MassAssembler2D:
        if self._mass_2d_consistent is None:
            raise RuntimeError(
                "Backend 未設定（mass_2d_consistent）。configure_frequency() を呼んでください。"
            )
        return self._mass_2d_consistent

    @property
    def mass_3d_consistent(self) -> MassAssembler3D:
        if self._mass_3d_consistent is None:
            raise RuntimeError(
                "Backend 未設定（mass_3d_consistent）。configure_frequency() を呼んでください。"
            )
        return self._mass_3d_consistent

    @property
    def mass_2d_lumped(self) -> MassAssembler2D:
        if self._mass_2d_lumped is None:
            raise RuntimeError(
                "Backend 未設定（mass_2d_lumped）。configure_frequency() を呼んでください。"
            )
        return self._mass_2d_lumped

    @property
    def mass_3d_lumped(self) -> MassAssembler3D:
        if self._mass_3d_lumped is None:
            raise RuntimeError(
                "Backend 未設定（mass_3d_lumped）。configure_frequency() を呼んでください。"
            )
        return self._mass_3d_lumped

    @property
    def beam2d_lumped_mass_local(self) -> Callable:
        if self._beam2d_lumped_mass_local is None:
            raise RuntimeError("Backend 未設定（beam2d_lumped_mass_local）。")
        return self._beam2d_lumped_mass_local

    @property
    def beam3d_lumped_mass_local(self) -> Callable:
        if self._beam3d_lumped_mass_local is None:
            raise RuntimeError("Backend 未設定（beam3d_lumped_mass_local）。")
        return self._beam3d_lumped_mass_local

    @property
    def beam2d_mass_global(self) -> Callable:
        if self._beam2d_mass_global is None:
            raise RuntimeError("Backend 未設定（beam2d_mass_global）。")
        return self._beam2d_mass_global

    @property
    def beam3d_mass_global(self) -> Callable:
        if self._beam3d_mass_global is None:
            raise RuntimeError("Backend 未設定（beam3d_mass_global）。")
        return self._beam3d_mass_global

    @property
    def beam3d_length_and_direction(self) -> Callable:
        if self._beam3d_length_and_direction is None:
            raise RuntimeError("Backend 未設定（beam3d_length_and_direction）。")
        return self._beam3d_length_and_direction

    # --- 動的試験用プロパティ ---

    @property
    def transient_config_class(self) -> type:
        if self._transient_config_class is None:
            raise RuntimeError(
                "Backend 未設定（transient_config_class）。configure_dynamic() を呼んでください。"
            )
        return self._transient_config_class

    @property
    def transient_solver(self) -> TransientSolver:
        if self._transient_solver is None:
            raise RuntimeError(
                "Backend 未設定（transient_solver）。configure_dynamic() を呼んでください。"
            )
        return self._transient_solver

    @property
    def cr_assembler_factory(self) -> NLAssemblerFactory:
        if self._cr_assembler_factory is None:
            raise RuntimeError("Backend 未設定（cr_assembler_factory）。")
        return self._cr_assembler_factory

    @property
    def cosserat_nl_assembler_factory(self) -> NLAssemblerFactory:
        if self._cosserat_nl_assembler_factory is None:
            raise RuntimeError("Backend 未設定（cosserat_nl_assembler_factory）。")
        return self._cosserat_nl_assembler_factory


# モジュールレベルシングルトン
backend = BackendRegistry()
