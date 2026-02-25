"""xkep_cae.core - 要素・構成則・断面の抽象インタフェース定義・戻り値型.

Protocol 階層:
  ElementProtocol              — 線形弾性アセンブリ用（local_stiffness + dof_indices）
  NonlinearElementProtocol     — 幾何学的/材料非線形用（+ internal_force, tangent_stiffness）
  DynamicElementProtocol       — 動解析用（+ mass_matrix）
  ConstitutiveProtocol         — 構成則最小限（tangent のみ）
  PlasticConstitutiveProtocol  — 弾塑性用（+ return_mapping）
"""

from xkep_cae.core.constitutive import ConstitutiveProtocol, PlasticConstitutiveProtocol
from xkep_cae.core.element import (
    DynamicElementProtocol,
    ElementProtocol,
    NonlinearElementProtocol,
)
from xkep_cae.core.results import (
    AssemblyResult,
    DirichletResult,
    FiberAssemblyResult,
    LinearSolveResult,
    PlasticAssemblyResult,
)
from xkep_cae.core.state import (
    CosseratFiberPlasticState,
    CosseratPlasticState,
    PlasticState1D,
)

__all__ = [
    "ElementProtocol",
    "NonlinearElementProtocol",
    "DynamicElementProtocol",
    "ConstitutiveProtocol",
    "PlasticConstitutiveProtocol",
    "PlasticState1D",
    "CosseratPlasticState",
    "CosseratFiberPlasticState",
    "LinearSolveResult",
    "DirichletResult",
    "AssemblyResult",
    "PlasticAssemblyResult",
    "FiberAssemblyResult",
]
