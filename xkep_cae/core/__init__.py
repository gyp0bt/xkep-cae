"""プロセスアーキテクチャ基盤.

AbstractProcess + Strategy Protocol によるソルバー契約化フレームワーク。
"""

from xkep_cae.core.base import AbstractProcess, ProcessMeta, ProcessMetaclass
from xkep_cae.core.categories import (
    BatchProcess,
    CompatibilityProcess,
    PostProcess,
    PreProcess,
    SolverProcess,
    VerifyProcess,
)
from xkep_cae.core.data import (
    AssembleCallbacks,
    BoundaryData,
    ContactFrictionInputData,
    ContactSetupData,
    MeshData,
    SolverResultData,
    SolverStrategies,
    VerifyInput,
    VerifyResult,
    default_strategies,
)
from xkep_cae.core.registry import ProcessRegistry
from xkep_cae.core.runner import ExecutionContext, ProcessRunner
from xkep_cae.core.slots import StrategySlot, collect_strategy_slots, collect_strategy_types
from xkep_cae.core.testing import binds_to
from xkep_cae.core.tree import NodeType, ProcessNode, ProcessTree

__all__ = [
    "AbstractProcess",
    "ProcessMeta",
    "ProcessMetaclass",
    "PreProcess",
    "SolverProcess",
    "PostProcess",
    "VerifyProcess",
    "BatchProcess",
    "CompatibilityProcess",
    "binds_to",
    "ProcessTree",
    "ProcessNode",
    "NodeType",
    "MeshData",
    "BoundaryData",
    "ContactSetupData",
    "AssembleCallbacks",
    "ContactFrictionInputData",
    "SolverResultData",
    "SolverStrategies",
    "VerifyInput",
    "VerifyResult",
    "ProcessRegistry",
    "ProcessRunner",
    "ExecutionContext",
    "StrategySlot",
    "collect_strategy_slots",
    "collect_strategy_types",
    "default_strategies",
]
