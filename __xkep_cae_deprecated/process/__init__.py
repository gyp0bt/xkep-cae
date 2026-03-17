"""プロセスアーキテクチャ基盤.

AbstractProcess + Strategy Protocol によるソルバー契約化フレームワーク。

公開API:
    - AbstractProcess, ProcessMeta, ProcessMetaclass: 基底クラス群
    - PreProcess, SolverProcess, PostProcess, VerifyProcess, BatchProcess: カテゴリ
    - binds_to: テスト1:1紐付けデコレータ
    - ProcessTree, ProcessNode, NodeType: 実行グラフ
    - データ型: MeshData, BoundaryData, ContactSetupData, etc.
"""

from __xkep_cae_deprecated.process.base import AbstractProcess, ProcessMeta, ProcessMetaclass
from __xkep_cae_deprecated.process.categories import (
    BatchProcess,
    CompatibilityProcess,
    PostProcess,
    PreProcess,
    SolverProcess,
    VerifyProcess,
)
from __xkep_cae_deprecated.process.data import (
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
from __xkep_cae_deprecated.process.registry import ProcessRegistry
from __xkep_cae_deprecated.process.runner import ExecutionContext, ProcessRunner
from __xkep_cae_deprecated.process.slots import StrategySlot, collect_strategy_slots, collect_strategy_types
from __xkep_cae_deprecated.process.testing import binds_to
from __xkep_cae_deprecated.process.tree import NodeType, ProcessNode, ProcessTree

__all__ = [
    # 基底
    "AbstractProcess",
    "ProcessMeta",
    "ProcessMetaclass",
    # カテゴリ
    "PreProcess",
    "SolverProcess",
    "PostProcess",
    "VerifyProcess",
    "BatchProcess",
    "CompatibilityProcess",
    # テスト
    "binds_to",
    # グラフ
    "ProcessTree",
    "ProcessNode",
    "NodeType",
    # データ型
    "MeshData",
    "BoundaryData",
    "ContactSetupData",
    "AssembleCallbacks",
    "ContactFrictionInputData",
    "SolverResultData",
    "SolverStrategies",
    "VerifyInput",
    "VerifyResult",
    # レジストリ
    "ProcessRegistry",
    # 実行管理
    "ProcessRunner",
    "ExecutionContext",
    # Strategy Slot
    "StrategySlot",
    "collect_strategy_slots",
    "collect_strategy_types",
    # ファクトリ
    "default_strategies",
]
