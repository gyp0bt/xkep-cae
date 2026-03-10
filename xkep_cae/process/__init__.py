"""プロセスアーキテクチャ基盤.

AbstractProcess + Strategy Protocol によるソルバー契約化フレームワーク。

設計仕様: docs/design/process-architecture.md

公開API:
    - AbstractProcess, ProcessMeta, ProcessMetaclass: 基底クラス群
    - PreProcess, SolverProcess, PostProcess, VerifyProcess, BatchProcess: カテゴリ
    - binds_to: テスト1:1紐付けデコレータ
    - ProcessTree, ProcessNode, NodeType: 実行グラフ
    - データ型: MeshData, BoundaryData, ContactSetupData, etc.
"""

from xkep_cae.process.base import AbstractProcess, ProcessMeta, ProcessMetaclass
from xkep_cae.process.categories import (
    BatchProcess,
    PostProcess,
    PreProcess,
    SolverProcess,
    VerifyProcess,
)
from xkep_cae.process.data import (
    AssembleCallbacks,
    BoundaryData,
    ContactSetupData,
    MeshData,
    SolverInputData,
    SolverResultData,
    VerifyInput,
    VerifyResult,
)
from xkep_cae.process.testing import binds_to
from xkep_cae.process.tree import NodeType, ProcessNode, ProcessTree

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
    "SolverInputData",
    "SolverResultData",
    "VerifyInput",
    "VerifyResult",
]
