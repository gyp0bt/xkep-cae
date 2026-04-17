"""MCDD（数理契約駆動開発）基盤パッケージ.

status-346 で新設。Process が満たすべき「方程式」「不変量」「残差・誤差定義」
を型として宣言可能にする `MathematicalContract` 型システムを提供する。

設計意図
--------
- Process アーキテクチャの既存契約 (C3〜C17) は「依存関係・メタ情報・コード衛生」
  の静的契約を扱うが、**「この Process が解くべき方程式」「満たすべき不変量」**
  は契約外であった。これを型とレジストリに昇格することで:

  1. `tangent_components()` のような「∂n̂/∂u の x/z 成分が抜けている」欠落が
     **契約違反として機械検出**可能になる
  2. 診断 (TangentFD / KcComponentFD) の散在トリガーを **契約違反 → 診断
     Process の自動 dispatch** に一本化できる
  3. 離散化方程式レジストリ (`docs/math/`) と Process の双方向リンクで
     status-289〜345 の調査文脈が永続化される

背景
----
status-345 で K_geo=0 誤認の根本原因が「`{:5.2f}` 精度バグ」という表示上の
アーティファクトであったことが判明。「診断の診断が必要な状態」を構造的に
解消するため、ユーザー指示で MCDD を最優先タスクとして導入する。

設計仕様: docs/mathematics.md
"""

from xkep_cae.mathematics.contracts import (
    FDConsistencyContract,
    IdentityContract,
    InequalityContract,
    MathematicalContract,
    SymmetryContract,
    TermExpansionContract,
)
from xkep_cae.mathematics.equation_index import (
    DuplicateAnchorError,
    EquationIndex,
    UnresolvedReferenceError,
)
from xkep_cae.mathematics.equation_index import load as load_equation_index
from xkep_cae.mathematics.registry import (
    DummyVerifyProcessError,
    ProcessContractRegistry,
    verified_by,
)

__all__ = [
    # Phase A-1: 契約型
    "MathematicalContract",
    "IdentityContract",
    "InequalityContract",
    "FDConsistencyContract",
    "SymmetryContract",
    "TermExpansionContract",
    # Phase A-2: 紐付けレジストリ + デコレータ
    "ProcessContractRegistry",
    "verified_by",
    "DummyVerifyProcessError",
    # Phase B-2: 台帳アンカー抽出・参照解決
    "EquationIndex",
    "DuplicateAnchorError",
    "UnresolvedReferenceError",
    "load_equation_index",
]
