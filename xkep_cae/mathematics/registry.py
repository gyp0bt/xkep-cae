"""ProcessContractRegistry と `@verified_by` デコレータ.

status-347（MCDD Phase A-2）で新設。status-346 で定義された
`MathematicalContract` を「どの Process がその契約を宣言しているか」
「どの VerifyProcess がその契約を検証しているか」という紐付け情報と一緒に
保持するレジストリと、検証 Process を紐付けるデコレータを提供する。

責務
----
- Process の **class-level `contracts` ClassVar** から数理契約を収集
- 契約と検証 Process の紐付け（`@verified_by(contract_name, verify_cls)`）
- 紐付け有無クエリ（C18 静的検査の前段）
- dummy VerifyProcess の紐付け拒否（脱法実装 pattern 2 の構造的ガード）

設計方針
--------
1. `ProcessRegistry`（`xkep_cae/core/registry.py`）と **直交** する独立レジストリ。
   既存レジストリを改変しない。
2. 契約の宣言は **class-level `contracts` ClassVar**（``uses`` と同様のパターン）
   または ``ProcessMeta.math_contracts`` フィールドを使用する。両者は
   `__init_subclass__` で自動的に合算される。
3. 検証 Process の紐付けは **`@verified_by(contract_name, verify_cls)`** で行う。
   dummy（`process()` 本体が `pass`/`...` のみ）は AST 検査で拒否。
4. `@verified_by` は **contract_name が Process に宣言されている** ことを
   登録時に検査（scope 外の contract 名で紐付けを作る脱法実装を防ぐ）。

C18 前段（Phase E で完成）
--------------------------
Phase E status-356 で追加される C18 静的検査（「`@verified_by` 紐付けが
全契約に存在するか」）の前段として、本 status では以下 API を提供する:

- `unverified_contracts(process_cls)`: 紐付け未完の契約一覧
- `all_bindings()`: 全紐付けの dump

これらは Phase E の C18 実装から呼ばれる。

関連
----
- 契約型: `xkep_cae/mathematics/contracts.py`
- 設計仕様: `xkep_cae/mathematics/docs/mathematics.md`
- 上位プラン: `/root/.claude/plans/deep-wiggling-seal.md`
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xkep_cae.mathematics.contracts import MathematicalContract

if TYPE_CHECKING:
    from xkep_cae.core.base import AbstractProcess

logger = logging.getLogger(__name__)


class DummyVerifyProcessError(TypeError):
    """`@verified_by` で空の dummy VerifyProcess を紐付けた際の例外.

    脱法実装 pattern 2（『dummy VerifyProcess を `@verified_by` に紐付けて
    C18 を通す』）を構造的に拒否するため、`ProcessContractRegistry.bind_verifier`
    が AST 検査で dummy と判定した場合に送出する。
    """


def _process_class_name(cls_or_name: type | str) -> str:
    """`type` または `str` から一貫してクラス名を取り出す."""
    if isinstance(cls_or_name, str):
        return cls_or_name
    return cls_or_name.__name__


class ProcessContractRegistry:
    """Process-数理契約-検証 Process の三者紐付けレジストリ.

    シングルトン: `ProcessContractRegistry.default()` でグローバルインスタンスに
    アクセス。テスト時は `ProcessContractRegistry()` で新規インスタンスを作成し、
    `_set_default()` で差し替え可能。

    内部状態
    --------
    - `_contracts_by_process`: `{process_class_name: tuple[MathematicalContract, ...]}`
      Process と宣言契約の対応。`register_contracts` で追加。
    - `_verifiers`: `{(process_class_name, contract_name): verify_class}`
      契約 → 検証 Process の紐付け。`bind_verifier` で追加。
    """

    _default_instance: ProcessContractRegistry | None = None

    def __init__(self) -> None:
        self._contracts_by_process: dict[str, tuple[MathematicalContract, ...]] = {}
        self._verifiers: dict[tuple[str, str], type[AbstractProcess]] = {}

    # --- シングルトンアクセス ---

    @classmethod
    def default(cls) -> ProcessContractRegistry:
        """グローバルレジストリを返す（遅延初期化）."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def _set_default(cls, instance: ProcessContractRegistry | None) -> None:
        """テスト用: デフォルトインスタンスを差し替える（`None` でクリア）."""
        cls._default_instance = instance

    # --- 契約の登録・クエリ ---

    def register_contracts(
        self,
        process_cls: type | str,
        contracts: tuple[MathematicalContract, ...],
    ) -> None:
        """Process クラスに契約を登録する.

        同一 Process 内での契約名重複は拒否する（脱法実装 pattern 2 の
        前段ガード: 同名契約で二重カウントする抜け道を塞ぐ）。
        同一 Process が複数回登録された場合は後追い登録を許容（mixin 等を想定）
        するが、名前重複は全登録合算ベースで検査する。

        Args:
            process_cls: 契約を宣言する Process クラス（または名前文字列）。
            contracts: 宣言する `MathematicalContract` インスタンスのタプル。

        Raises:
            TypeError: `contracts` が tuple でない場合、または要素が
                `MathematicalContract` のサブクラスでない場合。
            ValueError: `contracts` 内に同名の契約が含まれる場合、
                または既登録分と合わせて名前衝突する場合。
        """
        if not isinstance(contracts, tuple):
            raise TypeError(
                f"register_contracts: contracts は tuple 必須 (frozen 契約との整合): "
                f"got {type(contracts).__name__}"
            )
        for c in contracts:
            if not isinstance(c, MathematicalContract):
                raise TypeError(
                    f"register_contracts: {c!r} は MathematicalContract のサブクラスではありません"
                )

        name = _process_class_name(process_cls)

        existing = self._contracts_by_process.get(name, ())
        combined = existing + contracts

        # 名前重複の全域チェック（pattern 2 の構造的封じ込め）
        seen: set[str] = set()
        for c in combined:
            if c.name in seen:
                raise ValueError(
                    f"ProcessContractRegistry.register_contracts({name}): "
                    f"契約名 '{c.name}' が重複しています。"
                    "同一 Process 内での contract 名重複は禁止です（脱法実装防止）"
                )
            seen.add(c.name)

        self._contracts_by_process[name] = combined

    def contracts_of(self, process_cls: type | str) -> tuple[MathematicalContract, ...]:
        """Process の宣言契約タプルを返す（未登録は空 tuple）."""
        return self._contracts_by_process.get(_process_class_name(process_cls), ())

    def contract_by_name(
        self, process_cls: type | str, contract_name: str
    ) -> MathematicalContract | None:
        """Process のうち、指定名の契約を返す（なければ None）."""
        for c in self.contracts_of(process_cls):
            if c.name == contract_name:
                return c
        return None

    def all_contracts(self) -> dict[str, tuple[MathematicalContract, ...]]:
        """全 Process の契約登録のスナップショット dict を返す."""
        return dict(self._contracts_by_process)

    # --- 検証 Process の紐付け ---

    def bind_verifier(
        self,
        process_cls: type | str,
        contract_name: str,
        verify_cls: type[AbstractProcess],
    ) -> None:
        """Process の特定契約に VerifyProcess を紐付ける.

        紐付け時の検査:

        1. `verify_cls` が `AbstractProcess` のサブクラスかどうか
        2. `verify_cls` が具象（非抽象）かどうか
        3. 対象契約 `contract_name` が `process_cls` に登録済みかどうか
        4. `verify_cls.process()` が dummy（`pass`/`...` のみ）でないかを
           AST で検査（脱法実装 pattern 2 の構造的封じ込め）

        Raises:
            TypeError: `verify_cls` が AbstractProcess でない、または抽象クラスの場合。
            ValueError: 契約が未登録、または既に別 `verify_cls` が紐付いている場合。
            DummyVerifyProcessError: `verify_cls.process()` が dummy と判定された場合。
        """
        # 遅延 import（循環回避）
        from xkep_cae.core.base import AbstractProcess

        name = _process_class_name(process_cls)
        key = (name, contract_name)

        # 1. サブクラスチェック
        if not (isinstance(verify_cls, type) and issubclass(verify_cls, AbstractProcess)):
            raise TypeError(
                f"@verified_by({contract_name!r}): {verify_cls!r} は "
                "AbstractProcess のサブクラスではありません"
            )

        # 2. 具象クラスチェック
        abstract_methods = getattr(verify_cls, "__abstractmethods__", None)
        if abstract_methods:
            raise TypeError(
                f"@verified_by({contract_name!r}): {verify_cls.__name__} は抽象クラスです "
                f"(未実装 abstract method: {sorted(abstract_methods)}). concrete クラスを指定してください"
            )

        # 3. 契約が登録済みかをチェック
        if self.contract_by_name(process_cls, contract_name) is None:
            known = [c.name for c in self.contracts_of(process_cls)]
            raise ValueError(
                f"@verified_by({contract_name!r}): Process '{name}' に契約 "
                f"'{contract_name}' が宣言されていません。宣言済み契約: {known}"
            )

        # 4. dummy 本体チェック（脱法実装 pattern 2 の構造的封じ込め）
        _reject_dummy_process(verify_cls)

        # 二重紐付けは「同じ verify_cls」なら idempotent、違うクラスならエラー
        existing = self._verifiers.get(key)
        if existing is not None and existing is not verify_cls:
            raise ValueError(
                f"@verified_by({contract_name!r}): Process '{name}' 契約 "
                f"'{contract_name}' は既に {existing.__name__} に紐付いています "
                f"(上書き指定: {verify_cls.__name__})"
            )

        self._verifiers[key] = verify_cls

    def verifier_of(
        self, process_cls: type | str, contract_name: str
    ) -> type[AbstractProcess] | None:
        """Process の特定契約に紐付けられた VerifyProcess を返す（なければ None）."""
        return self._verifiers.get((_process_class_name(process_cls), contract_name))

    def unverified_contracts(self, process_cls: type | str) -> tuple[MathematicalContract, ...]:
        """紐付け VerifyProcess が無い契約のタプルを返す（C18 静的検査の前段）.

        Phase E status-356 の C18（`@verified_by` 紐付け検査）実装から
        呼び出される予定。severity=``soft`` の契約は人間可読ドキュメント扱いの
        ため紐付け必須ではないことを想定しているが、本 API は severity 問わず
        全件返す（フィルタは呼び出し側責務）。
        """
        name = _process_class_name(process_cls)
        return tuple(
            c for c in self.contracts_of(process_cls) if (name, c.name) not in self._verifiers
        )

    def all_bindings(self) -> dict[tuple[str, str], type[AbstractProcess]]:
        """全紐付けのスナップショット dict を返す（C18 検査用）."""
        return dict(self._verifiers)

    # --- テスト用ユーティリティ ---

    def clear(self) -> None:
        """全登録をクリア（テスト用）."""
        self._contracts_by_process.clear()
        self._verifiers.clear()

    def __repr__(self) -> str:
        return (
            f"ProcessContractRegistry("
            f"{len(self._contracts_by_process)} processes, "
            f"{len(self._verifiers)} bindings)"
        )


# ======================================================================
# @verified_by デコレータ
# ======================================================================


def verified_by(
    contract_name: str,
    verify_cls: type[AbstractProcess],
) -> Callable[[type[AbstractProcess]], type[AbstractProcess]]:
    """Process の特定契約と検証 Process を紐付けるデコレータ.

    Args:
        contract_name: `MathematicalContract.name` と一致する契約名。
        verify_cls: 検証を担う `VerifyProcess` の具象クラス。

    Returns:
        Process クラスを受け取って紐付けを `ProcessContractRegistry.default()`
        に追加した上で、元のクラスを返すデコレータ。

    Example:
        >>> from xkep_cae.mathematics import FDConsistencyContract
        >>> # class ContactKcComponentFDDiagnosticProcess(VerifyProcess): ...
        >>> # @verified_by("K_c_fd_consistency", ContactKcComponentFDDiagnosticProcess)
        >>> # class HuberContactForceProcess(SolverProcess[...]):
        >>> #     contracts = (FDConsistencyContract(name="K_c_fd_consistency", ...),)

    Raises:
        DummyVerifyProcessError / TypeError / ValueError:
            `ProcessContractRegistry.bind_verifier` の検査に失敗した場合。

    Notes:
        デコレータ適用時点で `process_cls.contracts`（および
        `process_cls.meta.math_contracts`）が `ProcessContractRegistry` に
        登録されている必要がある。通常は `AbstractProcess.__init_subclass__`
        が自動登録するため、クラス定義が完了した時点で適用すれば問題ない。
    """

    def decorator(process_cls: type[AbstractProcess]) -> type[AbstractProcess]:
        ProcessContractRegistry.default().bind_verifier(process_cls, contract_name, verify_cls)
        return process_cls

    return decorator


# ======================================================================
# 内部ヘルパ: dummy VerifyProcess 検出
# ======================================================================


def _extract_process_method_source(verify_cls: type) -> str | None:
    """`verify_cls.process` のソースを AST で解析可能な形で取り出す.

    メタクラスによる `functools.wraps` ラップがあれば `inspect.unwrap` で
    原型に戻してからソースを取得する。取得不能な場合は `None` を返す。
    """
    method: Any = getattr(verify_cls, "process", None)
    if method is None:
        return None
    try:
        unwrapped = inspect.unwrap(method)
    except ValueError:
        unwrapped = method

    try:
        src = inspect.getsource(unwrapped)
    except (OSError, TypeError) as err:
        logger.debug(
            "_extract_process_method_source: %s.process のソース取得失敗: %s",
            verify_cls.__name__,
            err,
        )
        return None

    # クラス内部のメソッドはインデントがかかっているので dedent
    return textwrap.dedent(src)


def _is_trivial_stmt(stmt: ast.stmt) -> bool:
    """文 `stmt` が trivial（実装不在）と見なせるかを判定する.

    - `pass`
    - 裸の `...`（`Expr(Constant(Ellipsis))`）
    - 裸の `None`（`Expr(Constant(None))`）
    - `return`（値なし）
    - `return None` / `return ...`
    - `raise NotImplementedError(...)`
    """
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        if stmt.value.value is None or stmt.value.value is Ellipsis:
            return True
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return True
        if isinstance(stmt.value, ast.Constant) and stmt.value.value in (None, Ellipsis):
            return True
    if isinstance(stmt, ast.Raise):
        exc = stmt.exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
        if (
            isinstance(exc, ast.Call)
            and isinstance(exc.func, ast.Name)
            and exc.func.id == "NotImplementedError"
        ):
            return True
    return False


def _reject_dummy_process(verify_cls: type) -> None:
    """`verify_cls.process()` が dummy（実装不在）なら `DummyVerifyProcessError`.

    判定基準:

    - 本体が全て `_is_trivial_stmt` を満たす文のみの場合は dummy と判定。
    - docstring は除外（最初が文字列定数 Expr なら読み飛ばす）。
    - ソース取得不能な場合は検査スキップ（動的生成クラス等）。警告ログ。

    脱法実装 pattern 2（『dummy VerifyProcess を紐付けて C18 を通す』）の
    構造的封じ込め。
    """
    src = _extract_process_method_source(verify_cls)
    if src is None:
        logger.warning(
            "_reject_dummy_process: %s.process のソースを取得できないため dummy 検査をスキップします",
            verify_cls.__name__,
        )
        return

    try:
        tree = ast.parse(src)
    except SyntaxError as err:
        logger.warning(
            "_reject_dummy_process: %s.process の AST 解析に失敗したため dummy 検査をスキップ: %s",
            verify_cls.__name__,
            err,
        )
        return

    # モジュールレベルで最初の FunctionDef / AsyncFunctionDef を拾う
    func_defs = [
        node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not func_defs:
        logger.warning(
            "_reject_dummy_process: %s.process の FunctionDef が見つからないため検査スキップ",
            verify_cls.__name__,
        )
        return

    body: list[ast.stmt] = list(func_defs[0].body)

    # 先頭の docstring を除外
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if not body:
        raise DummyVerifyProcessError(
            f"@verified_by: {verify_cls.__name__}.process() の本体が空です "
            "(docstring 以外ありません)。dummy VerifyProcess の紐付けは禁止されています "
            "(脱法実装 pattern 2)"
        )

    if all(_is_trivial_stmt(stmt) for stmt in body):
        raise DummyVerifyProcessError(
            f"@verified_by: {verify_cls.__name__}.process() の本体が trivial です "
            "(pass / None / ... / 裸の return / NotImplementedError のみ)。"
            "dummy VerifyProcess の紐付けは禁止されています (脱法実装 pattern 2)"
        )


__all__ = [
    "DummyVerifyProcessError",
    "ProcessContractRegistry",
    "verified_by",
]
