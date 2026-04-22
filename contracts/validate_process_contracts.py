"""プロセス契約違反検出スクリプト.

process-architecture.md §13 で定義された契約抜け腐敗シナリオを機械的に検出する:

法律（C: Contract）:
- C3: テスト未紐付けのプロセス（_test_class is None）
- C5: process() 内の未宣言依存（AST解析）
- C6: Strategy Protocol の意味論的契約違反（具象クラス未検証）
- C7: process() のメタクラスラップ漏れ
- C8: _runtime_uses / StrategySlot の動的依存カバー
- C9: frozen dataclass numpy 配列変更検出（execute() チェックサム未実装）
- C11: uses チェーンの推移的依存漏れ
- C12: BatchProcess 具象クラスの順序依存検証
- C13: active プロセスが CompatibilityProcess を uses している場合はエラー
- C14: xkep_cae/ 内から __xkep_cae_deprecated をインポートしていないか検出
- C15: ProcessMeta.document_path で指定されたドキュメントが実在するか検証
- C15(math): MathematicalContract.equation_ref が docs/math/ 台帳で解決可能か検証
  （status-349 Phase B-2 で追加）
- C16: 新パッケージ滅菌 — core/ 以外の全モジュール内のクラス/関数を分類検査
       __init__.py のクラス再エクスポートも型検査対象
- C17: プライベートモジュール dataclass 衛生 — _xxx.py 内の dataclass が
       frozen=True でない場合、またはクラス名が Input/Output で終わらない場合を検出
- C18: `@verified_by` 紐付け検査 — severity in {"hard", "nightly"} の
       `MathematicalContract` ごとに `ProcessContractRegistry` に検証 Process
       が紐付いているか検証（status-357 Phase E、MCDD 脱法 pattern 2 対策）
- C19: `TermExpansionContract.providers` 実在検査 — 列挙された Process クラス名が
       `ProcessRegistry` に実在するか検証（status-357 Phase E、MCDD 脱法 pattern 4 対策）
- C20: `TermExpansionContract` 双方向紐付け整合性検査 — `providers` に列挙された
       Process クラスが、自身の `contracts` ClassVar で **同一の** `TermExpansion
       Contract` を宣言していることを検証（status-358 Phase E、MCDD 脱法 pattern 4
       の rename 亜種「providers 追加のみで provider 側の contracts 宣言漏れ」対策）
- C21: `TermExpansionContract.term_names` / `providers` 重複検出 — 各 Process の
       `contracts` ClassVar に宣言された `TermExpansionContract` について、
       term_names・providers の要素重複を静的に検出（status-360 Phase E、
       __post_init__ が通る同名項の二重宣言シナリオは無いはずだが belt-and-suspenders）
- C22: `contracts` ClassVar 同名契約重複検出 — 同一 Process 内の `contracts`
       ClassVar に同一 `name` の `MathematicalContract` が複数存在するケースを
       検出（status-360 Phase E、`ProcessContractRegistry.register_contracts`
       の動的検査を静的検査で先行させる補強）
- C23: `@verified_by` 検証 Process カテゴリ検査 — `ProcessContractRegistry`
       に登録された VerifyProcess 側が `SolverProcess` もしくは `VerifyProcess`
       カテゴリを継承していることを検証（status-360 Phase E、`bind_verifier`
       の動的検査の静的版。PreProcess 等を verifier に指定する脱法を排除）

条例（O: Ordinance）:
- O1: テストが Process ラッパーのある関数を直接呼び出していないか検出
- O2: BackendRegistry パターン（configure/reset シングルトン）の検出
- O3: テスト conftest での backend.configure() 注入の検出

使用方法:
    python contracts/validate_process_contracts.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap
from pathlib import Path

# プロジェクトルートをパスに追加
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from xkep_cae.core.base import AbstractProcess  # noqa: E402
from xkep_cae.core.registry import ProcessRegistry  # noqa: E402


def _ast_fallback_binds_to(py_file: Path, registry: dict | None = None) -> None:
    """pytest 未インストール環境用: AST で @binds_to(XxxProcess) を検出し紐付け."""
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return

    reg = registry if registry is not None else ProcessRegistry.default()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            # @binds_to(XxxProcess) パターンを検出
            if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name):
                if deco.func.id == "binds_to" and deco.args:
                    arg = deco.args[0]
                    if isinstance(arg, ast.Name):
                        process_name = arg.id
                        if process_name in reg and reg[process_name]._test_class is None:
                            mod_path = py_file.relative_to(_project_root).with_suffix("")
                            mod_name = str(mod_path).replace("/", ".")
                            test_path = f"{mod_name}::{node.name}"
                            reg[process_name]._test_class = test_path


def _import_all_modules() -> None:
    """全プロセスモジュール + テストモジュールをインポートしてレジストリを構築.

    xkep_cae/ 配下の全サブパッケージの .py ファイルを
    ファイルシステム走査で自動検出する。
    """
    # 走査対象ルート: xkep_cae/ 配下の全サブパッケージ + トップレベル tests/
    xkep_root = _project_root / "xkep_cae"
    scan_roots = [d for d in sorted(xkep_root.iterdir()) if d.is_dir() and d.name != "__pycache__"]
    top_tests = _project_root / "tests"
    if top_tests.exists():
        scan_roots.append(top_tests)

    # 除外対象: テストファイル、__init__.py、基盤モジュール
    _SKIP_NAMES = {"__init__", "base", "categories", "data", "slots", "tree", "runner"}

    process_modules = []
    test_files = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for py_file in sorted(scan_root.rglob("*.py")):
            if py_file.parent.name == "__pycache__":
                continue
            # テストファイルはあとで別処理
            if py_file.parent.name == "tests" or py_file.name.startswith("test_"):
                test_files.append(py_file)
                continue
            if py_file.stem in _SKIP_NAMES:
                continue
            if not py_file.stem.isidentifier():
                continue
            mod_path = py_file.relative_to(_project_root).with_suffix("")
            mod_name = str(mod_path).replace("/", ".")
            process_modules.append(mod_name)

    for mod_name in process_modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            print(f"  警告: {mod_name} のインポートに失敗: {e}")

    # テストモジュール（@binds_to 発動用）を走査・インポート
    # pytest がない環境では AST で @binds_to を検出してフォールバック
    for py_file in test_files:
        mod_path = py_file.relative_to(_project_root).with_suffix("")
        mod_name = str(mod_path).replace("/", ".")
        try:
            importlib.import_module(mod_name)
        except Exception:
            _ast_fallback_binds_to(py_file, registry=None)


def _is_test_fixture(cls: type) -> bool:
    """テスト用フィクスチャ（tests/ 配下で定義されたプロセス）かどうか."""
    module = getattr(cls, "__module__", "")
    return ".tests." in module or module.startswith("tests.")


def check_c3_test_binding(registry: dict[str, type]) -> list[str]:
    """C3: テスト未紐付けのプロセスを検出."""
    errors = []
    for name, cls in sorted(registry.items()):
        if hasattr(cls, "meta") and cls.meta.deprecated:
            continue
        if _is_test_fixture(cls):
            continue
        if cls._test_class is None:
            errors.append(f"C3: {name} にテストが紐付けられていない (@binds_to 未使用)")
    return errors


def check_c5_undeclared_deps(registry: dict[str, type]) -> list[str]:
    """C5: process() 内の未宣言依存をAST解析で検出."""
    errors = []
    registry_names = set(registry.keys())

    for name, cls in sorted(registry.items()):
        # deprecated プロセスはスキップ（後継への委譲は正常動作）
        meta = getattr(cls, "meta", None)
        if meta is not None and getattr(meta, "deprecated", False):
            continue
        # ラップ前の元関数を取得
        method = getattr(cls, "process", None)
        if method is None:
            continue
        if hasattr(method, "__wrapped__"):
            method = method.__wrapped__

        try:
            source = inspect.getsource(method)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue

        used_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in registry_names
        }
        declared = {dep.__name__ for dep in cls.uses}
        undeclared = used_names - declared - {name}

        for u in sorted(undeclared):
            errors.append(f"C5: {name}.process() が uses 未宣言の {u} を参照")

    return errors


def check_c7_metaclass_wrap(registry: dict[str, type]) -> list[str]:
    """C7: process() のメタクラスラップ漏れを検出."""
    errors = []
    for name, cls in sorted(registry.items()):
        method = getattr(cls, "process", None)
        if method is None:
            continue
        # メタクラスでラップ済みなら __wrapped__ を持つ
        if not hasattr(method, "__wrapped__"):
            errors.append(f"C7: {name}.process() がメタクラスでラップされていない")
    return errors


def check_c8_strategy_slot(registry: dict[str, type]) -> list[str]:
    """C8: StrategySlot の動的依存カバーを検証.

    Phase 9-B: _runtime_uses 廃止後の StrategySlot ベース検証。
    StrategySlot を持つクラスに対して:
    1. デフォルト引数でインスタンス化が成功すること
    2. collect_strategy_types() が空でないこと（strategy が正しく注入されていること）
    3. effective_uses() が StrategySlot の型を含むこと
    を検証する。
    """
    from xkep_cae.core.slots import collect_strategy_slots, collect_strategy_types

    errors = []
    for name, cls in sorted(registry.items()):
        slots = collect_strategy_slots(cls)
        if not slots:
            continue

        # デフォルト引数でインスタンス化を試みる
        try:
            instance = cls()
        except Exception as e:
            errors.append(f"C8: {name} のデフォルトインスタンス化に失敗: {e}")
            continue

        slot_types = collect_strategy_types(instance)
        if not slot_types:
            errors.append(
                f"C8: {name} の StrategySlot が全て空（strategy 注入に失敗している可能性）"
            )
            continue

        # effective_uses() が StrategySlot の型を含むか
        effective = instance.effective_uses()
        effective_ids = {id(dep) for dep in effective}
        for dep in slot_types:
            if id(dep) not in effective_ids:
                dep_name = dep.__name__ if hasattr(dep, "__name__") else str(dep)
                errors.append(
                    f"C8: {name} の StrategySlot 型 {dep_name} が effective_uses() に含まれていない"
                )

    return errors


def check_c6_strategy_semantics(registry: dict[str, type]) -> list[str]:
    """C6: Strategy Protocol の意味論的契約テストが存在するか検証.

    各 Strategy 具象クラスが Protocol の意味論的不変条件を満たすことを
    テストで保証しているかチェックする。ここでは「テストの存在」を検証し、
    実際の意味論チェックは test_contracts.py で実施する。
    """
    errors = []

    try:
        from xkep_cae.core.strategies.protocols import (
            ContactForceStrategy,
            ContactGeometryStrategy,
            FrictionStrategy,
            PenaltyStrategy,
            TimeIntegrationStrategy,
        )
    except ImportError:
        errors.append("C6: strategies.protocols のインポートに失敗")
        return errors

    strategy_protocols: dict[str, type] = {
        "ContactForceStrategy": ContactForceStrategy,
        "FrictionStrategy": FrictionStrategy,
        "TimeIntegrationStrategy": TimeIntegrationStrategy,
        "ContactGeometryStrategy": ContactGeometryStrategy,
        "PenaltyStrategy": PenaltyStrategy,
    }

    for protocol_name, protocol_cls in strategy_protocols.items():
        # Protocol を実装する具象クラスを isinstance で検出
        matching = []
        for name, cls in registry.items():
            try:
                instance = cls.__new__(cls)
                if isinstance(instance, protocol_cls):
                    matching.append(name)
            except Exception:
                continue

        if not matching:
            errors.append(f"C6: {protocol_name} の具象クラスがレジストリに存在しない")
            continue

        for name in matching:
            cls = registry[name]
            if cls._test_class is None:
                errors.append(f"C6: {name} ({protocol_name} 実装) に意味論テストがない")

    return errors


def check_c9_frozen_immutability(registry: dict[str, type]) -> list[str]:
    """C9: execute() に入力データ不変性チェックが実装されているか検証.

    frozen dataclass の numpy 配列は in-place 変更可能なため、
    execute() 入口でチェックサムを記録し出口で検証する仕組みが必要。
    """
    errors = []

    # AbstractProcess.execute() のソースを検査
    try:
        source = inspect.getsource(AbstractProcess.execute)
        if "checksum" not in source and "hash" not in source:
            errors.append(
                "C9: AbstractProcess.execute() に入力データ不変性チェックが未実装"
                "（checksum/hash ベースの検証が必要）"
            )
    except (OSError, TypeError):
        errors.append("C9: AbstractProcess.execute() のソースを取得できない")

    return errors


def check_c11_transitive_deps(registry: dict[str, type]) -> list[str]:
    """C11: uses チェーンの推移的依存漏れを検出.

    process() 内で self.xxx.process() のようなチェーン呼び出しを
    AST 解析で検出し、推移的依存が uses で宣言されているか検証。
    """
    errors = []

    for name, cls in sorted(registry.items()):
        # テスト用フィクスチャは検査対象外
        if _is_test_fixture(cls):
            continue
        method = getattr(cls, "process", None)
        if method is None:
            continue
        if hasattr(method, "__wrapped__"):
            method = method.__wrapped__

        try:
            source = inspect.getsource(method)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue

        # self.xxx.process() のパターンを検出
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # self.xxx.process() パターン
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "process"
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
            ):
                attr_name = func.value.attr
                # self._xxx_instance.process() のような呼び出しは
                # uses で宣言された依存のインスタンスであるべき
                # ここでは検出のみ（情報提供レベル）
                declared_names = {dep.__name__.lower() for dep in cls.uses}
                if not any(d_name in attr_name.lower() for d_name in declared_names):
                    errors.append(
                        f"C11: {name}.process() 内の self.{attr_name}.process() は"
                        f" uses で宣言された依存と一致しない可能性"
                    )

    return errors


def check_c12_batch_order(registry: dict[str, type]) -> list[str]:
    """C12: BatchProcess 具象クラスの順序依存検証.

    BatchProcess を継承する具象クラスが存在するか、
    存在する場合は uses 宣言順と process() 内の実行順が一致するか検証。
    """
    errors = []

    # BatchProcess の具象クラスを検出
    batch_classes = []
    for name, cls in sorted(registry.items()):
        # カテゴリ判定: BatchProcess を継承しているか
        for base in cls.__mro__:
            if base.__name__ == "BatchProcess" and base is not cls:
                batch_classes.append((name, cls))
                break

    if not batch_classes:
        errors.append(
            "C12: BatchProcess の具象クラスが0件"
            "（process-architecture.md §6 で StrandBendingBatchProcess が必要）"
        )

    # 汎用ラッパー（実行対象を入力で受け取る BatchProcess）は
    # 静的 uses 宣言が不可能なため除外
    _GENERIC_WRAPPERS = {"BenchmarkRunnerProcess"}

    for name, cls in batch_classes:
        if not cls.uses and name not in _GENERIC_WRAPPERS:
            errors.append(f"C12: {name} の uses が空（順序依存検証不可）")

    return errors


def check_c13_compatibility_uses(registry: dict[str, type]) -> list[str]:
    """C13: active プロセスが CompatibilityProcess を uses している場合はエラー.

    CompatibilityProcess は deprecated プロセスの隔離カテゴリ。
    新規コード（active プロセス）からの uses 宣言を禁止する。
    """
    errors = []

    try:
        from xkep_cae.core.categories import CompatibilityProcess
    except ImportError:
        return errors  # CompatibilityProcess 未定義時はスキップ

    for name, cls in sorted(registry.items()):
        # deprecated プロセス自身はチェック対象外
        if hasattr(cls, "meta") and cls.meta.deprecated:
            continue
        # CompatibilityProcess 自身のサブクラスもチェック対象外
        if issubclass(cls, CompatibilityProcess):
            continue
        # テスト用フィクスチャは対象外
        if _is_test_fixture(cls):
            continue

        for dep in cls.uses:
            if issubclass(dep, CompatibilityProcess):
                errors.append(
                    f"C13: {name} が CompatibilityProcess である {dep.__name__} を uses に宣言"
                )

    return errors


def check_c14_deprecated_imports() -> list[str]:
    """C14: xkep_cae/ 内から __xkep_cae_deprecated をインポートしていないか検出.

    新パッケージが旧パッケージに依存していると脱出ポット計画が破綻する。
    AST 解析で以下を走査:
    - import __xkep_cae_deprecated...
    - from __xkep_cae_deprecated... import ...
    - importlib.import_module("__xkep_cae_deprecated...")
    """
    errors = []
    new_pkg = _project_root / "xkep_cae"

    for py_file in sorted(new_pkg.rglob("*.py")):
        if py_file.parent.name == "__pycache__":
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        rel = py_file.relative_to(_project_root)
        importlib_aliases = _collect_importlib_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("__xkep_cae_deprecated"):
                        errors.append(
                            f"C14: {rel}:{node.lineno} が __xkep_cae_deprecated をインポート"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("__xkep_cae_deprecated"):
                    errors.append(
                        f"C14: {rel}:{node.lineno} が __xkep_cae_deprecated からインポート"
                    )
            # importlib.import_module("__xkep_cae_deprecated...") の検出
            # エイリアス（import importlib as _il → _il.import_module(...)）も検出
            elif isinstance(node, ast.Call):
                if _is_importlib_deprecated_call(node, importlib_aliases):
                    errors.append(
                        f"C14: {rel}:{node.lineno} が importlib 経由で"
                        f" __xkep_cae_deprecated をインポート"
                    )

    return errors


def _collect_importlib_aliases(tree: ast.Module) -> set[str]:
    """AST から importlib のエイリアス名を収集.

    以下のパターンを検出:
    - import importlib           → {"importlib"}
    - import importlib as _il    → {"_il"}
    - from importlib import import_module  → (直接呼び出しは別途検出)
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    aliases.add(alias.asname or alias.name)
    return aliases


def _is_importlib_deprecated_call(
    node: ast.Call,
    importlib_aliases: set[str] | None = None,
) -> bool:
    """importlib.import_module("__xkep_cae_deprecated...") パターンを検出.

    importlib_aliases が指定された場合、エイリアス名も検出対象に含める。
    例: import importlib as _il → _il.import_module("__xkep_cae_deprecated...")
    """
    func = node.func
    aliases = importlib_aliases or {"importlib"}

    # XXX.import_module(...) パターン（XXX = importlib またはそのエイリアス）
    is_importlib = (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id in aliases
    )
    # _import_deprecated("__xkep_cae_deprecated...") 等のヘルパー関数パターン
    # → 引数の文字列リテラルを検査
    if not is_importlib:
        # ヘルパー関数呼び出しも引数に deprecated 文字列があれば検出
        if isinstance(func, ast.Name) and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value.startswith("__xkep_cae_deprecated")
        return False

    # importlib.import_module の第1引数が "__xkep_cae_deprecated..." であるか
    if node.args:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value.startswith("__xkep_cae_deprecated")
    return False


def check_c15_strategy_docs(registry: dict[str, type]) -> list[str]:
    """C15: ProcessMeta.document_path で指定されたドキュメントが実在するか検証.

    Strategy 具象クラスは ProcessMeta に document_path を宣言する必要があり、
    そのファイルが実際に存在しなければ契約違反。
    """
    errors = []

    for name, cls in sorted(registry.items()):
        meta = getattr(cls, "meta", None)
        if meta is None:
            continue
        doc_path = getattr(meta, "document_path", None)
        if not doc_path:
            continue

        # document_path はクラス定義ファイルからの相対パスで解決
        try:
            src_file = Path(inspect.getfile(cls))
            doc_full = (src_file.parent / doc_path).resolve()
            if not doc_full.exists():
                errors.append(
                    f"C15: {name} の document_path '{doc_path}' が存在しない"
                    f" (期待: {doc_full.relative_to(_project_root)})"
                )
        except (TypeError, OSError):
            errors.append(f"C15: {name} のソースファイルを取得できない")

    return errors


def check_c15_equation_refs(registry: dict[str, type]) -> list[str]:
    """C15 拡張（status-349）: MathematicalContract.equation_ref の台帳解決.

    `AbstractProcess.contracts` に宣言された `MathematicalContract` および
    `ProcessContractRegistry.all_contracts()` に登録された契約の
    `equation_ref` を `docs/math/` 台帳に対して解決する。

    検査対象:
        - 実行時に収集された `MathematicalContract` インスタンスのみ
          （contracts.py docstring 内の例題は ClassVar 未宣言のため収集されず
          自然に除外される）

    検出する不整合:
        - `bad_format`: ``<file>.md#<prefix>-<id>`` 形式に合致しない参照
        - `missing_file`: 台帳に存在しないファイルへの参照
        - `missing_anchor`: ファイルは存在するが未定義アンカーへの参照
        - `duplicate_anchor`: 同一ファイル内でアンカー多重定義
    """
    errors: list[str] = []

    # 循環 import 回避のため遅延 import
    try:
        from xkep_cae.mathematics import (
            ProcessContractRegistry,
            load_equation_index,
        )
    except ImportError as exc:
        errors.append(f"C15(math): mathematics パッケージのインポートに失敗: {exc}")
        return errors

    idx = load_equation_index()
    if idx.total_anchors == 0:
        errors.append("C15(math): docs/math/ 台帳が空（実行ディレクトリから見つからない可能性）")
        return errors

    # 重複アンカー検出（load() 側の集計を 1 つずつ契約違反として計上）
    for dup in idx.duplicates:
        errors.append(f"C15(math): 台帳アンカー重複 {dup.file_name}#{dup.anchor} ({dup.count} 回)")

    # 契約の収集: ProcessContractRegistry + 既存 ProcessRegistry の両経路
    seen_refs: set[tuple[str, str]] = set()  # (process_name, contract_name) 重複防止

    try:
        math_reg = ProcessContractRegistry.default()
        for proc_name, contracts in math_reg.all_contracts().items():
            for contract in contracts:
                key = (proc_name, contract.name)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                err = idx.resolve(contract.equation_ref)
                if err is not None:
                    errors.append(
                        f"C15(math): {proc_name}.contracts[{contract.name!r}]"
                        f".equation_ref {err.reason}: {contract.equation_ref}"
                    )
    except Exception as exc:
        errors.append(f"C15(math): ProcessContractRegistry 参照に失敗: {exc}")

    # ProcessRegistry 側からも直接 contracts ClassVar を参照（__init_subclass__ 経由）
    for proc_name, cls in sorted(registry.items()):
        contracts = getattr(cls, "contracts", ())
        if not contracts:
            continue
        for contract in contracts:
            # MathematicalContract 以外のゴミが入っていたら無視（Phase D 配線余地）
            equation_ref = getattr(contract, "equation_ref", None)
            contract_name = getattr(contract, "name", "?")
            if not equation_ref:
                continue
            key = (proc_name, contract_name)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            err = idx.resolve(equation_ref)
            if err is not None:
                errors.append(
                    f"C15(math): {proc_name}.contracts[{contract_name!r}]"
                    f".equation_ref {err.reason}: {equation_ref}"
                )

    return errors


def check_c18_verified_by_binding(registry: dict[str, type]) -> list[str]:
    """C18: `@verified_by` 紐付け検査（status-357 Phase E）.

    MCDD 脱法 pattern 2（dummy VerifyProcess を `@verified_by` に紐付けて
    C18 を通す）対策。各 Process の `contracts` ClassVar に宣言された
    `MathematicalContract` のうち severity ∈ {"hard", "nightly"} のものには
    `ProcessContractRegistry` で検証 Process が紐付いている必要がある。

    `severity="soft"` の契約は人間可読ドキュメント扱いのため紐付け不要。
    """
    errors: list[str] = []
    try:
        from xkep_cae.mathematics import ProcessContractRegistry
    except ImportError as exc:
        errors.append(f"C18: mathematics パッケージのインポートに失敗: {exc}")
        return errors

    reg = ProcessContractRegistry.default()
    bindings = reg.all_bindings()

    for proc_name, cls in sorted(registry.items()):
        contracts = getattr(cls, "contracts", ())
        if not contracts:
            continue
        for contract in contracts:
            contract_name = getattr(contract, "name", "?")
            severity = getattr(contract, "severity", "hard")
            if severity == "soft":
                continue
            if (proc_name, contract_name) not in bindings:
                errors.append(
                    f"C18: {proc_name}.contracts[{contract_name!r}] "
                    f"(severity={severity!r}) に @verified_by 紐付けが無い"
                )
    return errors


def check_c19_term_providers_exist(registry: dict[str, type]) -> list[str]:
    """C19: `TermExpansionContract.providers` 実在検査（status-357 Phase E）.

    MCDD 脱法 pattern 4（KcNormalDirectionStiffnessProcess を rename で済ませる）
    対策。`TermExpansionContract.providers` に列挙された Process クラス名が
    `ProcessRegistry` に実在するかを検査する。`providers` は文字列保持
    （循環 import 回避）なので実在確認は静的検査で担保する必要がある。
    """
    errors: list[str] = []
    try:
        from xkep_cae.mathematics.contracts import TermExpansionContract
    except ImportError as exc:
        errors.append(f"C19: mathematics.contracts のインポートに失敗: {exc}")
        return errors

    registry_names = set(registry.keys())
    # contract-level 重複チェックを避けるため (contract_name, provider) で seen
    seen: set[tuple[str, str]] = set()

    for proc_name, cls in sorted(registry.items()):
        contracts = getattr(cls, "contracts", ())
        if not contracts:
            continue
        for contract in contracts:
            if not isinstance(contract, TermExpansionContract):
                continue
            contract_name = contract.name
            for provider in contract.providers:
                if (contract_name, provider) in seen:
                    continue
                seen.add((contract_name, provider))
                if provider not in registry_names:
                    errors.append(
                        f"C19: TermExpansionContract[{contract_name!r}].providers "
                        f"に宣言された {provider!r} が ProcessRegistry に存在しない"
                        f"（宣言元 Process: {proc_name}）"
                    )
    return errors


def check_c20_term_expansion_bidirectional(registry: dict[str, type]) -> list[str]:
    """C20: `TermExpansionContract` 双方向紐付け整合性検査（status-358 Phase E）.

    MCDD 脱法 pattern 4 の rename 亜種対策。`TermExpansionContract.providers`
    に列挙された Process クラスが、自身の `contracts` ClassVar で**同一の**
    `TermExpansionContract`（`name` が一致するインスタンス）を宣言していることを
    検証する。

    C19 は providers の「registry 実在」のみを検査し、双方向整合性
    （provider → contract の逆方向宣言）は検査していない。これを放置すると:

    - 既存 Process に対して「providers に追加」だけ行い、Process 側の
      `contracts` 宣言を更新し忘れる。orchestrator `tangent_components()`
      からは呼ばれないまま契約には列挙されている状態。
    - rename（pattern 4）で新クラスを作成したが `contracts` 宣言を継承
      し忘れた場合（C19 は新クラス名が registry にあれば通る）。

    いずれも実行経路と契約宣言の乖離を招くため、型レベルで排除する。
    """
    errors: list[str] = []
    try:
        from xkep_cae.mathematics.contracts import TermExpansionContract
    except ImportError as exc:
        errors.append(f"C20: mathematics.contracts のインポートに失敗: {exc}")
        return errors

    # (contract_name, provider_name) 組を 1 度だけ検査
    seen: set[tuple[str, str]] = set()

    for proc_name, cls in sorted(registry.items()):
        contracts = getattr(cls, "contracts", ())
        if not contracts:
            continue
        for contract in contracts:
            if not isinstance(contract, TermExpansionContract):
                continue
            contract_name = contract.name
            for provider in contract.providers:
                key = (contract_name, provider)
                if key in seen:
                    continue
                seen.add(key)
                provider_cls = registry.get(provider)
                if provider_cls is None:
                    # C19 側で検出される（C20 では provider 実在は前提）
                    continue
                provider_contracts = getattr(provider_cls, "contracts", ())
                provider_contract_names = {getattr(c, "name", None) for c in provider_contracts}
                if contract_name not in provider_contract_names:
                    errors.append(
                        f"C20: TermExpansionContract[{contract_name!r}].providers "
                        f"に列挙された {provider!r} が自身の contracts で"
                        f" 同名契約を宣言していない"
                        f"（宣言元 Process: {proc_name}、双方向紐付け不整合）"
                    )
    return errors


def check_c21_term_expansion_no_duplicates(registry: dict[str, type]) -> list[str]:
    """C21: `TermExpansionContract.term_names` / `providers` 重複検出（status-360 Phase E）.

    `TermExpansionContract.__post_init__` が生成時に重複を弾くため、
    通常は静的検査でもヒットしない想定だが、以下の脱法すり抜けシナリオに備える:

    - `object.__setattr__` 等で __post_init__ を跳ね飛ばして frozen dataclass を
      再構成する。
    - 異なる Process で同一契約インスタンスを共有した際に重複項が生まれる
      ような副作用的 mutation（frozen 回避経路）。

    本検査はレジストリ走査により term_names・providers のセット化で重複を
    再検出することで二重の防御線を敷く。
    """
    errors: list[str] = []
    try:
        from xkep_cae.mathematics.contracts import TermExpansionContract
    except ImportError as exc:
        errors.append(f"C21: mathematics.contracts のインポートに失敗: {exc}")
        return errors

    seen_contracts: set[tuple[str, int]] = set()

    for proc_name, cls in sorted(registry.items()):
        contracts = getattr(cls, "contracts", ())
        if not contracts:
            continue
        for contract in contracts:
            if not isinstance(contract, TermExpansionContract):
                continue
            key = (proc_name, id(contract))
            if key in seen_contracts:
                continue
            seen_contracts.add(key)
            if len(set(contract.term_names)) != len(contract.term_names):
                errors.append(
                    f"C21: TermExpansionContract[{contract.name!r}] の"
                    f" term_names に重複: {contract.term_names}"
                    f"（宣言元 Process: {proc_name}）"
                )
            if len(set(contract.providers)) != len(contract.providers):
                errors.append(
                    f"C21: TermExpansionContract[{contract.name!r}] の"
                    f" providers に重複: {contract.providers}"
                    f"（宣言元 Process: {proc_name}）"
                )
    return errors


def check_c22_contracts_no_duplicate_names(registry: dict[str, type]) -> list[str]:
    """C22: `contracts` ClassVar 同名契約重複検出（status-360 Phase E）.

    `ProcessContractRegistry.register_contracts` は動的に同名契約の重複を
    弾いているが、静的検査として登録前（AbstractProcess.__init_subclass__
    による自動登録に依存しない経路）で先行検出する。

    単一 Process の `contracts` ClassVar に同一 `name` の
    `MathematicalContract` が複数存在するケースを検出する。
    """
    errors: list[str] = []

    for proc_name, cls in sorted(registry.items()):
        contracts = getattr(cls, "contracts", ())
        if not contracts:
            continue
        seen_names: dict[str, int] = {}
        for contract in contracts:
            name = getattr(contract, "name", None)
            if not isinstance(name, str) or not name:
                continue
            seen_names[name] = seen_names.get(name, 0) + 1
        duplicated = {n: cnt for n, cnt in seen_names.items() if cnt >= 2}
        for dup_name, cnt in sorted(duplicated.items()):
            errors.append(
                f"C22: {proc_name}.contracts に同名契約 {dup_name!r} が"
                f" {cnt} 回登場（同名契約の重複宣言は禁止）"
            )
    return errors


def check_c23_verifier_category(registry: dict[str, type]) -> list[str]:
    """C23: `@verified_by` 検証 Process カテゴリ検査（status-360 Phase E）.

    `ProcessContractRegistry.bind_verifier` は登録時に `SolverProcess` /
    `VerifyProcess` カテゴリ継承を要求するが、本検査は登録後の紐付け
    スナップショットを走査して静的に再確認する。レジストリが何らかの形で
    外部 mutate された場合の二重防御線。
    """
    errors: list[str] = []
    try:
        from xkep_cae.core.categories import SolverProcess, VerifyProcess
        from xkep_cae.mathematics import ProcessContractRegistry
    except ImportError as exc:
        errors.append(f"C23: 依存モジュールのインポートに失敗: {exc}")
        return errors

    reg = ProcessContractRegistry.default()
    bindings = reg.all_bindings()

    for (proc_name, contract_name), verify_cls in sorted(bindings.items()):
        if not isinstance(verify_cls, type):
            errors.append(
                f"C23: {proc_name}.contracts[{contract_name!r}] の紐付け "
                f"{verify_cls!r} が type ではない"
            )
            continue
        if not issubclass(verify_cls, (SolverProcess, VerifyProcess)):
            bases = ", ".join(b.__name__ for b in verify_cls.__mro__[1:4])
            errors.append(
                f"C23: {proc_name}.contracts[{contract_name!r}] の紐付け "
                f"{verify_cls.__name__} が SolverProcess / VerifyProcess の"
                f" いずれも継承していない（mro 上位: {bases}）"
            )
    return errors


def _check_reexported_class(cls: type, cls_name: str, rel: Path) -> list[str]:
    """__init__.py から再エクスポートされたクラスの C16 準拠を検査.

    private モジュール（_xxx.py）で定義されたクラスが __init__.py 経由で
    公開される場合、そのクラスも C16 ルールに従う必要がある。
    """
    import dataclasses
    import enum

    errors: list[str] = []

    # 1. AbstractProcess サブクラス → OK
    if issubclass(cls, AbstractProcess):
        return errors

    # 2. frozen dataclass
    if dataclasses.is_dataclass(cls):
        params = getattr(cls, "__dataclass_params__", None)
        if params and params.frozen:
            # メソッド検査: frozen dataclass にメソッドがあれば違反
            # 許可: property（派生フィールド）, classmethod（ファクトリ）,
            #       通常メソッド（派生計算）
            # frozen dataclass のメソッドは自身のフィールドからの
            # 純粋計算であり、状態変更を伴わないため全て許可する
            return errors
        errors.append(
            f"C16: {rel} が re-export する {cls_name} は non-frozen dataclass（frozen=True が必須）"
        )
        return errors

    # 3. Enum → OK
    if issubclass(cls, enum.Enum):
        return errors

    # いずれにも該当しない → 違反
    errors.append(
        f"C16: {rel} が re-export する {cls_name} は"
        f" Process でも frozen dataclass でもない"
        f"（許可: AbstractProcess subclass / frozen dataclass / Enum）"
    )
    return errors


def check_c16_sterilization() -> list[str]:
    """C16: 新パッケージ滅菌チェック — Process Architecture 外の型を完全検挙.

    xkep_cae/ 配下の全モジュール（core/・mathematics/ を除く）で定義された
    全クラスを分類し、以下のいずれにも該当しないものを契約違反とする:

    許可カテゴリ:
      1. AbstractProcess サブクラス（Process）
      2. frozen dataclass（入出力データ型）
      3. Enum サブクラス（設定値列挙）

    トップレベル関数は Protocol/Strategy(AbstractProcess)/Process のいずれかであるべき。
    純粋関数（非クラス）は __all__ エクスポートの有無に関わらず契約違反。

    除外:
      - core/ — 基盤モジュール（Protocol 定義、レジストリ等）は検査対象外
      - mathematics/ — MCDD 基盤モジュール（status-347 Phase A-2）。
        `ProcessContractRegistry` がミュータブルなシングルトンレジストリであり、
        `core/registry.py::ProcessRegistry` と構造的に同等の「基盤レジストリ」で
        あるため core/ と同様に滅菌検査から除外する。
        契約型自体は全て frozen dataclass として C16 準拠を満たす。
      - core/mathematics からのインポート関数も検査対象外
    """
    import dataclasses
    import enum

    errors = []
    # core/・mathematics/ を除く xkep_cae/ 配下の全サブパッケージを走査
    # (mathematics/ は status-347 で追加された MCDD 基盤。理由は docstring 参照)
    xkep_root = _project_root / "xkep_cae"
    scan_roots = [
        d
        for d in sorted(xkep_root.iterdir())
        if d.is_dir() and d.name not in ("__pycache__", "core", "mathematics")
    ]

    # protocols.py は Protocol 定義ファイルなのでスキップ
    _SKIP_STEMS = {"protocols"}

    all_py_files = []
    for root in scan_roots:
        if root.exists():
            all_py_files.extend(sorted(root.rglob("*.py")))

    for py_file in all_py_files:
        # テストファイル・__pycache__ はスキップ
        if py_file.parent.name == "tests" or py_file.name.startswith("test_"):
            continue
        if "__pycache__" in str(py_file):
            continue
        if py_file.stem in _SKIP_STEMS:
            continue
        # _ prefix のプライベートモジュールはスキップ（内部実装）
        if py_file.stem.startswith("_") and py_file.stem != "__init__":
            continue

        # AST でクラス名・関数名を収集
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        # __init__.py は re-export を検査（関数 + クラスの公開エイリアスを検出）
        if py_file.stem == "__init__":
            rel = py_file.relative_to(_project_root)
            # __init__.py のモジュールをインポートしてクラスの型を検査
            init_mod_path = py_file.relative_to(_project_root).with_suffix("")
            init_mod_name = str(init_mod_path).replace("/", ".")
            try:
                init_mod = importlib.import_module(init_mod_name)
            except Exception:
                init_mod = None

            for node in ast.iter_child_nodes(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for alias in node.names:
                    # _foo as foo パターン: private 関数を公開名で re-export
                    if (
                        alias.asname
                        and not alias.asname.startswith("_")
                        and alias.name.startswith("_")
                        and not alias.name.startswith("__")
                    ):
                        errors.append(
                            f"C16: {rel} が {alias.name} を"
                            f" {alias.asname} として公開 re-export"
                            f"（private 関数の公開エイリアス禁止）"
                        )
                    exported_name = alias.asname or alias.name
                    # _prefix は内部用なのでスキップ
                    if exported_name.startswith("_"):
                        continue
                    # 大文字開始 = クラスの可能性 → 型を検査
                    if exported_name[0].isupper() and init_mod is not None:
                        cls = getattr(init_mod, exported_name, None)
                        if cls is not None and isinstance(cls, type):
                            errors.extend(_check_reexported_class(cls, exported_name, rel))
                        continue
                    # 小文字開始 = 関数の可能性
                    if exported_name[0].islower() and not alias.name.startswith("_"):
                        errors.append(
                            f"C16: {rel} が {exported_name}() を公開 re-export"
                            f"（純粋関数の公開エクスポート禁止）"
                        )
            continue

        class_names = []
        func_names = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                func_names.append(node.name)

        if not class_names and not func_names:
            continue

        # モジュールをインポートして実際の型を検査
        mod_path = py_file.relative_to(_project_root).with_suffix("")
        mod_name = str(mod_path).replace("/", ".")
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            errors.append(f"C16: {mod_name} のインポート失敗: {e}")
            continue

        rel = py_file.relative_to(_project_root)

        # ── クラス検査 ──
        for cls_name in class_names:
            # private クラスは許可（内部ヘルパー）
            if cls_name.startswith("_"):
                continue

            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue

            # 1. AbstractProcess サブクラス
            if isinstance(cls, type) and issubclass(cls, AbstractProcess):
                continue

            # 2. frozen dataclass
            if dataclasses.is_dataclass(cls):
                params = getattr(cls, "__dataclass_params__", None)
                if params and params.frozen:
                    # frozen dataclass のメソッドは自身のフィールドからの
                    # 純粋計算であり、状態変更を伴わないため全て許可する
                    continue
                errors.append(
                    f"C16: {rel} の {cls_name} は non-frozen dataclass（frozen=True が必須）"
                )
                continue

            # 3. Enum
            if isinstance(cls, type) and issubclass(cls, enum.Enum):
                continue

            # いずれにも該当しない → 違反
            errors.append(
                f"C16: {rel} の {cls_name} は Process でも frozen dataclass でもない"
                f"（許可: AbstractProcess subclass / frozen dataclass / Enum）"
            )

        # ── 関数検査 ──
        # 純粋関数は Protocol / Strategy(AbstractProcess) / Process に変換すべき
        for fn_name in func_names:
            # private 関数は許可（内部ヘルパー）
            if fn_name.startswith("_"):
                continue
            # public 純粋関数 → 違反（Protocol/Strategy/Process に変換が必要）
            errors.append(
                f"C16: {rel} の {fn_name}() は純粋関数"
                f"（Protocol / Strategy / Process に変換が必要）"
            )

    return errors


def check_c17_private_dataclass_hygiene() -> list[str]:
    """C17: プライベートモジュール内 dataclass 衛生チェック.

    _xxx.py（プライベートモジュール）内で定義された dataclass を走査し、
    以下を違反として検出する:

    1. frozen=True でない dataclass（mutable state の温床）
    2. クラス名が Input / Output で終わらない dataclass
       （データ型の役割が不明瞭）
    3. dataclasses.replace() の使用（frozen 化の代替にならない）

    C16 はプライベートモジュールをスキップするため、
    このチェックが内部実装の品質を補完する。
    """
    import dataclasses

    errors: list[str] = []
    xkep_root = _project_root / "xkep_cae"
    scan_roots = [
        d
        for d in sorted(xkep_root.iterdir())
        if d.is_dir() and d.name != "__pycache__" and d.name != "core"
    ]

    for root in scan_roots:
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            # テストファイルはスキップ
            if py_file.parent.name == "tests" or py_file.name.startswith("test_"):
                continue
            # プライベートモジュールのみ対象（__init__.py は C16 で検査済み）
            if not py_file.stem.startswith("_") or py_file.stem == "__init__":
                continue

            # AST でクラス名を収集
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            class_names = [
                node.name for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef)
            ]
            if not class_names:
                continue

            # モジュールをインポートして実際の型を検査
            mod_path = py_file.relative_to(_project_root).with_suffix("")
            mod_name = str(mod_path).replace("/", ".")
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue

            rel = py_file.relative_to(_project_root)

            for cls_name in class_names:
                cls = getattr(mod, cls_name, None)
                if cls is None:
                    continue
                if not dataclasses.is_dataclass(cls):
                    continue

                params = getattr(cls, "__dataclass_params__", None)
                if not (params and params.frozen):
                    errors.append(
                        f"C17: {rel} の {cls_name} は non-frozen dataclass（frozen=True が必須）"
                    )

                # クラス名が Input / Result で終わるか検査
                if not (cls_name.endswith("Input") or cls_name.endswith("Output")):
                    errors.append(
                        f"C17: {rel} の {cls_name} は"
                        f" Input/Output で終わらない dataclass 名"
                        f"（データ型の役割を明示する命名が必要）"
                    )

            # dataclasses.replace() の使用を検出
            _replace_patterns = [
                r"\bdataclasses\.replace\s*\(",
                r"\breplace\s*\(",
            ]
            # import 文から replace を取得しているか確認
            _imports_replace = False
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "dataclasses" and node.names:
                        for alias in node.names:
                            if alias.name == "replace":
                                _imports_replace = True
            # dataclasses.replace() 直接呼び出し
            import re

            for line_no, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "dataclasses.replace(" in line:
                    errors.append(
                        f"C17: {rel}:{line_no} で dataclasses.replace() を使用"
                        f"（frozen 化の代替にならない — 構造的な不変設計が必要）"
                    )
                elif _imports_replace and re.search(r"\breplace\s*\(", line):
                    # from dataclasses import replace を使っている場合
                    # ただし def replace や他の replace は除外
                    if "def replace" not in line and "str.replace" not in line:
                        errors.append(
                            f"C17: {rel}:{line_no} で dataclasses.replace() を使用"
                            f"（frozen 化の代替にならない — 構造的な不変設計が必要）"
                        )

    return errors


def check_o1_test_direct_function_calls() -> list[str]:
    """O1（条例）: テストが Process ラッパーのある関数を直接呼び出していないか検出.

    Process 化された関数をテストが直接 import して使用する場合、
    Process Architecture の一貫性テストが不足している可能性がある。
    法律違反（C14/C16）ではないが、条例違反として検知する。
    """
    errors = []
    xkep_root = _project_root / "xkep_cae"

    # Process ラッパーが存在する既知のプライベート関数マッピング
    _KNOWN_PROCESS_WRAPPERS: dict[str, str] = {
        "_deformed_coords": "DeformedCoordsProcess",
        "_ncp_line_search": "NCPLineSearchProcess",
        "_snapshot_contact_graph": "ContactGraphProcess",
        "_format_diagnostics_report": "DiagnosticsReportProcess",
        "_check_initial_penetration": "InitialPenetrationProcess",
        "_adjust_initial_positions": "InitialPenetrationProcess",
    }

    # テストファイルを走査
    for py_file in sorted(xkep_root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        is_test = py_file.name.startswith("test_") or py_file.parent.name == "tests"
        if not is_test:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        rel = py_file.relative_to(_project_root)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("xkep_cae"):
                    for alias in node.names:
                        if alias.name in _KNOWN_PROCESS_WRAPPERS:
                            process_name = _KNOWN_PROCESS_WRAPPERS[alias.name]
                            errors.append(
                                f"O1: {rel}:{node.lineno} が {alias.name}() を直接 import"
                                f"（{process_name} 経由を推奨）"
                            )

    return errors


def check_o2_backend_injection() -> list[str]:
    """O2（条例）: xkep_cae/ 内の BackendRegistry パターンを検出.

    Process Architecture では依存注入はProcess.uses で宣言するのが正規手段。
    BackendRegistry のようなモジュールレベル・シングルトン注入は
    Process Architecture を迂回しており、廃止対象。
    """
    errors = []
    xkep_root = _project_root / "xkep_cae"

    # core/ は基盤モジュールなので検査対象外
    # TypeVar, frozen dataclass インスタンス等は正当なモジュールレベル定数
    _SAFE_FACTORY_NAMES = {"TypeVar", "ParamSpec", "NewType"}

    for py_file in sorted(xkep_root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        if py_file.parent.name == "tests" or py_file.name.startswith("test_"):
            continue
        # core/ は基盤モジュール — Registry パターン検査対象外
        try:
            rel = py_file.relative_to(xkep_root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "core":
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        rel = py_file.relative_to(_project_root)
        for node in ast.iter_child_nodes(tree):
            # クラス定義: configure/reset パターンの検出（Registry 系）
            if isinstance(node, ast.ClassDef):
                method_names = {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}
                if "configure" in method_names and "reset" in method_names:
                    errors.append(
                        f"O2: {rel}:{node.lineno} の {node.name} は"
                        f" configure()/reset() を持つ Registry パターン"
                        f"（Process.uses による依存宣言に移行が必要）"
                    )
            # モジュールレベル変数: シングルトンインスタンス検出
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                        func = node.value.func
                        if isinstance(func, ast.Name) and func.id[0].isupper():
                            # TypeVar 等の型ユーティリティは除外
                            if func.id in _SAFE_FACTORY_NAMES:
                                continue
                            # ALL_CAPS = 定数パターンは除外
                            if target.id.isupper():
                                continue
                            errors.append(
                                f"O2: {rel}:{node.lineno} の"
                                f" {target.id} = {func.id}() は"
                                f" モジュールレベルシングルトン"
                                f"（Process Architecture 外の状態管理）"
                            )

    return errors


def check_o3_test_backend_configure() -> list[str]:
    """O3（条例）: テスト conftest 等での backend.configure() 呼び出しを検出.

    BackendRegistry への注入はProcess Architecture迂回であり廃止対象。
    """
    errors = []
    # tests/ と xkep_cae/ 内のテスト conftest を走査
    scan_paths = [_project_root / "tests", _project_root / "xkep_cae"]

    for scan_root in scan_paths:
        for py_file in sorted(scan_root.rglob("conftest.py")):
            if "__pycache__" in str(py_file):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            rel = py_file.relative_to(_project_root)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # backend.configure(...) / backend.configure_frequency(...) 等
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr.startswith("configure")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "backend"
                ):
                    errors.append(
                        f"O3: {rel}:{node.lineno} の"
                        f" backend.{func.attr}() 呼び出し"
                        f"（BackendRegistry 注入は廃止対象。"
                        f"Process API に移行が必要）"
                    )

    return errors


def main() -> int:
    """全チェックを実行し、結果を表示."""
    print("=" * 60)
    print("プロセス契約違反検出スクリプト（C3-C23 + O1-O3）")
    print("=" * 60)

    print("\nモジュールインポート中...")
    _import_all_modules()

    registry = ProcessRegistry.default()
    print(f"レジストリ登録プロセス数: {len(registry)}")
    for name in sorted(registry.keys()):
        cls = registry[name]
        test = cls._test_class or "(未紐付け)"
        print(f"  {name}: test={test}")

    all_errors: list[str] = []
    checks = [
        ("C3: テスト紐付け", check_c3_test_binding),
        ("C5: 未宣言依存（AST）", check_c5_undeclared_deps),
        ("C6: Strategy意味論", check_c6_strategy_semantics),
        ("C7: メタクラスラップ", check_c7_metaclass_wrap),
        ("C8: StrategySlot 依存カバー", check_c8_strategy_slot),
        ("C9: frozen不変性", check_c9_frozen_immutability),
        ("C11: 推移的依存", check_c11_transitive_deps),
        ("C12: BatchProcess順序", check_c12_batch_order),
        ("C13: CompatibilityProcess uses禁止", check_c13_compatibility_uses),
        ("C15: Strategy ドキュメント存在", check_c15_strategy_docs),
        ("C15(math): MathematicalContract.equation_ref 解決", check_c15_equation_refs),
        ("C18: @verified_by 紐付け検査", check_c18_verified_by_binding),
        ("C19: TermExpansionContract.providers 実在", check_c19_term_providers_exist),
        ("C20: TermExpansionContract 双方向紐付け", check_c20_term_expansion_bidirectional),
        ("C21: TermExpansionContract 項名/providers 重複", check_c21_term_expansion_no_duplicates),
        ("C22: contracts ClassVar 同名契約重複", check_c22_contracts_no_duplicate_names),
        ("C23: @verified_by 検証器カテゴリ", check_c23_verifier_category),
    ]

    for label, check_fn in checks:
        print(f"\n--- {label} ---")
        errors = check_fn(registry)
        all_errors.extend(errors)
        if errors:
            for e in errors:
                print(f"  NG: {e}")
        else:
            print("  OK")

    # C14 はレジストリ不要（ファイルシステム走査のみ）
    print("\n--- C14: deprecated インポート禁止 ---")
    c14_errors = check_c14_deprecated_imports()
    all_errors.extend(c14_errors)
    if c14_errors:
        for e in c14_errors:
            print(f"  NG: {e}")
    else:
        print("  OK")

    # C16 もレジストリ不要（ファイルシステム走査 + ランタイム型検査）
    print("\n--- C16: 新パッケージ滅菌 ---")
    c16_errors = check_c16_sterilization()
    all_errors.extend(c16_errors)
    if c16_errors:
        for e in c16_errors:
            print(f"  NG: {e}")
    else:
        print("  OK")

    # C17: プライベートモジュール dataclass 衛生
    print("\n--- C17: プライベートモジュール dataclass 衛生 ---")
    c17_errors = check_c17_private_dataclass_hygiene()
    all_errors.extend(c17_errors)
    if c17_errors:
        for e in c17_errors:
            print(f"  NG: {e}")
    else:
        print("  OK")

    # O1-O3: 条例違反チェック
    all_ordinance: list[str] = []

    print("\n--- O1: テスト直接関数呼び出し（条例） ---")
    o1_warnings = check_o1_test_direct_function_calls()
    all_ordinance.extend(o1_warnings)
    if o1_warnings:
        for w in o1_warnings:
            print(f"  WARN: {w}")
    else:
        print("  OK")

    print("\n--- O2: BackendRegistry パターン（条例） ---")
    o2_warnings = check_o2_backend_injection()
    all_ordinance.extend(o2_warnings)
    if o2_warnings:
        for w in o2_warnings:
            print(f"  WARN: {w}")
    else:
        print("  OK")

    print("\n--- O3: テスト backend.configure() 注入（条例） ---")
    o3_warnings = check_o3_test_backend_configure()
    all_ordinance.extend(o3_warnings)
    if o3_warnings:
        for w in o3_warnings:
            print(f"  WARN: {w}")
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"契約違反: {len(all_errors)} 件")
    if all_ordinance:
        print(f"条例違反: {len(all_ordinance)} 件（警告）")
    if all_errors:
        print("\n修正ガイド:")
        print("  C3  → concrete/test_*.py を作成し @binds_to で紐付け")
        print("  C6  → test_contracts.py の意味論テストを実装で解消")
        print("  C9  → base.py execute() にチェックサム検証を追加")
        print("  C12 → batch/ に BatchProcess 具象クラスを実装")
        print("  C13 → active プロセスから CompatibilityProcess への uses を削除")
        print("  C14 → xkep_cae/ 内の deprecated インポートを除去（importlib 経由も禁止）")
        print("  C15 → ProcessMeta.document_path が指すドキュメントを作成")
        print("  C16 → クラスは AbstractProcess/frozen dataclass/Enum のみ許可。")
        print("       純粋関数は Protocol/Strategy/Process に変換するか _ prefix で private 化")
        print("       __init__.py の re-export クラスも検査対象")
        print("  C18 → @verified_by(contract_name, VerifyProcess) で契約に検証 Process を紐付け")
        print("       dummy/trivial VerifyProcess は禁止（_is_trivial_stmt で弾かれる）")
        print("  C19 → TermExpansionContract.providers の Process を ProcessRegistry に実装")
        print("       文字列保持のため rename は静的検査で検出される（脱法 pattern 4 防止）")
        print("  C20 → providers に列挙した Process は自身の contracts で同名契約を宣言")
        print("       双方向紐付けで providers 追加のみ / 宣言漏れ rename を検出")
        print("  C21 → TermExpansionContract の term_names / providers の重複を除去")
        print("       __post_init__ が通る異常経路（frozen 回避等）の構造的検出")
        print("  C22 → 同一 Process の contracts ClassVar に同名契約を複数並べない")
        print("       各契約に固有の name を付与すること")
        print("  C23 → @verified_by の検証器は SolverProcess / VerifyProcess 継承必須")
        print("       PreProcess / PostProcess / BatchProcess は検証器不可")
        print("  C17 → プライベートモジュール内 dataclass は frozen=True 必須。")
        print("       クラス名は Input/Output で終わる命名が必要")
        print("  O1  → テストで Process ラッパーのある関数を直接呼ばず Process API を使用")
        print("  O2  → BackendRegistry パターンを Process.uses に移行")
        print("  O3  → conftest の backend.configure() を廃止し Process API に移行")
        return 1
    elif all_ordinance:
        print(f"\n契約違反なし（条例違反 {len(all_ordinance)} 件は警告のみ）")
        return 0
    else:
        print("契約違反なし、条例違反なし")
        return 0


if __name__ == "__main__":
    sys.exit(main())
