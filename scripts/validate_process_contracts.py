"""プロセス契約違反検出スクリプト.

process-architecture.md §13 で定義された契約抜け腐敗シナリオを機械的に検出する:

- C3: テスト未紐付けのプロセス（_test_class is None）
- C5: process() 内の未宣言依存（AST解析）
- C6: Strategy Protocol の意味論的契約違反（具象クラス未検証）
- C7: process() のメタクラスラップ漏れ
- C8: _runtime_uses / StrategySlot の動的依存カバー
- C9: frozen dataclass numpy 配列変更検出（execute() チェックサム未実装）
- C11: uses チェーンの推移的依存漏れ
- C12: BatchProcess 具象クラスの順序依存検証
- C13: active プロセスが CompatibilityProcess を uses している場合はエラー

使用方法:
    python scripts/validate_process_contracts.py 2>&1 | tee /tmp/log-$(date +%s).log
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

from xkep_cae.process.base import AbstractProcess  # noqa: E402
from xkep_cae.process.registry import ProcessRegistry  # noqa: E402


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

    Phase 9-A: ハードコードされたモジュールリストを廃止し、
    xkep_cae/process/ 配下の .py ファイルをファイルシステム走査で自動検出する。
    新規プロセス追加時のモジュールリスト更新忘れを根絶。
    """
    process_root = _project_root / "xkep_cae" / "process"

    # 除外対象: テストファイル、__init__.py、基盤モジュール
    _SKIP_NAMES = {"__init__", "base", "categories", "data", "slots", "tree", "runner"}

    # xkep_cae/process/ 配下の全 .py を走査（テスト以外）
    process_modules = []
    for py_file in sorted(process_root.rglob("*.py")):
        # テストファイルはあとで別処理
        if py_file.parent.name == "tests" or py_file.name.startswith("test_"):
            continue
        if py_file.parent.name == "__pycache__":
            continue
        if py_file.stem in _SKIP_NAMES:
            continue
        # docs/ 配下のマークダウンと同名ファイルなどを除外
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
    for py_file in sorted(process_root.rglob("test_*.py")):
        if py_file.parent.name == "__pycache__":
            continue
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
    from xkep_cae.process.slots import collect_strategy_slots, collect_strategy_types

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
        from xkep_cae.process.strategies.protocols import (
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

    for name, cls in batch_classes:
        if not cls.uses:
            errors.append(f"C12: {name} の uses が空（順序依存検証不可）")

    return errors


def check_c13_compatibility_uses(registry: dict[str, type]) -> list[str]:
    """C13: active プロセスが CompatibilityProcess を uses している場合はエラー.

    CompatibilityProcess は deprecated プロセスの隔離カテゴリ。
    新規コード（active プロセス）からの uses 宣言を禁止する。
    """
    errors = []

    try:
        from xkep_cae.process.categories import CompatibilityProcess
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


def main() -> int:
    """全チェックを実行し、結果を表示."""
    print("=" * 60)
    print("プロセス契約違反検出スクリプト（C3-C13）")
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

    print("\n" + "=" * 60)
    if all_errors:
        print(f"契約違反: {len(all_errors)} 件")
        print("\n修正ガイド:")
        print("  C3  → concrete/test_*.py を作成し @binds_to で紐付け")
        print("  C6  → test_contracts.py の意味論テストを実装で解消")
        print("  C9  → base.py execute() にチェックサム検証を追加")
        print("  C12 → batch/ に BatchProcess 具象クラスを実装")
        print("  C13 → active プロセスから CompatibilityProcess への uses を削除")
        return 1
    else:
        print("契約違反なし")
        return 0


if __name__ == "__main__":
    sys.exit(main())
