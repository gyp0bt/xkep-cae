"""プロセス契約違反検出スクリプト.

process-architecture.md §13 で定義された契約抜け腐敗シナリオのうち、
以下を機械的に検出する:

- C3: テスト未紐付けのプロセス（_test_class is None）
- C5: process() 内の未宣言依存（AST解析）
- C7: process() のメタクラスラップ漏れ
- C8: _runtime_uses の静的 uses 未カバー

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


def _import_all_modules() -> None:
    """全プロセスモジュール + テストモジュールをインポートしてレジストリを構築."""
    # concrete プロセス
    concrete_modules = [
        "xkep_cae.process.concrete.pre_mesh",
        "xkep_cae.process.concrete.pre_contact",
        "xkep_cae.process.concrete.solve_ncp",
        "xkep_cae.process.concrete.post_export",
        "xkep_cae.process.concrete.post_render",
    ]
    # strategy プロセス
    strategy_modules = [
        "xkep_cae.process.strategies.penalty",
        "xkep_cae.process.strategies.friction",
        "xkep_cae.process.strategies.contact_force",
        "xkep_cae.process.strategies.contact_geometry",
        "xkep_cae.process.strategies.time_integration",
    ]

    for mod_name in concrete_modules + strategy_modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            print(f"  警告: {mod_name} のインポートに失敗: {e}")

    # テストモジュール（@binds_to 発動用）を動的インポート
    test_dirs = [
        _project_root / "xkep_cae" / "process" / "tests",
        _project_root / "xkep_cae" / "process" / "strategies" / "tests",
    ]
    for test_dir in test_dirs:
        if not test_dir.is_dir():
            continue
        for py_file in sorted(test_dir.glob("test_*.py")):
            mod_path = py_file.relative_to(_project_root).with_suffix("")
            mod_name = str(mod_path).replace("/", ".")
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                print(f"  警告: {mod_name} のインポートに失敗: {e}")


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


def check_c8_runtime_uses(registry: dict[str, type]) -> list[str]:
    """C8: _runtime_uses が静的 uses でカバーされていないケースを検出."""
    errors = []
    for name, cls in sorted(registry.items()):
        # _runtime_uses を設定するクラスかチェック
        try:
            source = inspect.getsource(cls.__init__)
            if "_runtime_uses" not in source:
                continue
        except (OSError, TypeError):
            continue

        # デフォルト引数でインスタンス化を試みる
        try:
            instance = cls()
        except Exception as e:
            errors.append(f"C8: {name} のデフォルトインスタンス化に失敗: {e}")
            continue

        runtime = getattr(instance, "_runtime_uses", [])
        static_names = {dep.__name__ for dep in cls.uses}

        for dep in runtime:
            dep_name = dep.__name__ if hasattr(dep, "__name__") else str(dep)
            if dep_name not in static_names:
                errors.append(
                    f"C8: {name}._runtime_uses に {dep_name} があるが、静的 uses に含まれていない"
                )
    return errors


def main() -> int:
    """全チェックを実行し、結果を表示."""
    print("=" * 60)
    print("プロセス契約違反検出スクリプト")
    print("=" * 60)

    print("\nモジュールインポート中...")
    _import_all_modules()

    registry = AbstractProcess._registry
    print(f"レジストリ登録プロセス数: {len(registry)}")
    for name in sorted(registry.keys()):
        cls = registry[name]
        test = cls._test_class or "(未紐付け)"
        print(f"  {name}: test={test}")

    all_errors: list[str] = []
    checks = [
        ("C3: テスト紐付け", check_c3_test_binding),
        ("C5: 未宣言依存（AST）", check_c5_undeclared_deps),
        ("C7: メタクラスラップ", check_c7_metaclass_wrap),
        ("C8: 動的依存カバー", check_c8_runtime_uses),
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
        return 1
    else:
        print("契約違反なし")
        return 0


if __name__ == "__main__":
    sys.exit(main())
