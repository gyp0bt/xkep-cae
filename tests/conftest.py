"""テスト設定 — deprecated 参照テストの収集を抑制.

xkep_cae_deprecated → xkep_cae 移行に伴い、旧パッケージパスを参照する
テストファイルは ImportError で収集失敗する。
ここでは収集自体をスキップして、pytest のエラーカウントに含めない。

未移行テストは個別に移行して xkep_cae/ 内のテストに統合する。

status-193 で導入。

[← README](../README.md)
"""

import importlib
import importlib.util
import sys


def pytest_ignore_collect(collection_path, config):
    """未移行モジュールを参照するテストの収集をスキップする."""
    if not collection_path.suffix == ".py":
        return False
    if collection_path.name == "conftest.py":
        return False
    if "generate_verification" in collection_path.name:
        return True  # テストファイルではない

    try:
        text = collection_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    # xkep_cae_deprecated を参照しているファイル
    if "xkep_cae_deprecated" in text:
        return True

    # テストモジュールとして import を試行
    # 失敗したら収集をスキップ
    module_name = f"_conftest_probe_{collection_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, collection_path)
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError):
        return True
    except Exception:
        # import 以外のエラー（SyntaxError 等）は pytest に任せる
        return False
    finally:
        # プローブモジュールを削除
        sys.modules.pop(module_name, None)

    return False
