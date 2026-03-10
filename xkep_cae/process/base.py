"""AbstractProcess 基底クラスとメタクラス.

全プロセスの基底クラス。メタクラスにより process() を自動ラップし、
実行トレース・プロファイリングを透過的に実現する。

"""

from __future__ import annotations

import functools
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class ProcessMeta:
    """プロセスのメタ情報."""

    name: str
    module: str  # "pre", "solve", "post", "verify", "batch" 等
    version: str = "0.1.0"
    deprecated: bool = False
    deprecated_by: str | None = None  # 後継プロセスのクラス名


class ProcessMetaclass(type(ABC)):
    """AbstractProcess のメタクラス.

    process() メソッドを自動ラップし、以下を実現する:
    - 実行トレース: どの process() が呼ばれたかを記録
    - プロファイリング: process() 単位の実行時間を自動計測
    """

    _call_stack: ClassVar[list[str]] = []
    _profile_data: ClassVar[dict[str, list[float]]] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # process() が定義されていればラップ
        if "process" in namespace and callable(namespace["process"]):
            original = namespace["process"]

            @functools.wraps(original)
            def traced_process(self, input_data):  # noqa: ANN001
                cls_name = type(self).__name__
                ProcessMetaclass._call_stack.append(cls_name)
                t0 = time.perf_counter()
                try:
                    result = original(self, input_data)
                finally:
                    elapsed = time.perf_counter() - t0
                    ProcessMetaclass._call_stack.pop()
                    if cls_name not in ProcessMetaclass._profile_data:
                        ProcessMetaclass._profile_data[cls_name] = []
                    ProcessMetaclass._profile_data[cls_name].append(elapsed)
                return result

            cls.process = traced_process

        return cls

    @classmethod
    def get_trace(mcs) -> list[str]:
        """現在の実行スタック（デバッグ用）."""
        return list(mcs._call_stack)

    @classmethod
    def get_profile_report(mcs) -> str:
        """全プロセスのプロファイルレポート."""
        lines = ["Process Profile Report", "=" * 40]
        for name, times in sorted(mcs._profile_data.items()):
            n = len(times)
            total = sum(times)
            avg = total / n if n > 0 else 0
            lines.append(f"  {name}: {n} calls, total={total:.3f}s, avg={avg:.3f}s")
        return "\n".join(lines)

    @classmethod
    def reset_profile(mcs) -> None:
        """プロファイルデータをリセット."""
        mcs._profile_data.clear()
        mcs._call_stack.clear()


class AbstractProcess(ABC, Generic[TIn, TOut], metaclass=ProcessMetaclass):
    """全プロセスの基底クラス.

    契約:
    - uses に宣言したプロセスのみを process() 内で使用可能
    - Input/Output型はジェネリックパラメータで明示
    - __init_subclass__ でクラス定義時に制約違反を検出
    - メタクラスが process() を自動ラップし、実行トレース + プロファイリング
    """

    # --- クラス変数（サブクラスで上書き） ---
    meta: ClassVar[ProcessMeta]
    uses: ClassVar[list[type[AbstractProcess]]] = []

    # --- 自動管理 ---
    _registry: ClassVar[dict[str, type[AbstractProcess]]] = {}
    _used_by: ClassVar[list[type[AbstractProcess]]] = []
    _test_class: ClassVar[str | None] = None
    _verify_scripts: ClassVar[list[str]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # 抽象中間クラス（SolverProcess等）はスキップ
        # ABC を直接継承 or process() が未実装の場合は抽象クラスとみなす
        abstract_methods = getattr(cls, "__abstractmethods__", frozenset())
        if abstract_methods or ABC in cls.__bases__:
            return

        # meta 必須チェック
        if not hasattr(cls, "meta") or not isinstance(cls.meta, ProcessMeta):
            raise TypeError(f"{cls.__name__} は ProcessMeta を定義してください")

        # deprecated チェック
        for dep in cls.uses:
            if hasattr(dep, "meta") and dep.meta.deprecated:
                warnings.warn(
                    f"{cls.__name__} は deprecated な {dep.__name__} を使用。"
                    f" 後継: {dep.meta.deprecated_by}",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # レジストリ登録
        cls._registry[cls.__name__] = cls
        # used_by を初期化（クラスごとに独立させる）
        cls._used_by = []

        # uses → used_by 双方向リンク
        for dep in cls.uses:
            if not hasattr(dep, "_used_by") or dep._used_by is AbstractProcess._used_by:
                dep._used_by = []
            dep._used_by.append(cls)

    @abstractmethod
    def process(self, input_data: TIn) -> TOut:
        """メイン処理. サブクラスで実装."""
        ...

    def execute(self, input_data: TIn) -> TOut:
        """process() の公開エントリポイント."""
        return self.process(input_data)

    @classmethod
    def get_dependency_tree(cls) -> dict:
        """再帰的に依存ツリーを返す."""
        return {
            "name": cls.__name__,
            "module": cls.meta.module if hasattr(cls, "meta") else "?",
            "uses": [dep.get_dependency_tree() for dep in cls.uses],
        }

    @classmethod
    def document_markdown(cls) -> str:
        """Markdownドキュメント自動生成."""
        lines = [
            f"## {cls.__name__}",
            f"- **モジュール**: {cls.meta.module}",
            f"- **バージョン**: {cls.meta.version}",
        ]
        if cls.meta.deprecated:
            lines.append(f"- **DEPRECATED** → {cls.meta.deprecated_by}")
        if cls.uses:
            lines.append(f"- **依存**: {', '.join(d.__name__ for d in cls.uses)}")
        if cls._used_by:
            lines.append(f"- **被依存**: {', '.join(d.__name__ for d in cls._used_by)}")
        if cls._test_class:
            lines.append(f"- **テスト**: `{cls._test_class}`")
        if cls._verify_scripts:
            lines.append("- **検証スクリプト**:")
            for vs in cls._verify_scripts:
                lines.append(f"  - `{vs}`")
        return "\n".join(lines)
