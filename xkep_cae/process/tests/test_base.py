"""AbstractProcess 基底クラスの1:1テスト.

テスト対象: xkep_cae/process/base.py
"""

from __future__ import annotations

import warnings

import pytest

from xkep_cae.process.base import AbstractProcess, ProcessMeta, ProcessMetaclass

# ============================================================
# テスト用プロセス定義
# ============================================================


class DummyProcessA(AbstractProcess[str, str]):
    meta = ProcessMeta(name="Dummy A", module="test")
    document_path = "docs/dummy.md"
    uses = []

    def process(self, input_data: str) -> str:
        return f"A:{input_data}"


class DummyProcessB(AbstractProcess[str, str]):
    meta = ProcessMeta(name="Dummy B", module="test")
    document_path = "docs/dummy.md"
    uses = [DummyProcessA]

    def __init__(self) -> None:
        self._a = DummyProcessA()

    def process(self, input_data: str) -> str:
        return f"B:{self._a.process(input_data)}"


class DeprecatedProcess(AbstractProcess[str, str]):
    meta = ProcessMeta(
        name="Legacy",
        module="test",
        deprecated=True,
        deprecated_by="DummyProcessA",
    )
    document_path = "docs/dummy.md"
    uses = []

    def process(self, input_data: str) -> str:
        return "legacy"


# ============================================================
# TestAbstractProcessAPI — API仕様テスト
# ============================================================


class TestAbstractProcessAPI:
    """AbstractProcess の API 契約テスト."""

    def test_registry_contains_concrete(self) -> None:
        """具象クラスがレジストリに登録されること."""
        assert "DummyProcessA" in AbstractProcess._registry
        assert "DummyProcessB" in AbstractProcess._registry

    def test_meta_required(self) -> None:
        """meta 未定義は TypeError."""
        with pytest.raises(TypeError, match="ProcessMeta を定義してください"):

            class BadProcess(AbstractProcess[str, str]):
                uses = []

                def process(self, input_data: str) -> str:
                    return ""

    def test_document_path_required(self) -> None:
        """document_path 未定義は TypeError."""
        with pytest.raises(TypeError, match="document_path.*を定義してください"):

            class NoDocProcess(AbstractProcess[str, str]):
                meta = ProcessMeta(name="NoDoc", module="test")
                uses = []

                def process(self, input_data: str) -> str:
                    return ""

    def test_document_path_missing_file(self) -> None:
        """document_path のファイルが存在しない場合は FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ドキュメントが見つかりません"):

            class BadDocProcess(AbstractProcess[str, str]):
                meta = ProcessMeta(name="BadDoc", module="test")
                document_path = "docs/nonexistent.md"
                uses = []

                def process(self, input_data: str) -> str:
                    return ""

    def test_document_path_in_markdown(self) -> None:
        """document_markdown() に設計文書パスが含まれること."""
        md = DummyProcessA.document_markdown()
        assert "docs/dummy.md" in md

    def test_process_execution(self) -> None:
        """process() が正しく実行されること."""
        a = DummyProcessA()
        assert a.process("x") == "A:x"

    def test_execute_delegates_to_process(self) -> None:
        """execute() は process() に委譲."""
        a = DummyProcessA()
        assert a.execute("y") == "A:y"

    def test_uses_builds_used_by(self) -> None:
        """uses 宣言で used_by が自動構築されること."""
        assert DummyProcessB in DummyProcessA._used_by

    def test_deprecated_warning(self) -> None:
        """deprecated プロセスを uses すると DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class UserOfDeprecated(AbstractProcess[str, str]):
                meta = ProcessMeta(name="User of deprecated", module="test")
                document_path = "docs/dummy.md"
                uses = [DeprecatedProcess]

                def process(self, input_data: str) -> str:
                    return ""

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)

    def test_dependency_tree(self) -> None:
        """get_dependency_tree() が再帰的にツリーを返すこと."""
        tree = DummyProcessB.get_dependency_tree()
        assert tree["name"] == "DummyProcessB"
        assert len(tree["uses"]) == 1
        assert tree["uses"][0]["name"] == "DummyProcessA"

    def test_document_markdown(self) -> None:
        """document_markdown() がMarkdownを生成すること."""
        md = DummyProcessA.document_markdown()
        assert "## DummyProcessA" in md
        assert "test" in md


# ============================================================
# TestProcessMetaclass — メタクラスのトレース・プロファイリング
# ============================================================


class TestProcessMetaclass:
    """ProcessMetaclass のトレース・プロファイリングテスト."""

    def setup_method(self) -> None:
        ProcessMetaclass.reset_profile()

    def test_profile_records_timing(self) -> None:
        """process() 実行でプロファイルが記録されること."""
        a = DummyProcessA()
        a.process("test")
        assert "DummyProcessA" in ProcessMetaclass._profile_data
        assert len(ProcessMetaclass._profile_data["DummyProcessA"]) == 1

    def test_profile_report_format(self) -> None:
        """get_profile_report() がレポート文字列を返すこと."""
        a = DummyProcessA()
        a.process("test")
        report = ProcessMetaclass.get_profile_report()
        assert "DummyProcessA" in report
        assert "1 calls" in report

    def test_nested_trace(self) -> None:
        """ネストした process() 呼び出しでスタックが正しく管理されること."""
        b = DummyProcessB()
        b.process("nested")
        # 実行後はスタックが空
        assert ProcessMetaclass.get_trace() == []
        # 両方のプロセスが記録されている
        assert "DummyProcessA" in ProcessMetaclass._profile_data
        assert "DummyProcessB" in ProcessMetaclass._profile_data

    def test_reset_clears_data(self) -> None:
        """reset_profile() でデータがクリアされること."""
        a = DummyProcessA()
        a.process("test")
        ProcessMetaclass.reset_profile()
        assert ProcessMetaclass._profile_data == {}
        assert ProcessMetaclass._call_stack == []
