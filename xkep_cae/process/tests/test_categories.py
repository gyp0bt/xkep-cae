"""プロセスカテゴリの1:1テスト.

テスト対象: xkep_cae/process/categories.py
"""

from __future__ import annotations

import pytest

from xkep_cae.process.base import AbstractProcess, ProcessMeta
from xkep_cae.process.categories import (
    BatchProcess,
    CompatibilityProcess,
    PostProcess,
    PreProcess,
    SolverProcess,
    VerifyProcess,
)

# ============================================================
# テスト用具象プロセス
# ============================================================


class DummyPre(PreProcess[str, str]):
    meta = ProcessMeta(name="Dummy Pre", module="pre", document_path="docs/dummy.md")
    uses = []
    _skip_registry = True

    def process(self, input_data: str) -> str:
        return f"pre:{input_data}"


class DummySolver(SolverProcess[str, str]):
    meta = ProcessMeta(name="Dummy Solver", module="solve", document_path="docs/dummy.md")
    uses = []
    _skip_registry = True

    def process(self, input_data: str) -> str:
        return f"solve:{input_data}"


class DummyPost(PostProcess[str, str]):
    meta = ProcessMeta(name="Dummy Post", module="post", document_path="docs/dummy.md")
    uses = []
    _skip_registry = True

    def process(self, input_data: str) -> str:
        return f"post:{input_data}"


class DummyVerify(VerifyProcess[str, str]):
    meta = ProcessMeta(name="Dummy Verify", module="verify", document_path="docs/dummy.md")
    uses = []
    _skip_registry = True

    def process(self, input_data: str) -> str:
        return f"verify:{input_data}"


class DummyBatch(BatchProcess[str, str]):
    meta = ProcessMeta(name="Dummy Batch", module="batch", document_path="docs/dummy.md")
    uses = [DummyPre, DummySolver, DummyPost]
    _skip_registry = True

    def process(self, input_data: str) -> str:
        return f"batch:{input_data}"


# ============================================================
# TestCategoriesAPI
# ============================================================


class TestCategoriesAPI:
    """カテゴリ中間クラスのAPI契約テスト."""

    def test_categories_are_abstract(self) -> None:
        """カテゴリクラスは直接インスタンス化できない."""
        for cls in (
            PreProcess,
            SolverProcess,
            PostProcess,
            VerifyProcess,
            BatchProcess,
            CompatibilityProcess,
        ):
            with pytest.raises(TypeError):
                cls()  # type: ignore[abstract]

    def test_concrete_inherits_abstract_process(self) -> None:
        """具象クラスは AbstractProcess のサブクラス."""
        for cls in (DummyPre, DummySolver, DummyPost, DummyVerify, DummyBatch):
            assert issubclass(cls, AbstractProcess)

    def test_skip_registry_excludes_test_fixtures(self) -> None:
        """_skip_registry = True のテスト用クラスはレジストリに登録されないこと."""
        for name in ("DummyPre", "DummySolver", "DummyPost", "DummyVerify", "DummyBatch"):
            assert name not in AbstractProcess._registry

    def test_concrete_execution(self) -> None:
        """各カテゴリの具象クラスが正しく実行されること."""
        assert DummyPre().process("x") == "pre:x"
        assert DummySolver().process("x") == "solve:x"
        assert DummyPost().process("x") == "post:x"
        assert DummyVerify().process("x") == "verify:x"
        assert DummyBatch().process("x") == "batch:x"

    def test_batch_uses_multiple_processes(self) -> None:
        """BatchProcess が複数プロセスに依存できること."""
        assert len(DummyBatch.uses) == 3
        assert DummyPre in DummyBatch.uses
        assert DummySolver in DummyBatch.uses
        assert DummyPost in DummyBatch.uses

    def test_compatibility_process_is_abstract(self) -> None:
        """CompatibilityProcess は AbstractProcess のサブクラス."""
        assert issubclass(CompatibilityProcess, AbstractProcess)
