"""testing.py の1:1テスト.

テスト対象: xkep_cae/process/testing.py
"""

from __future__ import annotations

import pytest

from xkep_cae.process.base import AbstractProcess, ProcessMeta
from xkep_cae.process.testing import binds_to


class TestBindsTo:
    """binds_to デコレータのテスト."""

    def test_binds_to_sets_test_class(self) -> None:
        """binds_to で _test_class が設定されること."""

        class TargetProcess(AbstractProcess[str, str]):
            meta = ProcessMeta(name="Target", module="test")
            document_path = "docs/dummy.md"
            uses = []

            def process(self, input_data: str) -> str:
                return ""

        @binds_to(TargetProcess)
        class TestTarget:
            pass

        assert TargetProcess._test_class is not None
        assert "TestTarget" in TargetProcess._test_class

    def test_binds_to_rejects_duplicate(self) -> None:
        """同一プロセスに2つ目のテストクラスを紐付けると ValueError."""

        class Target2Process(AbstractProcess[str, str]):
            meta = ProcessMeta(name="Target2", module="test")
            document_path = "docs/dummy.md"
            uses = []

            def process(self, input_data: str) -> str:
                return ""

        @binds_to(Target2Process)
        class TestTarget2A:
            pass

        with pytest.raises(ValueError, match="1:1 対応を維持"):

            @binds_to(Target2Process)
            class TestTarget2B:
                pass
