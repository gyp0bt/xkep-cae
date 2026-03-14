"""ProcessRegistry のテスト.

テスト対象: xkep_cae/process/registry.py
"""

from __future__ import annotations

from xkep_cae.process.base import AbstractProcess, ProcessMeta
from xkep_cae.process.registry import ProcessRegistry
from xkep_cae.process.tests.test_base import DummyProcessA, DummyProcessB  # noqa: F401

# ============================================================
# TestProcessRegistryAPI — ProcessRegistry の基本 API テスト
# ============================================================


class TestProcessRegistryAPI:
    """ProcessRegistry の API 契約テスト."""

    def test_default_singleton(self) -> None:
        """default() が常に同一インスタンスを返すこと."""
        r1 = ProcessRegistry.default()
        r2 = ProcessRegistry.default()
        assert r1 is r2

    def test_registry_contains_concrete(self) -> None:
        """AbstractProcess サブクラスが default レジストリに登録されていること."""
        registry = ProcessRegistry.default()
        # test_base.py の DummyProcessA が登録済みのはず
        assert len(registry) > 0

    def test_proxy_backward_compat(self) -> None:
        """AbstractProcess._registry が ProcessRegistry と同期していること."""
        registry = ProcessRegistry.default()
        for name in registry:
            assert name in AbstractProcess._registry

    def test_contains(self) -> None:
        """__contains__ が動作すること."""
        registry = ProcessRegistry.default()
        # レジストリに何か登録されていれば OK
        names = list(registry.keys())
        if names:
            assert names[0] in registry

    def test_items_keys_values(self) -> None:
        """items/keys/values が dict 互換で動作すること."""
        registry = ProcessRegistry.default()
        items = list(registry.items())
        keys = list(registry.keys())
        values = list(registry.values())
        assert len(items) == len(keys) == len(values)

    def test_register_new_process(self) -> None:
        """register() で新しいプロセスを登録できること."""
        registry = ProcessRegistry.default()

        class _TempRegistryTestProcess(AbstractProcess[str, str]):
            meta = ProcessMeta(name="RegistryTest", module="test", document_path="docs/dummy.md")
            _skip_registry = True
            uses = []

            def process(self, input_data: str) -> str:
                return input_data

        # _skip_registry=True なので自動登録されていない
        assert "_TempRegistryTestProcess" not in registry

        # 手動登録
        registry.register(_TempRegistryTestProcess)
        assert "_TempRegistryTestProcess" in registry
        assert registry.get("_TempRegistryTestProcess") is _TempRegistryTestProcess

        # クリーンアップ
        del registry._store["_TempRegistryTestProcess"]


# ============================================================
# TestProcessRegistryIsolation — テスト隔離
# ============================================================


class TestProcessRegistryIsolation:
    """ProcessRegistry.isolate() のテスト."""

    def test_isolate_creates_copy(self) -> None:
        """isolate() がスナップショットコピーを返すこと."""
        original = ProcessRegistry.default()
        isolated = original.isolate()
        assert isolated is not original
        assert len(isolated) == len(original)

    def test_isolate_does_not_affect_original(self) -> None:
        """隔離レジストリへの変更が元のレジストリに影響しないこと."""
        original = ProcessRegistry.default()
        isolated = original.isolate()
        original_count = len(original)

        # 隔離レジストリに追加
        isolated._store["__test_isolated__"] = type  # type: ignore[assignment]
        assert "__test_isolated__" in isolated
        assert "__test_isolated__" not in original
        assert len(original) == original_count


# ============================================================
# TestProcessRegistryFiltering — フィルタリング
# ============================================================


class TestProcessRegistryFiltering:
    """ProcessRegistry のフィルタリングテスト."""

    def test_concrete_processes(self) -> None:
        """concrete_processes() がテストフィクスチャを除外すること."""
        registry = ProcessRegistry.default()
        concrete = registry.concrete_processes()
        for _name, cls in concrete:
            module = getattr(cls, "__module__", "")
            assert ".tests." not in module

    def test_non_deprecated(self) -> None:
        """non_deprecated() が deprecated プロセスを除外すること."""
        registry = ProcessRegistry.default()
        non_dep = registry.non_deprecated()
        for _name, cls in non_dep:
            if hasattr(cls, "meta"):
                assert not cls.meta.deprecated

    def test_dependencies_of(self) -> None:
        """dependencies_of() が uses を正しく返すこと."""
        registry = ProcessRegistry.default()
        # テスト用 DummyProcessB は DummyProcessA に依存
        if "DummyProcessB" in registry:
            deps = registry.dependencies_of("DummyProcessB")
            assert "DummyProcessA" in deps

    def test_dependencies_of_unknown(self) -> None:
        """存在しないプロセス名で空リストを返すこと."""
        registry = ProcessRegistry.default()
        assert registry.dependencies_of("__nonexistent__") == []

    def test_dependants_of(self) -> None:
        """dependants_of() が used_by を正しく返すこと."""
        registry = ProcessRegistry.default()
        if "DummyProcessA" in registry:
            dependants = registry.dependants_of("DummyProcessA")
            assert "DummyProcessB" in dependants
