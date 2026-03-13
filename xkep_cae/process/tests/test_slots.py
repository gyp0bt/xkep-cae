"""StrategySlot テスト.

設計仕様: phase8-design.md §B
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from xkep_cae.process.slots import StrategySlot, collect_strategy_slots, collect_strategy_types

# --- テスト用 Protocol & 具象クラス ---


@runtime_checkable
class DummyStrategy(Protocol):
    def compute(self, x: float) -> float: ...


class GoodImpl:
    """DummyStrategy を満たす."""

    def compute(self, x: float) -> float:
        return x * 2


class BadImpl:
    """DummyStrategy を満たさない."""

    def something_else(self) -> None:
        pass


@runtime_checkable
class AnotherStrategy(Protocol):
    def evaluate(self) -> int: ...


class AnotherImpl:
    def evaluate(self) -> int:
        return 42


# --- StrategySlot を使うクラス ---


class _TestHost:
    strategy_a = StrategySlot(DummyStrategy)
    strategy_b = StrategySlot(AnotherStrategy, required=False)


# --- テスト ---


class TestStrategySlot:
    def test_set_valid(self):
        host = _TestHost()
        impl = GoodImpl()
        host.strategy_a = impl
        assert host.strategy_a is impl

    def test_set_invalid_raises_type_error(self):
        host = _TestHost()
        with pytest.raises(TypeError, match="DummyStrategy を満たしていない"):
            host.strategy_a = BadImpl()

    def test_required_unset_raises_attribute_error(self):
        host = _TestHost()
        with pytest.raises(AttributeError, match="未設定"):
            _ = host.strategy_a

    def test_optional_unset_returns_none(self):
        host = _TestHost()
        assert host.strategy_b is None

    def test_optional_set_none(self):
        host = _TestHost()
        host.strategy_b = None
        assert host.strategy_b is None

    def test_required_set_none_raises(self):
        host = _TestHost()
        with pytest.raises(TypeError, match="None は設定不可"):
            host.strategy_a = None

    def test_class_access_returns_descriptor(self):
        slot = _TestHost.strategy_a
        assert isinstance(slot, StrategySlot)
        assert slot.protocol is DummyStrategy
        assert slot.required is True

    def test_multiple_instances_independent(self):
        h1 = _TestHost()
        h2 = _TestHost()
        impl1 = GoodImpl()
        impl2 = GoodImpl()
        h1.strategy_a = impl1
        h2.strategy_a = impl2
        assert h1.strategy_a is impl1
        assert h2.strategy_a is impl2
        assert h1.strategy_a is not h2.strategy_a


class TestCollectStrategySlots:
    def test_collect_from_class(self):
        slots = collect_strategy_slots(_TestHost)
        assert "strategy_a" in slots
        assert "strategy_b" in slots
        assert len(slots) == 2

    def test_collect_inherits(self):
        class _Child(_TestHost):
            strategy_c = StrategySlot(DummyStrategy, required=False)

        slots = collect_strategy_slots(_Child)
        assert len(slots) == 3
        assert "strategy_c" in slots


class TestCollectStrategyTypes:
    def test_collect_types(self):
        host = _TestHost()
        host.strategy_a = GoodImpl()
        host.strategy_b = AnotherImpl()
        types = collect_strategy_types(host)
        assert GoodImpl in types
        assert AnotherImpl in types

    def test_collect_types_optional_unset(self):
        host = _TestHost()
        host.strategy_a = GoodImpl()
        types = collect_strategy_types(host)
        assert GoodImpl in types
        assert len(types) == 1
