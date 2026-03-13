"""StrategySlot — Strategy slot の型付きディスクリプタ.

クラス変数として宣言し、インスタンスで具象 Strategy を設定する。
設計仕様: phase8-design.md §B

メタクラス (ProcessMetaclass.__new__) との相互作用:
- __set_name__ は type.__new__ 内で呼ばれる（メタクラスの __new__ より前）
- ディスクリプタの __get__/__set__ はインスタンスアクセス時に呼ばれる

_runtime_uses からの移行:
1. StrategySlot が __set_name__ で自動登録
2. collect_strategy_types() でクラスのスロットから型情報を収集
3. effective_uses() は既存の _runtime_uses を優先し、StrategySlot をフォールバック
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")

_SENTINEL = object()


class StrategySlot(Generic[T]):
    """Strategy slot の型付きディスクリプタ.

    クラス変数として宣言し、インスタンスで具象 Strategy を設定する。

    Usage:
        class MySolver(SolverProcess[...]):
            penalty = StrategySlot(PenaltyStrategy)
            friction = StrategySlot(FrictionStrategy, required=False)

            def __init__(self, strategies):
                self.penalty = strategies.penalty    # __set__ で Protocol 検証
                self.friction = strategies.friction
    """

    def __init__(self, protocol: type[T], *, required: bool = True) -> None:
        self.protocol = protocol
        self.required = required
        self._attr_name: str = ""
        self._public_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = f"_slot_{name}"
        self._public_name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        val = getattr(obj, self._attr_name, _SENTINEL)
        if val is _SENTINEL:
            if self.required:
                raise AttributeError(
                    f"{type(obj).__name__}.{self._public_name} は未設定 "
                    f"(required StrategySlot)"
                )
            return None
        return val

    def __set__(self, obj: Any, value: T) -> None:
        if value is None:
            if self.required:
                raise TypeError(
                    f"{type(obj).__name__}.{self._public_name}: "
                    f"required StrategySlot に None は設定不可"
                )
            setattr(obj, self._attr_name, None)
            return
        if not isinstance(value, self.protocol):
            raise TypeError(
                f"{type(obj).__name__}.{self._public_name}: "
                f"{type(value).__name__} は {self.protocol.__name__} を満たしていない"
            )
        setattr(obj, self._attr_name, value)


def collect_strategy_slots(cls: type) -> dict[str, StrategySlot]:
    """クラスの全 StrategySlot を name → StrategySlot で返す."""
    result = {}
    for klass in reversed(cls.__mro__):
        for name, attr in vars(klass).items():
            if isinstance(attr, StrategySlot):
                result[name] = attr
    return result


def collect_strategy_types(instance: Any) -> list[type]:
    """インスタンスの全 StrategySlot に設定された具象クラスの型リストを返す.

    _runtime_uses の代替として使用可能。
    """
    slots = collect_strategy_slots(type(instance))
    types = []
    for _name, slot in slots.items():
        val = getattr(instance, slot._attr_name, _SENTINEL)
        if val is not _SENTINEL and val is not None:
            types.append(type(val))
    return types
