"""梁の断面特性モデル."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BeamSection2D:
    """2D梁の断面特性.

    Attributes:
        A: 断面積
        I: 断面二次モーメント（Iz）
    """

    A: float
    I: float  # noqa: E741

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError(f"断面積 A は正値でなければなりません: {self.A}")
        if self.I <= 0:
            raise ValueError(f"断面二次モーメント I は正値でなければなりません: {self.I}")

    @classmethod
    def rectangle(cls, b: float, h: float) -> BeamSection2D:
        """矩形断面を生成する.

        Args:
            b: 幅
            h: 高さ

        Returns:
            BeamSection2D インスタンス
        """
        return cls(A=b * h, I=b * h**3 / 12.0)

    @classmethod
    def circle(cls, d: float) -> BeamSection2D:
        """円形断面を生成する.

        Args:
            d: 直径

        Returns:
            BeamSection2D インスタンス
        """
        import math

        r = d / 2.0
        return cls(A=math.pi * r**2, I=math.pi * r**4 / 4.0)
