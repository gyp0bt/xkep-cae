"""梁の断面特性モデル.

Cowperのせん断補正係数:
  Abaqus は横せん断剛性に Cowper (1966) のν依存せん断補正係数を使用する。
  xkep-cae では BeamSection2D.cowper_kappa(nu) で同等の値を取得できる。

  矩形断面: κ = 10(1+ν) / (12+11ν)   （Abaqusデフォルト）
  円形断面: κ = 6(1+ν) / (7+6ν)       （Abaqusデフォルト）
  一般断面: κ = 5/6 にフォールバック

  参考文献:
    Cowper, G.R. (1966) "The shear coefficient in Timoshenko's beam theory",
    J. Applied Mechanics, 33, 335-340.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BeamSection2D:
    """2D梁の断面特性.

    Attributes:
        A: 断面積
        I: 断面二次モーメント（Iz）
        shape: 断面形状 ("rectangle", "circle", "general")
    """

    A: float
    I: float  # noqa: E741
    shape: str = field(default="general", repr=False)

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError(f"断面積 A は正値でなければなりません: {self.A}")
        if self.I <= 0:
            raise ValueError(f"断面二次モーメント I は正値でなければなりません: {self.I}")
        valid_shapes = ("rectangle", "circle", "general")
        if self.shape not in valid_shapes:
            raise ValueError(
                f"shape は {valid_shapes} のいずれかでなければなりません: {self.shape}"
            )

    def cowper_kappa(self, nu: float) -> float:
        """Cowper (1966) のν依存せん断補正係数を返す.

        Abaqusと同じ定式化。断面形状に応じた厳密なκ値を計算する。

        Args:
            nu: ポアソン比

        Returns:
            κ: せん断補正係数

        Abaqusとの対応:
            矩形断面: κ = 10(1+ν)/(12+11ν)  （ν=0.3 で κ≈0.8497）
            円形断面: κ = 6(1+ν)/(7+6ν)      （ν=0.3 で κ≈0.8864）
            xkep-cae旧デフォルト: κ = 5/6 ≈ 0.8333（ν非依存）
        """
        if self.shape == "rectangle":
            return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
        elif self.shape == "circle":
            return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
        else:
            return 5.0 / 6.0

    @classmethod
    def rectangle(cls, b: float, h: float) -> BeamSection2D:
        """矩形断面を生成する.

        Args:
            b: 幅
            h: 高さ

        Returns:
            BeamSection2D インスタンス
        """
        return cls(A=b * h, I=b * h**3 / 12.0, shape="rectangle")

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
        return cls(A=math.pi * r**2, I=math.pi * r**4 / 4.0, shape="circle")
