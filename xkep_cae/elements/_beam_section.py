"""梁の断面特性モデル.

2Dモデル: BeamSection2DInput — 平面内曲げ（Iz のみ）
3Dモデル: BeamSectionInput   — 二軸曲げ + ねじり（Iy, Iz, J）

Cowperのせん断補正係数:
  矩形断面: κ = 10(1+ν) / (12+11ν)
  円形断面: κ = 6(1+ν) / (7+6ν)
  一般断面: κ = 5/6 にフォールバック
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BeamSection2DInput:
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
        """Cowper (1966) のν依存せん断補正係数を返す."""
        return _cowper_kappa(self.shape, nu)

    @classmethod
    def rectangle(cls, b: float, h: float) -> BeamSection2DInput:
        """矩形断面を生成する."""
        return cls(A=b * h, I=b * h**3 / 12.0, shape="rectangle")

    @classmethod
    def circle(cls, d: float) -> BeamSection2DInput:
        """円形断面を生成する."""
        r = d / 2.0
        return cls(A=math.pi * r**2, I=math.pi * r**4 / 4.0, shape="circle")


def _cowper_kappa(shape: str, nu: float) -> float:
    """Cowper (1966) のν依存せん断補正係数を返す（共通ヘルパー）."""
    if shape == "rectangle":
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
    elif shape == "circle":
        return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    else:
        return 5.0 / 6.0


@dataclass(frozen=True)
class BeamSectionInput:
    """3D梁の断面特性.

    二軸曲げ + ねじりに対応する一般的な梁断面モデル。

    Attributes:
        A:  断面積
        Iy: y軸（局所）まわり断面二次モーメント（xz面内曲げ）
        Iz: z軸（局所）まわり断面二次モーメント（xy面内曲げ）
        J:  ねじり定数（St. Venant）
        shape: 断面形状 ("rectangle", "circle", "general")
    """

    A: float
    Iy: float
    Iz: float
    J: float
    shape: str = field(default="general", repr=False)

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError(f"断面積 A は正値でなければなりません: {self.A}")
        if self.Iy <= 0:
            raise ValueError(f"断面二次モーメント Iy は正値でなければなりません: {self.Iy}")
        if self.Iz <= 0:
            raise ValueError(f"断面二次モーメント Iz は正値でなければなりません: {self.Iz}")
        if self.J <= 0:
            raise ValueError(f"ねじり定数 J は正値でなければなりません: {self.J}")
        valid_shapes = ("rectangle", "circle", "general")
        if self.shape not in valid_shapes:
            raise ValueError(
                f"shape は {valid_shapes} のいずれかでなければなりません: {self.shape}"
            )

    def cowper_kappa_y(self, nu: float) -> float:
        """y方向せん断のCowper補正係数."""
        return _cowper_kappa(self.shape, nu)

    def cowper_kappa_z(self, nu: float) -> float:
        """z方向せん断のCowper補正係数."""
        return _cowper_kappa(self.shape, nu)

    def to_2d(self) -> BeamSection2DInput:
        """xy面内（Iz ベース）の 2D 断面に変換する."""
        return BeamSection2DInput(A=self.A, I=self.Iz, shape=self.shape)

    @classmethod
    def rectangle(cls, b: float, h: float) -> BeamSectionInput:
        """矩形断面を生成する."""
        A = b * h
        Iy = b * h**3 / 12.0
        Iz = h * b**3 / 12.0
        a_long = max(b, h)
        b_short = min(b, h)
        ratio = b_short / a_long
        J = a_long * b_short**3 * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))
        return cls(A=A, Iy=Iy, Iz=Iz, J=J, shape="rectangle")

    @classmethod
    def circle(cls, d: float) -> BeamSectionInput:
        """円形断面を生成する."""
        r = d / 2.0
        A = math.pi * r**2
        I_val = math.pi * d**4 / 64.0
        J = math.pi * d**4 / 32.0
        return cls(A=A, Iy=I_val, Iz=I_val, J=J, shape="circle")

    @classmethod
    def pipe(cls, d_outer: float, d_inner: float) -> BeamSectionInput:
        """中空円形（パイプ）断面を生成する."""
        if d_inner >= d_outer:
            raise ValueError(
                f"内径は外径より小さくなければなりません: d_inner={d_inner}, d_outer={d_outer}"
            )
        r_o = d_outer / 2.0
        r_i = d_inner / 2.0
        A = math.pi * (r_o**2 - r_i**2)
        I_val = math.pi * (d_outer**4 - d_inner**4) / 64.0
        J = math.pi * (d_outer**4 - d_inner**4) / 32.0
        return cls(A=A, Iy=I_val, Iz=I_val, J=J, shape="circle")
