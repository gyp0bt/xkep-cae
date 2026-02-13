"""梁の断面特性モデル.

2Dモデル: BeamSection2D — 平面内曲げ（Iz のみ）
3Dモデル: BeamSection   — 二軸曲げ + ねじり（Iy, Iz, J）

Cowperのせん断補正係数:
  Abaqus は横せん断剛性に Cowper (1966) のν依存せん断補正係数を使用する。
  xkep-cae では BeamSection2D.cowper_kappa(nu) / BeamSection.cowper_kappa_y(nu) 等で
  同等の値を取得できる。

  矩形断面: κ = 10(1+ν) / (12+11ν)   （Abaqusデフォルト）
  円形断面: κ = 6(1+ν) / (7+6ν)       （Abaqusデフォルト）
  一般断面: κ = 5/6 にフォールバック

  参考文献:
    Cowper, G.R. (1966) "The shear coefficient in Timoshenko's beam theory",
    J. Applied Mechanics, 33, 335-340.
"""

from __future__ import annotations

import math
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
class BeamSection:
    """3D梁の断面特性.

    二軸曲げ + ねじりに対応する一般的な梁断面モデル。

    Attributes:
        A:  断面積
        Iy: y軸（局所）まわり断面二次モーメント（xz面内曲げ）
        Iz: z軸（局所）まわり断面二次モーメント（xy面内曲げ）
        J:  ねじり定数（St. Venant）
        shape: 断面形状 ("rectangle", "circle", "general")

    座標系の規約（Abaqus準拠）:
        - x: 梁軸方向（節点1→節点2）
        - y: 局所y軸（断面の第1主軸方向、ユーザー定義 or 自動）
        - z: 局所z軸（x × y で決定）
        - Iy: y軸まわりの曲げ → xz面内のたわみに関与
        - Iz: z軸まわりの曲げ → xy面内のたわみに関与
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
        """y方向せん断のCowper補正係数（xz面内曲げのせん断）."""
        return _cowper_kappa(self.shape, nu)

    def cowper_kappa_z(self, nu: float) -> float:
        """z方向せん断のCowper補正係数（xy面内曲げのせん断）.

        矩形断面では y/z 方向で同じ Cowper κ（正方形断面を想定）。
        長方形断面で異なる κ が必要な場合は将来拡張。
        """
        return _cowper_kappa(self.shape, nu)

    def to_2d(self) -> BeamSection2D:
        """xy面内（Iz ベース）の 2D 断面に変換する."""
        return BeamSection2D(A=self.A, I=self.Iz, shape=self.shape)

    @classmethod
    def rectangle(cls, b: float, h: float) -> BeamSection:
        """矩形断面を生成する.

        Args:
            b: 幅（y方向）
            h: 高さ（z方向）

        Returns:
            BeamSection インスタンス

        断面二次モーメント:
            Iy = b·h³/12（z方向高さ → xz面の曲げ）
            Iz = h·b³/12（y方向幅 → xy面の曲げ）

        ねじり定数（St. Venant, 近似）:
            J ≈ a·b³ · (1/3 - 0.21·b/a · (1 - b⁴/(12a⁴)))
            ここで a = max(b,h), b_min = min(b,h)
        """
        A = b * h
        Iy = b * h**3 / 12.0
        Iz = h * b**3 / 12.0

        # St. Venant ねじり定数（矩形断面の近似公式）
        a_long = max(b, h)
        b_short = min(b, h)
        ratio = b_short / a_long
        J = a_long * b_short**3 * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))

        return cls(A=A, Iy=Iy, Iz=Iz, J=J, shape="rectangle")

    @classmethod
    def circle(cls, d: float) -> BeamSection:
        """円形断面を生成する.

        Args:
            d: 直径

        Returns:
            BeamSection インスタンス

        断面二次モーメント:
            Iy = Iz = π·d⁴/64

        ねじり定数:
            J = π·d⁴/32（= 2·Iy、円形断面の厳密解）
        """
        r = d / 2.0
        A = math.pi * r**2
        I_val = math.pi * d**4 / 64.0
        J = math.pi * d**4 / 32.0

        return cls(A=A, Iy=I_val, Iz=I_val, J=J, shape="circle")

    @classmethod
    def pipe(cls, d_outer: float, d_inner: float) -> BeamSection:
        """中空円形（パイプ）断面を生成する.

        Args:
            d_outer: 外径
            d_inner: 内径

        Returns:
            BeamSection インスタンス
        """
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
