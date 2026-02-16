"""ファイバーモデル断面.

断面をファイバー（微小断面要素）に分割し、各ファイバーの位置と面積を保持する。
Cosserat rod の弾塑性解析（曲げの塑性化）において、断面内のひずみ分布を
ファイバーごとに追跡するために使用する。

ファイバーひずみ:
  epsilon_i = Gamma_1 + kappa_2 * z_i - kappa_3 * y_i

断面力（ファイバー積分）:
  N  = Sum(sigma_i * A_i)
  My = Sum(sigma_i * z_i * A_i)
  Mz = -Sum(sigma_i * y_i * A_i)

参考文献:
  - de Souza Neto et al. (2008) "Computational Methods for Plasticity", Ch.14
  - Spacone et al. (1996) "Fibre beam-column model for non-linear analysis"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FiberSection:
    """ファイバーモデル断面.

    断面を離散的なファイバーに分割し、各ファイバーの位置 (y_i, z_i) と
    面積 A_i を保持する。弾塑性構成則を各ファイバーに独立に適用することで、
    断面内の非一様な応力分布（曲げの塑性化）を表現できる。

    座標系（Cosserat rod 準拠）:
      - x: 梁軸方向
      - y: 局所y軸（断面の第1主軸方向）
      - z: 局所z軸（x x y で決定）

    Attributes:
        y: (n_fibers,) 各ファイバーのy座標
        z: (n_fibers,) 各ファイバーのz座標
        areas: (n_fibers,) 各ファイバーの面積
        J: ねじり定数（St. Venant、弾性のまま）
        shape: 断面形状 ("rectangle", "circle", "general")
    """

    y: np.ndarray
    z: np.ndarray
    areas: np.ndarray
    J: float
    shape: str = field(default="general", repr=False)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        self.areas = np.asarray(self.areas, dtype=float)
        if self.y.ndim != 1:
            raise ValueError("y は1次元配列でなければなりません")
        if self.z.ndim != 1:
            raise ValueError("z は1次元配列でなければなりません")
        if self.areas.ndim != 1:
            raise ValueError("areas は1次元配列でなければなりません")
        n = len(self.y)
        if len(self.z) != n or len(self.areas) != n:
            raise ValueError(
                f"y, z, areas の長さが一致しません: {len(self.y)}, {len(self.z)}, {len(self.areas)}"
            )
        if n == 0:
            raise ValueError("ファイバー数は1以上でなければなりません")
        if np.any(self.areas <= 0):
            raise ValueError("ファイバー面積は全て正値でなければなりません")
        if self.J <= 0:
            raise ValueError(f"ねじり定数 J は正値でなければなりません: {self.J}")

    @property
    def n_fibers(self) -> int:
        """ファイバー数."""
        return len(self.areas)

    @property
    def A(self) -> float:
        """総断面積."""
        return float(np.sum(self.areas))

    @property
    def Iy(self) -> float:
        """y軸まわりの断面二次モーメント（ファイバー近似）."""
        return float(np.sum(self.areas * self.z**2))

    @property
    def Iz(self) -> float:
        """z軸まわりの断面二次モーメント（ファイバー近似）."""
        return float(np.sum(self.areas * self.y**2))

    def cowper_kappa_y(self, nu: float) -> float:
        """y方向せん断のCowper補正係数."""
        return _cowper_kappa(self.shape, nu)

    def cowper_kappa_z(self, nu: float) -> float:
        """z方向せん断のCowper補正係数."""
        return _cowper_kappa(self.shape, nu)

    @classmethod
    def rectangle(cls, b: float, h: float, ny: int, nz: int) -> FiberSection:
        """矩形断面をファイバー分割する.

        Args:
            b: 幅（y方向）
            h: 高さ（z方向）
            ny: y方向の分割数
            nz: z方向の分割数

        Returns:
            FiberSection インスタンス（ny * nz ファイバー）
        """
        if b <= 0 or h <= 0:
            raise ValueError(f"b, h は正値: b={b}, h={h}")
        if ny < 1 or nz < 1:
            raise ValueError(f"ny, nz は1以上: ny={ny}, nz={nz}")

        dy = b / ny
        dz = h / nz
        A_fiber = dy * dz

        y_centers = np.linspace(-b / 2 + dy / 2, b / 2 - dy / 2, ny)
        z_centers = np.linspace(-h / 2 + dz / 2, h / 2 - dz / 2, nz)

        yy, zz = np.meshgrid(y_centers, z_centers, indexing="ij")
        y_flat = yy.ravel()
        z_flat = zz.ravel()
        areas = np.full(ny * nz, A_fiber)

        # St. Venant ねじり定数（矩形断面の近似公式）
        a_long = max(b, h)
        b_short = min(b, h)
        ratio = b_short / a_long
        J = a_long * b_short**3 * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))

        return cls(y=y_flat, z=z_flat, areas=areas, J=J, shape="rectangle")

    @classmethod
    def circle(cls, d: float, nr: int, nt: int) -> FiberSection:
        """円形断面をファイバー分割する.

        同心円リング × 等角度でファイバーを配置する。

        Args:
            d: 直径
            nr: 半径方向の分割数
            nt: 周方向の分割数

        Returns:
            FiberSection インスタンス
        """
        if d <= 0:
            raise ValueError(f"直径 d は正値: {d}")
        if nr < 1 or nt < 1:
            raise ValueError(f"nr, nt は1以上: nr={nr}, nt={nt}")

        R = d / 2.0
        dr = R / nr

        y_list: list[float] = []
        z_list: list[float] = []
        a_list: list[float] = []

        for ir in range(nr):
            r_inner = ir * dr
            r_outer = (ir + 1) * dr
            r_mid = (r_inner + r_outer) / 2.0
            # リングの面積を等角度で分割
            ring_area = math.pi * (r_outer**2 - r_inner**2)
            fiber_area = ring_area / nt

            for it in range(nt):
                theta = 2.0 * math.pi * (it + 0.5) / nt
                y_list.append(r_mid * math.cos(theta))
                z_list.append(r_mid * math.sin(theta))
                a_list.append(fiber_area)

        J = math.pi * d**4 / 32.0

        return cls(
            y=np.array(y_list),
            z=np.array(z_list),
            areas=np.array(a_list),
            J=J,
            shape="circle",
        )

    @classmethod
    def pipe(cls, d_outer: float, d_inner: float, nr: int, nt: int) -> FiberSection:
        """パイプ（中空円形）断面をファイバー分割する.

        Args:
            d_outer: 外径
            d_inner: 内径
            nr: 半径方向（肉厚方向）の分割数
            nt: 周方向の分割数

        Returns:
            FiberSection インスタンス
        """
        if d_inner >= d_outer:
            raise ValueError(
                f"内径は外径より小さくなければなりません: d_inner={d_inner}, d_outer={d_outer}"
            )
        if nr < 1 or nt < 1:
            raise ValueError(f"nr, nt は1以上: nr={nr}, nt={nt}")

        R_outer = d_outer / 2.0
        R_inner = d_inner / 2.0
        dr = (R_outer - R_inner) / nr

        y_list: list[float] = []
        z_list: list[float] = []
        a_list: list[float] = []

        for ir in range(nr):
            r_inner_ring = R_inner + ir * dr
            r_outer_ring = R_inner + (ir + 1) * dr
            r_mid = (r_inner_ring + r_outer_ring) / 2.0
            ring_area = math.pi * (r_outer_ring**2 - r_inner_ring**2)
            fiber_area = ring_area / nt

            for it in range(nt):
                theta = 2.0 * math.pi * (it + 0.5) / nt
                y_list.append(r_mid * math.cos(theta))
                z_list.append(r_mid * math.sin(theta))
                a_list.append(fiber_area)

        J = math.pi * (d_outer**4 - d_inner**4) / 32.0

        return cls(
            y=np.array(y_list),
            z=np.array(z_list),
            areas=np.array(a_list),
            J=J,
            shape="circle",
        )


def _cowper_kappa(shape: str, nu: float) -> float:
    """Cowper (1966) のnu依存せん断補正係数."""
    if shape == "rectangle":
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
    elif shape == "circle":
        return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    else:
        return 5.0 / 6.0
