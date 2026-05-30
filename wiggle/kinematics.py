"""撚線機の運動学モデル（単位: mm / rad / s）。

幾何モデル（純粋キネマティクス、応力ひずみは扱わない）:

- 中心芯線: z 軸上の直線
- 外周 strand: ボビン側で半径 R_bobbin、レイプレートを境に半径 R_layer に絞られ、
  下流では一定半径 R_layer のヘリックスを描く
- 機械が角速度 omega で回ると、ボビン位相は φ_i − ω·t（+z 視点で CW 方向に回す
  ことで Z-lay 撚り上げを表現）。下流ヘリックスは過去時刻の位相が空間に焼かれて
  いるので θ(z, t) = φ_i − ω·t + 2π(z − z_lay)/L_pitch となり、ヘリックス模様は
  時間とともに **下流** に流れる（material flow と整合）

符号の物理的根拠: 軸位置 z の材料は時刻 t_pass = t − (z − z_lay)/v_feed に
レイプレートを通過した。その時のボビン位相が φ_i − ω·t_pass であり、
v_feed = ω·L_pitch/(2π) を代入すると上記の式に帰着する。
+ω·t の符号で書くと dz/dt|_θ = −ωL_pitch/(2π) < 0 となり upstream に流れる
「撚りほどき」表現になるので注意。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StranderConfig:
    n_outer: int = 6
    R_layer: float = 5.5
    R_bobbin: float = 120.0
    L_pitch: float = 45.0
    z_lay: float = 250.0
    L_tail_in: float = 450.0
    L_tail_out: float = 400.0
    omega: float = 2 * np.pi / 2.0
    n_segments: int = 400
    core_radius: float = 2.0
    outer_radius: float = 1.8

    @property
    def z0(self) -> float:
        return self.z_lay - self.L_tail_in

    @property
    def z1(self) -> float:
        return self.z_lay + self.L_tail_out

    @property
    def helix_angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(2 * np.pi * self.R_layer, self.L_pitch)))


def _z_grid(cfg: StranderConfig) -> np.ndarray:
    return np.linspace(cfg.z0, cfg.z1, cfg.n_segments)


def outer_strand_centerline(cfg: StranderConfig, i: int, t: float) -> np.ndarray:
    z = _z_grid(cfg)
    phi0 = 2.0 * np.pi * i / cfg.n_outer
    theta_lay = phi0 - cfg.omega * t

    upstream = z < cfg.z_lay
    ramp = np.clip((z - cfg.z0) / cfg.L_tail_in, 0.0, 1.0)
    radius = np.where(
        upstream,
        cfg.R_bobbin + (cfg.R_layer - cfg.R_bobbin) * ramp,
        cfg.R_layer,
    )
    theta = np.where(
        upstream,
        theta_lay,
        theta_lay + 2.0 * np.pi * (z - cfg.z_lay) / cfg.L_pitch,
    )
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def core_strand_centerline(cfg: StranderConfig, t: float) -> np.ndarray:
    z = _z_grid(cfg)
    zeros = np.zeros_like(z)
    return np.column_stack([zeros, zeros, z])


def bobbin_position(cfg: StranderConfig, i: int, t: float) -> np.ndarray:
    phi0 = 2.0 * np.pi * i / cfg.n_outer - cfg.omega * t
    return np.array(
        [cfg.R_bobbin * np.cos(phi0), cfg.R_bobbin * np.sin(phi0), cfg.z0 - 40.0],
        dtype=float,
    )


def bobbin_axis(cfg: StranderConfig, i: int, t: float) -> np.ndarray:
    """ボビンの軸方向（中心軸へ向く半径方向）。装飾円柱の direction に使う。"""
    phi0 = 2.0 * np.pi * i / cfg.n_outer - cfg.omega * t
    return np.array([-np.cos(phi0), -np.sin(phi0), 0.0])
