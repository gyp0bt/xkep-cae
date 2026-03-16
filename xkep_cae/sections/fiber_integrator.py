"""ファイバー断面積分モジュール.

FiberSection と Plasticity1D を統合し、断面力と consistent tangent を計算する。
CR梁要素の弾塑性解析に使用する。

断面変形 → ファイバーひずみ:
  epsilon_i = eps_axial + kappa_y * z_i - kappa_z * y_i

断面力（ファイバー積分）:
  N  = Sum(sigma_i * A_i)
  My = Sum(sigma_i * z_i * A_i)
  Mz = -Sum(sigma_i * y_i * A_i)

断面 consistent tangent (3x3):
  C_ss[a,b] = Sum(D_ep_i * B_i[a] * B_i[b] * A_i)
  B_i = [1, z_i, -y_i]

参考文献:
  - Spacone et al. (1996) "Fibre beam-column model for non-linear analysis"
  - de Souza Neto et al. (2008) "Computational Methods for Plasticity", Ch.14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from xkep_cae.core.state import PlasticState1D
from xkep_cae.materials.plasticity_1d import Plasticity1D
from xkep_cae.sections.fiber import FiberSection


class FiberSectionResult(NamedTuple):
    """ファイバー断面積分の結果.

    Attributes:
        N: 軸力
        My: y軸まわり曲げモーメント
        Mz: z軸まわり曲げモーメント
        C_sec: (3, 3) 断面 consistent tangent [[dN/dε, dN/dκy, dN/dκz], ...]
        states_new: 更新されたファイバー塑性状態のリスト
    """

    N: float
    My: float
    Mz: float
    C_sec: np.ndarray
    states_new: list[PlasticState1D]


@dataclass
class FiberIntegrator:
    """ファイバー断面積分器.

    FiberSection の各ファイバーに Plasticity1D の return mapping を適用し、
    断面力と consistent tangent を計算する。

    Attributes:
        section: ファイバー断面
        material: 1D弾塑性構成則
        states: 各ファイバーの塑性状態（n_fibers 個）
    """

    section: FiberSection
    material: Plasticity1D
    states: list[PlasticState1D] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.states:
            self.states = [PlasticState1D() for _ in range(self.section.n_fibers)]
        if len(self.states) != self.section.n_fibers:
            raise ValueError(
                f"states の数がファイバー数と不一致: {len(self.states)} != {self.section.n_fibers}"
            )

    def integrate(
        self,
        eps_axial: float,
        kappa_y: float,
        kappa_z: float,
    ) -> FiberSectionResult:
        """断面変形からファイバー積分で断面力と tangent を計算する.

        Args:
            eps_axial: 軸ひずみ (Gamma_1)
            kappa_y: y軸まわり曲率 (kappa_2)
            kappa_z: z軸まわり曲率 (kappa_3)

        Returns:
            FiberSectionResult: (N, My, Mz, C_sec, states_new)
        """
        sec = self.section
        y = sec.y
        z = sec.z
        areas = sec.areas

        N = 0.0
        My = 0.0
        Mz = 0.0
        C_sec = np.zeros((3, 3), dtype=float)
        states_new: list[PlasticState1D] = []

        for i in range(sec.n_fibers):
            # ファイバーひずみ
            eps_i = eps_axial + kappa_y * z[i] - kappa_z * y[i]

            # return mapping
            result = self.material.return_mapping(eps_i, self.states[i])
            sigma_i = result.stress
            D_ep_i = result.tangent
            states_new.append(result.state_new)

            Ai = areas[i]

            # 断面力
            N += sigma_i * Ai
            My += sigma_i * z[i] * Ai
            Mz += -sigma_i * y[i] * Ai

            # consistent tangent: B_i = [1, z_i, -y_i]
            B_i = np.array([1.0, z[i], -y[i]])
            C_sec += D_ep_i * Ai * np.outer(B_i, B_i)

        return FiberSectionResult(
            N=N,
            My=My,
            Mz=Mz,
            C_sec=C_sec,
            states_new=states_new,
        )

    def update_states(self, new_states: list[PlasticState1D]) -> None:
        """収束後に塑性状態を更新する（commit）."""
        if len(new_states) != self.section.n_fibers:
            raise ValueError(
                f"new_states の数がファイバー数と不一致: "
                f"{len(new_states)} != {self.section.n_fibers}"
            )
        self.states = [s.copy() for s in new_states]

    def copy_states(self) -> list[PlasticState1D]:
        """現在の塑性状態のコピーを返す."""
        return [s.copy() for s in self.states]
