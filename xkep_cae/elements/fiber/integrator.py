"""ファイバー断面積分 Process.

ファイバー離散化された断面に対して、軸ひずみ + 曲率から
断面力 (N, M_y, M_z) と接線行列 C_section を計算する。

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F3
[← README](../../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core.base import AbstractProcess, ProcessMeta
from xkep_cae.elements.fiber.section import CircularFiberSection
from xkep_cae.elements.fiber.state import Fiber1DState, SectionState


@dataclass(frozen=True)
class FiberIntegratorConfig:
    """断面積分の入力.

    Attributes:
        eps0: 軸ひずみ（断面中心）
        kappa_y: y 方向曲率
        kappa_z: z 方向曲率（平面曲げでは 0）
        section: ファイバー断面定義
        material: 1D 材料則（Fiber1DMaterialStrategy 準拠）
        state: 旧セクション状態
    """

    eps0: float
    kappa_y: float
    kappa_z: float
    section: CircularFiberSection
    material: object  # Fiber1DMaterialStrategy
    state: SectionState


@dataclass(frozen=True)
class FiberIntegratorResult:
    """断面積分の出力.

    Attributes:
        N: 軸力
        M_y: y 方向曲げモーメント
        M_z: z 方向曲げモーメント
        C_section: 接線行列 (3x3 or 2x2)
        state_new: 新セクション状態
    """

    N: float
    M_y: float
    M_z: float
    C_section: np.ndarray
    state_new: SectionState


class FiberSectionIntegratorProcess(
    AbstractProcess[FiberIntegratorConfig, FiberIntegratorResult],
):
    """ファイバー断面積分 Process.

    ファイバーループで逐次 material.evaluate() を呼び出し、
    断面力と接線行列を積算する。

    接線行列:
      C_section[i,j] = Σ_k E_t_k * A_k * B_k[i] * B_k[j]
      ただし B_k = [1, -y_k, z_k]

    設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
    """

    meta = ProcessMeta(
        name="FiberSectionIntegrator",
        module="elements",
        version="1.0.0",
        document_path="../docs/fiber_beam_strand.md",
    )
    uses: list = []

    def process(
        self,
        input_data: FiberIntegratorConfig,
    ) -> FiberIntegratorResult:
        """断面積分を実行.

        Args:
            input_data: FiberIntegratorConfig

        Returns:
            FiberIntegratorResult
        """
        cfg = input_data
        sec = cfg.section
        mat = cfg.material
        n = sec.n_fiber

        N = 0.0
        M_y = 0.0
        M_z = 0.0
        C = np.zeros((3, 3))

        fiber_states_new: list[Fiber1DState] = []

        for i in range(n):
            yi = sec.y[i]
            zi = sec.z[i]
            ai = sec.area[i]

            # ファイバーひずみ: eps_i = eps0 - kappa_y * y_i + kappa_z * z_i
            eps_i = cfg.eps0 - cfg.kappa_y * yi + cfg.kappa_z * zi

            sigma_i, E_t_i, state_new_i = mat.evaluate(eps_i, cfg.state.fibers[i])

            # 断面力の積算
            N += sigma_i * ai
            M_y -= sigma_i * yi * ai
            M_z += sigma_i * zi * ai

            # 接線行列の積算: C += E_t * A * B @ B^T
            # B = [1, -y, z]
            b = np.array([1.0, -yi, zi])
            C += E_t_i * ai * np.outer(b, b)

            fiber_states_new.append(state_new_i)

        state_new = SectionState(fibers=tuple(fiber_states_new))

        return FiberIntegratorResult(
            N=N,
            M_y=M_y,
            M_z=M_z,
            C_section=C,
            state_new=state_new,
        )
