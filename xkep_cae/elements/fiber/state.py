"""ファイバー1D材料の状態 dataclass.

全ての履歴変数は frozen dataclass として管理し、C17 準拠（no mutation）。
材料の evaluate() は新しい状態を返し、呼び出し側が明示的に保持する。

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
[← README](../../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fiber1DState:
    """1ファイバー点の履歴変数.

    Attributes:
        eps_p: 塑性ひずみ（BilinearKinematicHardening1D 用）
        alpha: 背応力（移動硬化の中心シフト）
        slip: 摩擦層スリップ変位（MultiLayerFrictionDegrading1D 用、Phase F2）
        slipped: 摩擦層スリップ履歴フラグ（Phase F2）
    """

    eps_p: float = 0.0
    alpha: float = 0.0
    slip: tuple[float, ...] = ()
    slipped: tuple[bool, ...] = ()


@dataclass(frozen=True)
class SectionState:
    """断面全ファイバーの履歴変数.

    Attributes:
        fibers: 各ファイバー点の状態タプル（len = n_fiber）
    """

    fibers: tuple[Fiber1DState, ...] = ()
