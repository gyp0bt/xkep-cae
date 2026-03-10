"""ContactGeometry Strategy 具象実装.

接触幾何の評価方法を Strategy として実装する。

設計仕様: xkep_cae/process/process-architecture.md §2.4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess


@dataclass(frozen=True)
class ContactGeometryInput:
    """ContactGeometry Strategy の入力."""

    node_coords: np.ndarray
    connectivity: np.ndarray
    radii: np.ndarray | float


@dataclass(frozen=True)
class ContactGeometryOutput:
    """ContactGeometry Strategy の出力."""

    contact_pairs: list


class PointToPointProcess(SolverProcess[ContactGeometryInput, ContactGeometryOutput]):
    """最近接点ペア（Point-to-Point）による接触検出.

    各要素ペアの最近接パラメータ (s, t) を求め、
    ギャップ g = ||x_B - x_A|| - r_A - r_B を評価する。
    基本的な接触検出で、小規模問題に適する。

    制約ヤコビアン:
        ∂g_n/∂u の係数: [(1-s), s, -(1-t), -t]
    """

    meta = ProcessMeta(name="PointToPoint", module="solve", version="0.1.0")

    def __init__(self, *, exclude_same_layer: bool = True) -> None:
        self._exclude_same_layer = exclude_same_layer

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出.

        Phase 3 で ContactManager.update_geometry() から移植。
        現在は Protocol 準拠スタブ。
        """
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算.

        g = ||x_B(t) - x_A(s)|| - r_A - r_B
        """
        if hasattr(pair, "state") and hasattr(pair.state, "gap"):
            return float(pair.state.gap)
        return 0.0

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


class LineToLineGaussProcess(SolverProcess[ContactGeometryInput, ContactGeometryOutput]):
    """Line-to-Line Gauss 積分による接触評価.

    要素ペアの相互作用領域をGauss積分点で離散化し、
    接触力と接触剛性を積分する。大規模問題で精度・安定性が向上。

    パラメータ:
        n_gauss: 1次元あたりの Gauss 点数（デフォルト: 2）
    """

    meta = ProcessMeta(name="LineToLineGauss", module="solve", version="0.1.0")

    def __init__(
        self,
        *,
        n_gauss: int = 2,
        exclude_same_layer: bool = True,
    ) -> None:
        self._n_gauss = n_gauss
        self._exclude_same_layer = exclude_same_layer

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出.

        Phase 3 で line_contact.py の検出ロジックを移植。
        """
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算.

        L2L では各 Gauss 点でギャップを評価し、
        最小ギャップを返す。
        """
        if hasattr(pair, "state") and hasattr(pair.state, "gap"):
            return float(pair.state.gap)
        return 0.0

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


class MortarSegmentProcess(SolverProcess[ContactGeometryInput, ContactGeometryOutput]):
    """Mortar 法セグメントによる接触評価.

    Line-to-Line に加えて mortar 射影を行い、
    接触面の連続性を保証する。大規模問題のロバスト性に寄与。

    前提条件: line_contact=True（L2L との併用が必須）
    """

    meta = ProcessMeta(name="MortarSegment", module="solve", version="0.1.0")

    def __init__(
        self,
        *,
        n_gauss: int = 2,
        exclude_same_layer: bool = True,
    ) -> None:
        self._n_gauss = n_gauss
        self._exclude_same_layer = exclude_same_layer

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出 + mortar ノード同定.

        Phase 3 で mortar.py から移植。
        """
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算（mortar 射影ベース）."""
        if hasattr(pair, "state") and hasattr(pair.state, "gap"):
            return float(pair.state.gap)
        return 0.0

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)
