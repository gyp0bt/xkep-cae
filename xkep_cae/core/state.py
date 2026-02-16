"""状態変数（履歴変数）の管理.

弾塑性構成則の return mapping で更新される内部変数を保持する。
要素ごと・積分点ごとに独立のインスタンスを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlasticState1D:
    """1D弾塑性の状態変数.

    各積分点での塑性履歴を保持する。

    Attributes:
        eps_p: 塑性歪み
        alpha: 累積塑性歪み（等方硬化内部変数）
        beta: 背応力（移動硬化 Armstrong-Frederick）
    """

    eps_p: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    def copy(self) -> PlasticState1D:
        """深いコピーを返す."""
        return PlasticState1D(
            eps_p=self.eps_p,
            alpha=self.alpha,
            beta=self.beta,
        )


@dataclass
class CosseratPlasticState:
    """Cosserat rod 1要素の弾塑性状態.

    6成分の一般化歪みのうち、弾塑性が有効な成分にのみ
    PlasticState1D を保持する。Phase 4.1 では axial (Gamma_1) のみ。

    Attributes:
        axial: 軸方向 (Gamma_1) の塑性状態
    """

    axial: PlasticState1D = field(default_factory=PlasticState1D)

    def copy(self) -> CosseratPlasticState:
        """深いコピーを返す."""
        return CosseratPlasticState(axial=self.axial.copy())


@dataclass
class CosseratFiberPlasticState:
    """Cosserat rod 1積分点のファイバーモデル弾塑性状態.

    各ファイバーに独立な PlasticState1D を保持する。
    ファイバーモデルでは軸方向 + 曲げによる非一様ひずみ分布を追跡する。

    ファイバー i のひずみ:
      epsilon_i = Gamma_1 + kappa_2 * z_i - kappa_3 * y_i

    Attributes:
        fiber_states: 各ファイバーの塑性状態リスト
    """

    fiber_states: list[PlasticState1D] = field(default_factory=list)

    @classmethod
    def create(cls, n_fibers: int) -> CosseratFiberPlasticState:
        """指定数のファイバーを持つ初期状態を生成する."""
        return cls(fiber_states=[PlasticState1D() for _ in range(n_fibers)])

    def copy(self) -> CosseratFiberPlasticState:
        """深いコピーを返す."""
        return CosseratFiberPlasticState(
            fiber_states=[s.copy() for s in self.fiber_states],
        )
