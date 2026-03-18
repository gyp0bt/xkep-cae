"""Coating Strategy サブパッケージ.

被膜接触モデル（Kelvin-Voigt弾性+粘性ダッシュポット）。
"""

from xkep_cae.contact.coating.strategy import (
    KelvinVoigtCoatingProcess,
    NoCoatingProcess,
)

__all__ = [
    "KelvinVoigtCoatingProcess",
    "NoCoatingProcess",
]
