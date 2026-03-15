"""Coating Strategy サブパッケージ.

被膜接触モデル（Kelvin-Voigt弾性+粘性ダッシュポット）。
旧 xkep_cae_deprecated/process/strategies/coating.py からの移行（status-181）。
"""

from xkep_cae.contact.coating.strategy import (
    KelvinVoigtCoatingProcess,
    NoCoatingProcess,
)

__all__ = [
    "KelvinVoigtCoatingProcess",
    "NoCoatingProcess",
]
