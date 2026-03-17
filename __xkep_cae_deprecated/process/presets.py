"""SolverPreset — 検証済みの Strategy 組み合わせ.

Preset は SolverStrategies の名前付き・検証済み構成を提供する。
設計仕様: phase8-design.md §D

ファクトリパターン:
  Strategy 具象クラスは実行時パラメータ（ndof, beam_E, L_elem 等）を必要とする。
  Preset はデフォルト構成を定義し、create() で具体パラメータを渡して
  SolverStrategies を生成する。内部では data.default_strategies() を活用する。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from __xkep_cae_deprecated.process.data import SolverStrategies


@dataclass(frozen=True)
class SolverPreset:
    """検証済みの Strategy 組み合わせ.

    name: プリセット名（一意）
    factory: SolverStrategies を生成するファクトリ
    verified_by: 検証 status 番号
    description: 用途説明
    default_kwargs: ファクトリへのデフォルト引数
    """

    name: str
    factory: Callable[..., SolverStrategies]
    verified_by: str  # e.g. "status-147"
    description: str = ""
    # frozen dataclass で dict を使うため tuple of pairs で保持
    default_overrides: tuple[tuple[str, object], ...] = ()

    def create(self, **kwargs: object) -> SolverStrategies:
        """Preset の設定 + ユーザー指定パラメータから SolverStrategies を生成.

        Usage:
            strategies = PRESET_SMOOTH_PENALTY.create(ndof=1000, beam_E=200e3)
        """
        merged = dict(self.default_overrides)
        merged.update(kwargs)
        return self.factory(**merged)


def _smooth_penalty_factory(**kwargs: object) -> SolverStrategies:
    """smooth_penalty 構成の SolverStrategies を生成."""
    from __xkep_cae_deprecated.process.data import default_strategies

    defaults = {
        "use_friction": True,
        "contact_mode": "smooth_penalty",
    }
    defaults.update(kwargs)
    return default_strategies(**defaults)


def _ncp_frictionless_factory(**kwargs: object) -> SolverStrategies:
    """NCP frictionless 構成の SolverStrategies を生成."""
    from __xkep_cae_deprecated.process.data import default_strategies

    defaults = {
        "use_friction": False,
        "contact_mode": "ncp",
    }
    defaults.update(kwargs)
    return default_strategies(**defaults)


# --- 組み込みプリセット定義 ---

PRESET_SMOOTH_PENALTY = SolverPreset(
    name="smooth_penalty_quasi_static",
    factory=_smooth_penalty_factory,
    verified_by="status-147",
    description="7本撚線曲げ揺動収束実績あり。基軸構成。",
    default_overrides=(
        ("use_friction", True),
        ("contact_mode", "smooth_penalty"),
    ),
)

PRESET_NCP_FRICTIONLESS = SolverPreset(
    name="ncp_frictionless",
    factory=_ncp_frictionless_factory,
    verified_by="status-112",
    description="NCP法線 + 摩擦なし。基本テスト用。",
    default_overrides=(
        ("use_friction", False),
        ("contact_mode", "ncp"),
    ),
)

PRESETS: dict[str, SolverPreset] = {
    p.name: p for p in [PRESET_SMOOTH_PENALTY, PRESET_NCP_FRICTIONLESS]
}


def get_presets() -> dict[str, SolverPreset]:
    """プリセットレジストリを返す."""
    return PRESETS


def get_preset(name: str) -> SolverPreset:
    """名前でプリセットを取得."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"プリセット '{name}' が見つかりません。利用可能: {available}")
    return PRESETS[name]
