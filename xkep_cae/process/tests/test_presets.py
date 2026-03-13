"""SolverPreset テスト.

設計仕様: phase8-design.md §D
"""

from __future__ import annotations

import pytest

from xkep_cae.process.data import SolverStrategies
from xkep_cae.process.presets import (
    PRESET_NCP_FRICTIONLESS,
    PRESET_SMOOTH_PENALTY,
    SolverPreset,
    get_preset,
    get_presets,
)


class TestSolverPreset:
    def test_smooth_penalty_preset_exists(self):
        presets = get_presets()
        assert "smooth_penalty_quasi_static" in presets

    def test_ncp_frictionless_preset_exists(self):
        presets = get_presets()
        assert "ncp_frictionless" in presets

    def test_preset_fields(self):
        preset = get_preset("smooth_penalty_quasi_static")
        assert preset.name == "smooth_penalty_quasi_static"
        assert preset.verified_by == "status-147"
        assert preset.description != ""
        assert callable(preset.factory)

    def test_smooth_penalty_create(self):
        strategies = PRESET_SMOOTH_PENALTY.create()
        assert isinstance(strategies, SolverStrategies)
        assert strategies.penalty is not None
        assert strategies.friction is not None
        assert strategies.time_integration is not None

    def test_ncp_frictionless_create(self):
        strategies = PRESET_NCP_FRICTIONLESS.create()
        assert isinstance(strategies, SolverStrategies)
        assert strategies.penalty is not None
        assert strategies.friction is not None
        assert strategies.time_integration is not None

    def test_get_preset_not_found(self):
        with pytest.raises(KeyError, match="見つかりません"):
            get_preset("nonexistent")

    def test_create_with_override(self):
        strategies = PRESET_SMOOTH_PENALTY.create(mu=0.3)
        assert isinstance(strategies, SolverStrategies)

    def test_preset_frozen(self):
        preset = get_preset("smooth_penalty_quasi_static")
        with pytest.raises(AttributeError):
            preset.name = "changed"

    def test_presets_unique_names(self):
        presets = get_presets()
        names = list(presets.keys())
        assert len(names) == len(set(names))

    def test_custom_preset(self):
        def custom_factory(**kwargs):
            from xkep_cae.process.data import default_strategies

            return default_strategies(**kwargs)

        preset = SolverPreset(
            name="custom",
            factory=custom_factory,
            verified_by="test",
            description="テスト用カスタムプリセット",
        )
        strategies = preset.create()
        assert isinstance(strategies, SolverStrategies)
