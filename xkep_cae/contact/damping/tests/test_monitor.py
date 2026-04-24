"""ContactDampingEnergyMonitorProcess ユニットテスト (status-366 Phase 2).

[← damping README](../__init__.py) | [← project README](../../../../README.md)

E_damp / E_strain 比の集計と budget 超過判定を検証する。
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.contact.damping import (
    ContactDampingEnergyMonitorInput,
    ContactDampingEnergyMonitorOutput,
    ContactDampingEnergyMonitorProcess,
)
from xkep_cae.core.testing import binds_to


@dataclass
class _FakeEntry:
    step: int
    strain_energy: float


@dataclass
class _FakeHistory:
    entries: list


@binds_to(ContactDampingEnergyMonitorProcess)
class TestContactDampingEnergyMonitorAPI:
    """no-op / budget 超過判定 / 出力型の検証."""

    def test_empty_damping_history(self) -> None:
        """damping_energy_history が空なら short-circuit で全ゼロ出力."""
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=(),
            energy_history=_FakeHistory(entries=[]),
            budget_ratio=0.1,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        assert isinstance(out, ContactDampingEnergyMonitorOutput)
        assert out.max_ratio == 0.0
        assert out.final_ratio == 0.0
        assert out.final_damping_energy == 0.0
        assert out.final_strain_energy == 0.0
        assert out.n_violations == 0
        assert out.budget_violated is False

    def test_within_budget_no_violation(self) -> None:
        """比が budget 以下なら violations=0、budget_violated=False."""
        damp_hist = (
            (0.25, 1.0),
            (0.50, 2.0),
            (1.00, 3.0),
        )
        entries = [
            _FakeEntry(step=0, strain_energy=100.0),
            _FakeEntry(step=1, strain_energy=200.0),
            _FakeEntry(step=2, strain_energy=300.0),
        ]
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=damp_hist,
            energy_history=_FakeHistory(entries=entries),
            budget_ratio=0.05,  # 5%
            log_every_n_steps=1,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        # 比は 1/100, 2/200, 3/300 = 0.01 で均一
        assert abs(out.max_ratio - 0.01) < 1e-12
        assert abs(out.final_ratio - 0.01) < 1e-12
        assert out.final_damping_energy == 3.0
        assert out.final_strain_energy == 300.0
        assert out.n_violations == 0
        assert not out.budget_violated

    def test_budget_violation_flags_correctly(self) -> None:
        """比が budget 超過するインクリメントを n_violations にカウント."""
        damp_hist = (
            (0.5, 10.0),
            (1.0, 50.0),
        )
        entries = [
            _FakeEntry(step=0, strain_energy=100.0),
            _FakeEntry(step=1, strain_energy=100.0),
        ]
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=damp_hist,
            energy_history=_FakeHistory(entries=entries),
            budget_ratio=0.20,  # 20% 上限
            log_every_n_steps=0,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        # 比 0.10 / 0.50 → 2 件目のみ超過
        assert abs(out.max_ratio - 0.50) < 1e-12
        assert abs(out.final_ratio - 0.50) < 1e-12
        assert out.n_violations == 1
        assert out.budget_violated is True

    def test_budget_zero_disables_check(self) -> None:
        """budget_ratio=0 ならチェック無効化で n_violations=0 固定."""
        damp_hist = ((1.0, 9999.0),)
        entries = [_FakeEntry(step=0, strain_energy=1.0)]
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=damp_hist,
            energy_history=_FakeHistory(entries=entries),
            budget_ratio=0.0,
            log_every_n_steps=0,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        # 比は 9999/1 でも budget=0 → violations=0
        assert out.max_ratio > 1000.0
        assert out.n_violations == 0
        assert not out.budget_violated

    def test_report_is_str(self) -> None:
        """report フィールドは空でない文字列."""
        damp_hist = ((1.0, 1.0),)
        entries = [_FakeEntry(step=0, strain_energy=10.0)]
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=damp_hist,
            energy_history=_FakeHistory(entries=entries),
            budget_ratio=0.2,
            log_every_n_steps=1,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        assert isinstance(out.report, str)
        assert "DampingMonitor" in out.report

    def test_strain_fallback_when_shorter(self) -> None:
        """energy_history が damping_energy_history より短い場合は末尾を使う."""
        damp_hist = (
            (0.5, 1.0),
            (0.75, 2.0),
            (1.0, 3.0),
        )
        # 2 エントリのみ（最後の strain=50 を padding に使用）
        entries = [
            _FakeEntry(step=0, strain_energy=100.0),
            _FakeEntry(step=1, strain_energy=50.0),
        ]
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=damp_hist,
            energy_history=_FakeHistory(entries=entries),
            budget_ratio=0.10,
            log_every_n_steps=0,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        # idx 0: 1/100=0.01, idx 1: 2/50=0.04, idx 2 は 50 で padding → 3/50=0.06
        assert abs(out.max_ratio - 0.06) < 1e-12
        assert abs(out.final_ratio - 0.06) < 1e-12
        assert out.n_violations == 0  # budget=0.10 なので全件 under

    def test_output_types(self) -> None:
        inp = ContactDampingEnergyMonitorInput(
            damping_energy_history=((1.0, 1.0),),
            energy_history=_FakeHistory(entries=[_FakeEntry(step=0, strain_energy=10.0)]),
            budget_ratio=0.0,
        )
        out = ContactDampingEnergyMonitorProcess().process(inp)
        assert isinstance(out.max_ratio, float)
        assert isinstance(out.final_ratio, float)
        assert isinstance(out.final_damping_energy, float)
        assert isinstance(out.final_strain_energy, float)
        assert isinstance(out.n_violations, int)
        assert isinstance(out.budget_violated, bool)
        assert isinstance(out.report, str)
