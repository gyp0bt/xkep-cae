"""StepEnergyDiagnosticsProcess のテスト."""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.solver._energy_diagnostics import (
    EnergyHistory,
    EnergyHistoryEntry,
    StepEnergyDiagnosticsProcess,
    StepEnergyInput,
)
from xkep_cae.core.testing import binds_to


@binds_to(StepEnergyDiagnosticsProcess)
class TestStepEnergyDiagnosticsAPI:
    """StepEnergyDiagnosticsProcess の API テスト."""

    def test_zero_state_returns_zero_energy(self):
        """ゼロ状態ではエネルギーがゼロ."""
        ndof = 12
        proc = StepEnergyDiagnosticsProcess()
        result = proc.process(
            StepEnergyInput(
                u=np.zeros(ndof),
                velocity=np.zeros(ndof),
                mass_matrix=sp.eye(ndof),
                f_int=np.zeros(ndof),
                f_ext=np.zeros(ndof),
                f_c=np.zeros(ndof),
                dt=0.01,
                step=0,
            )
        )
        assert result.kinetic_energy == 0.0
        assert result.strain_energy == 0.0
        assert result.external_work == 0.0
        assert result.contact_work == 0.0
        assert result.total_energy == 0.0

    def test_kinetic_energy_positive(self):
        """速度ありで運動エネルギーが正."""
        ndof = 6
        M = sp.diags([2.0] * ndof)
        v = np.ones(ndof)
        proc = StepEnergyDiagnosticsProcess()
        result = proc.process(
            StepEnergyInput(
                u=np.zeros(ndof),
                velocity=v,
                mass_matrix=M,
                f_int=np.zeros(ndof),
                f_ext=np.zeros(ndof),
                f_c=np.zeros(ndof),
                dt=0.01,
                step=1,
            )
        )
        # KE = 0.5 * v^T M v = 0.5 * 6 * 2.0 * 1.0² = 6.0
        assert abs(result.kinetic_energy - 6.0) < 1e-10

    def test_strain_energy_positive(self):
        """変位+内力ありでひずみエネルギーが正."""
        ndof = 6
        u = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        f_int = np.array([10.0, 20.0, 0.0, 0.0, 0.0, 0.0])
        proc = StepEnergyDiagnosticsProcess()
        result = proc.process(
            StepEnergyInput(
                u=u,
                velocity=np.zeros(ndof),
                mass_matrix=sp.eye(ndof),
                f_int=f_int,
                f_ext=np.zeros(ndof),
                f_c=np.zeros(ndof),
                dt=0.01,
                step=1,
            )
        )
        # SE = 0.5 * (1*10 + 2*20) = 0.5 * 50 = 25.0
        assert abs(result.strain_energy - 25.0) < 1e-10

    def test_dense_mass_matrix(self):
        """密行列の質量行列でも動作する."""
        ndof = 4
        M = 3.0 * np.eye(ndof)
        v = np.array([1.0, 0.0, 0.0, 0.0])
        proc = StepEnergyDiagnosticsProcess()
        result = proc.process(
            StepEnergyInput(
                u=np.zeros(ndof),
                velocity=v,
                mass_matrix=M,
                f_int=np.zeros(ndof),
                f_ext=np.zeros(ndof),
                f_c=np.zeros(ndof),
                dt=0.01,
                step=0,
            )
        )
        assert abs(result.kinetic_energy - 1.5) < 1e-10


class TestEnergyHistoryAPI:
    """EnergyHistory の API テスト."""

    def test_empty_history_decay_ratio(self):
        """空の履歴では decay_ratio = 1.0."""
        eh = EnergyHistory()
        assert eh.decay_ratio == 1.0

    def test_single_entry_decay_ratio(self):
        """1エントリでは decay_ratio = 1.0."""
        eh = EnergyHistory()
        eh.append(
            EnergyHistoryEntry(
                step=0,
                time=0.0,
                kinetic_energy=10.0,
                strain_energy=0.0,
                external_work=0.0,
                contact_work=0.0,
                total_energy=10.0,
                energy_ratio=1.0,
            )
        )
        assert eh.decay_ratio == 1.0

    def test_decay_ratio_decreasing(self):
        """減衰するエネルギー → decay_ratio < 1.0."""
        eh = EnergyHistory()
        eh.append(
            EnergyHistoryEntry(
                step=0,
                time=0.0,
                kinetic_energy=10.0,
                strain_energy=0.0,
                external_work=0.0,
                contact_work=0.0,
                total_energy=10.0,
                energy_ratio=1.0,
            )
        )
        eh.append(
            EnergyHistoryEntry(
                step=1,
                time=0.1,
                kinetic_energy=3.0,
                strain_energy=5.0,
                external_work=0.0,
                contact_work=0.0,
                total_energy=8.0,
                energy_ratio=0.8,
            )
        )
        assert abs(eh.decay_ratio - 0.8) < 1e-10

    def test_summary_not_empty(self):
        """サマリが空文字でない."""
        eh = EnergyHistory()
        eh.append(
            EnergyHistoryEntry(
                step=0,
                time=0.0,
                kinetic_energy=1.0,
                strain_energy=2.0,
                external_work=3.0,
                contact_work=0.0,
                total_energy=3.0,
                energy_ratio=1.0,
            )
        )
        assert len(eh.summary()) > 0
        assert "エネルギー収支サマリ" in eh.summary()


class TestDivergenceDetectionIntegration:
    """発散検知の統合テスト（NewtonUzawaDynamicInput）."""

    def test_divergence_window_default(self):
        """divergence_window のデフォルト値が5."""
        from xkep_cae.contact.solver._newton_uzawa_dynamic import (
            NewtonUzawaDynamicInput,
        )

        cfg = NewtonUzawaDynamicInput()
        assert cfg.divergence_window == 5


class TestAdaptiveSteppingCutbackTracking:
    """AdaptiveStepping のカットバック追跡テスト."""

    def test_cutback_count_increments(self):
        """FAILURE でカットバック数が増加."""
        from xkep_cae.contact.solver._adaptive_stepping import (
            AdaptiveStepInput,
            AdaptiveSteppingInput,
            AdaptiveSteppingProcess,
            StepAction,
        )

        config = AdaptiveSteppingInput(dt_initial_fraction=0.5)
        proc = AdaptiveSteppingProcess(config)

        # QUERY → get first step
        q = proc.process(AdaptiveStepInput(action=StepAction.QUERY))
        assert q.has_more_steps

        # FAILURE
        f = proc.process(
            AdaptiveStepInput(
                action=StepAction.FAILURE,
                load_frac=q.next_load_frac,
                load_frac_prev=0.0,
            )
        )
        assert f.can_retry
        assert f.n_cutbacks == 1

    def test_diverged_flag_aggressive_shrink(self):
        """diverged=True でより積極的な縮小."""
        from xkep_cae.contact.solver._adaptive_stepping import (
            AdaptiveStepInput,
            AdaptiveSteppingInput,
            AdaptiveSteppingProcess,
            StepAction,
        )

        config = AdaptiveSteppingInput(
            dt_initial_fraction=0.5,
            dt_shrink_factor=0.5,
        )
        proc_normal = AdaptiveSteppingProcess(config)
        proc_diverge = AdaptiveSteppingProcess(config)

        # Normal failure
        q = proc_normal.process(AdaptiveStepInput(action=StepAction.QUERY))
        proc_normal.process(
            AdaptiveStepInput(
                action=StepAction.FAILURE,
                load_frac=q.next_load_frac,
                load_frac_prev=0.0,
                diverged=False,
            )
        )

        # Divergence failure
        q2 = proc_diverge.process(AdaptiveStepInput(action=StepAction.QUERY))
        proc_diverge.process(
            AdaptiveStepInput(
                action=StepAction.FAILURE,
                load_frac=q2.next_load_frac,
                load_frac_prev=0.0,
                diverged=True,
            )
        )

        # diverged ケースの次ステップは normal より小さい
        q_normal_next = proc_normal.process(AdaptiveStepInput(action=StepAction.QUERY))
        q_diverge_next = proc_diverge.process(AdaptiveStepInput(action=StepAction.QUERY))
        assert q_diverge_next.next_load_frac < q_normal_next.next_load_frac


class TestSolverResultDataExtension:
    """SolverResultData の拡張テスト."""

    def test_energy_history_field_exists(self):
        """energy_history フィールドが存在."""
        from xkep_cae.core.data import SolverResultData

        result = SolverResultData(
            u=np.zeros(6),
            converged=True,
            n_increments=1,
            total_attempts=1,
        )
        assert result.energy_history is None
        assert result.n_cutbacks == 0

    def test_energy_history_field_settable(self):
        """energy_history にオブジェクトを設定可能."""
        from xkep_cae.core.data import SolverResultData

        eh = EnergyHistory()
        result = SolverResultData(
            u=np.zeros(6),
            converged=True,
            n_increments=1,
            total_attempts=1,
            energy_history=eh,
            n_cutbacks=3,
        )
        assert result.energy_history is eh
        assert result.n_cutbacks == 3


class TestConvergenceDiagnosticsExtension:
    """ConvergenceDiagnosticsOutput のエネルギー項テスト."""

    def test_energy_fields_default(self):
        """エネルギー項のデフォルト値."""
        from xkep_cae.contact.solver._diagnostics import (
            ConvergenceDiagnosticsOutput,
        )

        diag = ConvergenceDiagnosticsOutput()
        assert diag.kinetic_energy == 0.0
        assert diag.strain_energy == 0.0
        assert diag.total_energy == 0.0
        assert diag.energy_ratio == 1.0

    def test_energy_fields_settable(self):
        """エネルギー項が設定可能."""
        from xkep_cae.contact.solver._diagnostics import (
            ConvergenceDiagnosticsOutput,
        )

        diag = ConvergenceDiagnosticsOutput(
            kinetic_energy=5.0,
            strain_energy=3.0,
            total_energy=8.0,
            energy_ratio=0.9,
        )
        assert diag.kinetic_energy == 5.0
        assert diag.total_energy == 8.0
