"""Penalty Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import math
import warnings

import pytest

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess
from xkep_cae.process.strategies.penalty import (
    AutoBeamEIProcess,
    AutoEALProcess,
    ContinuationPenaltyProcess,
    ManualPenaltyProcess,
    PenaltyInput,
    create_penalty_strategy,
)
from xkep_cae.process.strategies.protocols import PenaltyStrategy
from xkep_cae.process.testing import binds_to

# --- Protocol 準拠チェック ---


class TestPenaltyProtocolConformance:
    """全 Penalty 具象が PenaltyStrategy Protocol を満たすことを検証."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (AutoBeamEIProcess, {"beam_E": 210e3, "beam_I": 1e-4, "L_elem": 1.0}),
            (AutoEALProcess, {"beam_E": 210e3, "beam_A": 1e-2, "L_elem": 1.0}),
            (ManualPenaltyProcess, {"k_pen": 1e6}),
            (
                ContinuationPenaltyProcess,
                {"k_pen_target": 1e6, "start_fraction": 0.01, "ramp_steps": 5},
            ),
        ],
    )
    def test_protocol_conformance(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, PenaltyStrategy)


# --- AutoBeamEI ---


@binds_to(AutoBeamEIProcess)
class TestAutoBeamEIProcess:
    """AutoBeamEIProcess の単体テスト."""

    def test_basic_computation(self):
        """基本計算: k_pen = scale * 12 * E * I / L^3 / n_pairs."""
        proc = AutoBeamEIProcess(beam_E=210e3, beam_I=1e-4, L_elem=1.0, scale=0.1)
        expected = 0.1 * 12.0 * 210e3 * 1e-4 / 1.0
        assert proc.compute_k_pen(0, 10) == pytest.approx(expected, rel=1e-10)

    def test_sqrt_scaling(self):
        """sqrt スケーリング."""
        proc = AutoBeamEIProcess(
            beam_E=210e3,
            beam_I=1e-4,
            L_elem=1.0,
            n_contact_pairs=100,
            scaling="sqrt",
        )
        expected = 0.1 * 12.0 * 210e3 * 1e-4 / math.sqrt(100)
        assert proc.compute_k_pen(0, 10) == pytest.approx(expected, rel=1e-10)

    def test_linear_scaling(self):
        """linear スケーリング."""
        proc = AutoBeamEIProcess(
            beam_E=210e3,
            beam_I=1e-4,
            L_elem=1.0,
            n_contact_pairs=10,
            scaling="linear",
        )
        expected = 0.1 * 12.0 * 210e3 * 1e-4 / 10.0
        assert proc.compute_k_pen(0, 10) == pytest.approx(expected, rel=1e-10)

    def test_step_independent(self):
        """ステップに依存しない（一定値）."""
        proc = AutoBeamEIProcess(beam_E=210e3, beam_I=1e-4, L_elem=1.0)
        k0 = proc.compute_k_pen(0, 10)
        k5 = proc.compute_k_pen(5, 10)
        assert k0 == k5

    def test_process_method(self):
        """process() メソッドの動作."""
        proc = AutoBeamEIProcess(beam_E=210e3, beam_I=1e-4, L_elem=1.0)
        result = proc.process(PenaltyInput(step=0, total_steps=10))
        assert result.k_pen == proc.compute_k_pen(0, 10)

    def test_tiny_L_elem(self):
        """極小要素長でもゼロ除算しない."""
        proc = AutoBeamEIProcess(beam_E=210e3, beam_I=1e-4, L_elem=0.0)
        assert proc.compute_k_pen(0, 10) > 0

    def test_meta(self):
        assert AutoBeamEIProcess.meta.name == "AutoBeamEI"
        assert AutoBeamEIProcess.meta.module == "solve"
        assert not AutoBeamEIProcess.meta.deprecated


# --- AutoEAL ---


@binds_to(AutoEALProcess)
class TestAutoEALProcess:
    """AutoEALProcess の単体テスト."""

    def test_basic_computation(self):
        """基本計算: k_pen = scale * E * A / L."""
        proc = AutoEALProcess(beam_E=210e3, beam_A=1e-2, L_elem=2.0, scale=1.0)
        expected = 1.0 * 210e3 * 1e-2 / 2.0
        assert proc.compute_k_pen(0, 10) == pytest.approx(expected, rel=1e-10)

    def test_custom_scale(self):
        proc = AutoEALProcess(beam_E=210e3, beam_A=1e-2, L_elem=1.0, scale=0.5)
        expected = 0.5 * 210e3 * 1e-2 / 1.0
        assert proc.compute_k_pen(0, 10) == pytest.approx(expected, rel=1e-10)

    def test_process_method(self):
        proc = AutoEALProcess(beam_E=210e3, beam_A=1e-2, L_elem=1.0)
        result = proc.process(PenaltyInput(step=3, total_steps=10))
        assert result.k_pen > 0


# --- ManualPenalty ---


@binds_to(ManualPenaltyProcess)
class TestManualPenaltyProcess:
    """ManualPenaltyProcess の単体テスト."""

    def test_returns_specified_value(self):
        proc = ManualPenaltyProcess(k_pen=1e6)
        assert proc.compute_k_pen(0, 10) == 1e6
        assert proc.compute_k_pen(9, 10) == 1e6

    def test_deprecated_warning(self):
        """deprecated マーカーが設定されている."""
        assert ManualPenaltyProcess.meta.deprecated
        assert ManualPenaltyProcess.meta.deprecated_by == "AutoBeamEIProcess"

    def test_deprecated_warning_on_use(self):
        """uses に ManualPenaltyProcess を指定した時に DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class _DummyUser(
                SolverProcess[PenaltyInput, PenaltyInput],
            ):
                meta = ProcessMeta(
                    name="_DummyUser", module="test", document_path="../docs/penalty.md"
                )
                uses = [ManualPenaltyProcess]

                def process(self, input_data):
                    return input_data

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1


# --- ContinuationPenalty ---


@binds_to(ContinuationPenaltyProcess)
class TestContinuationPenaltyProcess:
    """ContinuationPenaltyProcess の単体テスト."""

    def test_initial_step(self):
        """step=0 では start_fraction * k_pen_target."""
        proc = ContinuationPenaltyProcess(k_pen_target=1e6, start_fraction=0.01, ramp_steps=5)
        assert proc.compute_k_pen(0, 10) == pytest.approx(0.01 * 1e6, rel=1e-10)

    def test_ramp_midpoint_linear(self):
        """ランプ途中で線形補間（mode=linear）."""
        proc = ContinuationPenaltyProcess(
            k_pen_target=1e6, start_fraction=0.0, ramp_steps=10, mode="linear"
        )
        k5 = proc.compute_k_pen(5, 10)
        assert k5 == pytest.approx(0.5 * 1e6, rel=1e-10)

    def test_ramp_midpoint_geometric(self):
        """ランプ途中で幾何級数（デフォルトmode=geometric）."""
        proc = ContinuationPenaltyProcess(k_pen_target=1e6, start_fraction=0.01, ramp_steps=5)
        # step=0: 0.01*1e6 = 1e4
        # geometric ratio = (1/0.01)^(1/5) = 100^0.2 ≈ 2.5119
        k0 = proc.compute_k_pen(0, 10)
        k1 = proc.compute_k_pen(1, 10)
        assert k0 == pytest.approx(0.01 * 1e6, rel=1e-10)
        assert k1 > k0  # 単調増加
        assert k1 < 1e6  # ターゲット未満

    def test_after_ramp(self):
        """ramp_steps 以降はターゲット値."""
        proc = ContinuationPenaltyProcess(k_pen_target=1e6, start_fraction=0.01, ramp_steps=5)
        assert proc.compute_k_pen(5, 10) == pytest.approx(1e6, rel=1e-10)
        assert proc.compute_k_pen(10, 10) == pytest.approx(1e6, rel=1e-10)

    def test_monotonic_increase(self):
        """ステップごとに単調増加."""
        proc = ContinuationPenaltyProcess(k_pen_target=1e6, start_fraction=0.01, ramp_steps=5)
        values = [proc.compute_k_pen(i, 10) for i in range(10)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_process_method(self):
        proc = ContinuationPenaltyProcess(k_pen_target=1e6)
        result = proc.process(PenaltyInput(step=0, total_steps=10))
        assert result.k_pen > 0


# --- create_penalty_strategy ファクトリ ---


class _MockConfig:
    """テスト用モック ContactConfig."""

    def __init__(self, **kwargs):
        defaults = {
            "beam_E": 210e3,
            "beam_I": 1e-4,
            "beam_A": 1e-2,
            "k_pen_scale": 0.1,
            "k_pen_mode": "beam_ei",
            "k_pen_scaling": "linear",
            "k_pen_continuation": False,
            "k_pen_continuation_start": 0.01,
            "k_pen_continuation_steps": 5,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class _MockManager:
    """テスト用モック ContactManager."""

    def __init__(self, config=None, n_pairs=10):
        self.config = config or _MockConfig()
        self.n_pairs = n_pairs


class TestCreatePenaltyStrategy:
    """create_penalty_strategy ファクトリのテスト."""

    def test_manual_k_pen(self):
        """k_pen > 0 → ManualPenaltyProcess."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            strategy = create_penalty_strategy(k_pen=1e6)
        assert isinstance(strategy, ManualPenaltyProcess)
        assert strategy.compute_k_pen(0, 10) == 1e6

    def test_manual_k_pen_deprecated_warning(self):
        """k_pen > 0 で DeprecationWarning が出る."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_penalty_strategy(k_pen=1e6)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1

    def test_auto_beam_ei(self):
        """beam_E > 0, k_pen_mode='beam_ei' → AutoBeamEIProcess."""
        import numpy as np

        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        conn = np.array([[0, 1], [1, 2]])
        mgr = _MockManager(config=_MockConfig(k_pen_mode="beam_ei"))
        strategy = create_penalty_strategy(manager=mgr, node_coords_ref=coords, connectivity=conn)
        assert isinstance(strategy, AutoBeamEIProcess)
        assert strategy.compute_k_pen(0, 10) > 0

    def test_auto_ea_l(self):
        """beam_E > 0, k_pen_mode='ea_l' → AutoEALProcess."""
        import numpy as np

        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]])
        mgr = _MockManager(config=_MockConfig(k_pen_mode="ea_l"))
        strategy = create_penalty_strategy(manager=mgr, node_coords_ref=coords, connectivity=conn)
        assert isinstance(strategy, AutoEALProcess)

    def test_continuation(self):
        """k_pen_continuation=True → ContinuationPenaltyProcess."""
        import numpy as np

        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]])
        mgr = _MockManager(config=_MockConfig(k_pen_continuation=True))
        strategy = create_penalty_strategy(manager=mgr, node_coords_ref=coords, connectivity=conn)
        assert isinstance(strategy, ContinuationPenaltyProcess)
        # step=0 では start_fraction * target
        k0 = strategy.compute_k_pen(0, 10)
        k_target = strategy.k_pen_target
        assert k0 == pytest.approx(0.01 * k_target, rel=1e-10)

    def test_no_manager(self):
        """manager=None → ManualPenaltyProcess(1.0) フォールバック."""
        strategy = create_penalty_strategy()
        assert isinstance(strategy, ManualPenaltyProcess)
        assert strategy.compute_k_pen(0, 10) == 1.0

    def test_beam_E_zero_fallback(self):
        """beam_E=0 → k_pen_scale フォールバック."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mgr = _MockManager(config=_MockConfig(beam_E=0.0, k_pen_scale=5.0))
            strategy = create_penalty_strategy(manager=mgr)
        assert isinstance(strategy, ManualPenaltyProcess)
        assert strategy.compute_k_pen(0, 10) == 5.0

    def test_k_pen_with_continuation(self):
        """k_pen > 0 + continuation → ContinuationPenaltyProcess."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mgr = _MockManager(config=_MockConfig(k_pen_continuation=True))
            strategy = create_penalty_strategy(k_pen=1e6, manager=mgr)
        assert isinstance(strategy, ContinuationPenaltyProcess)
        assert strategy.k_pen_target == 1e6
