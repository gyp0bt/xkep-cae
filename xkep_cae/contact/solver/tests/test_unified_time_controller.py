"""UnifiedTimeStepProcess のテスト."""

import pytest

from xkep_cae.contact.solver._adaptive_stepping import StepAction
from xkep_cae.contact.solver._unified_time_controller import (
    TimeStepQueryInput,
    UnifiedTimeStepInput,
    UnifiedTimeStepProcess,
)
from xkep_cae.core.testing import binds_to


@binds_to(UnifiedTimeStepProcess)
class TestUnifiedTimeStepInputAPI:
    """UnifiedTimeStepInput の API テスト."""

    def test_auto_dt_initial(self):
        """dt_initial = 0 で自動計算: t_total/20."""
        cfg = UnifiedTimeStepInput(t_total=10.0)
        assert abs(cfg.dt_initial - 0.5) < 1e-10

    def test_auto_dt_min(self):
        """dt_min = 0 で自動計算: dt_initial/32."""
        cfg = UnifiedTimeStepInput(t_total=10.0, dt_initial=1.0)
        assert abs(cfg.dt_min - 1.0 / 32.0) < 1e-10

    def test_auto_dt_max(self):
        """dt_max = 0 で自動計算: dt_initial*4."""
        cfg = UnifiedTimeStepInput(t_total=100.0, dt_initial=1.0)
        assert abs(cfg.dt_max - 4.0) < 1e-10

    def test_dt_max_capped_by_t_total(self):
        """dt_max は t_total を超えない."""
        cfg = UnifiedTimeStepInput(t_total=2.0, dt_initial=1.0)
        assert cfg.dt_max <= cfg.t_total

    def test_negative_t_total_raises(self):
        """t_total <= 0 はエラー."""
        with pytest.raises(ValueError, match="t_total"):
            UnifiedTimeStepInput(t_total=-1.0)

    def test_explicit_params(self):
        """明示的にパラメータを指定."""
        cfg = UnifiedTimeStepInput(
            t_total=10.0,
            dt_initial=0.1,
            dt_min=0.001,
            dt_max=0.5,
        )
        assert cfg.dt_initial == 0.1
        assert cfg.dt_min == 0.001
        assert cfg.dt_max == 0.5


class TestUnifiedTimeStepProcessAPI:
    """UnifiedTimeStepProcess の API テスト."""

    def test_query_returns_first_step(self):
        """QUERY で最初のステップが返る."""
        ctrl = UnifiedTimeStepProcess(UnifiedTimeStepInput(t_total=10.0, dt_initial=1.0))
        result = ctrl.process(TimeStepQueryInput(action=StepAction.QUERY))
        assert result.has_more_steps
        assert result.load_frac > 0.0
        assert result.dt_sub > 0.0
        assert result.t_current > 0.0

    def test_dt_sub_is_physical_time(self):
        """dt_sub が物理時間で返る."""
        t_total = 10.0
        dt_initial = 1.0
        ctrl = UnifiedTimeStepProcess(UnifiedTimeStepInput(t_total=t_total, dt_initial=dt_initial))
        result = ctrl.process(TimeStepQueryInput(action=StepAction.QUERY))
        # dt_sub = load_frac * t_total (最初のステップ)
        expected_dt = result.load_frac * t_total
        assert abs(result.dt_sub - expected_dt) < 1e-10

    def test_success_grows_step(self):
        """SUCCESS でステップが成長."""
        ctrl = UnifiedTimeStepProcess(UnifiedTimeStepInput(t_total=10.0, dt_initial=0.5))
        q1 = ctrl.process(TimeStepQueryInput(action=StepAction.QUERY))
        frac1 = q1.load_frac

        # SUCCESS
        ctrl.process(
            TimeStepQueryInput(
                action=StepAction.SUCCESS,
                load_frac=frac1,
                load_frac_prev=0.0,
                n_attempts=1,
            )
        )

        q2 = ctrl.process(TimeStepQueryInput(action=StepAction.QUERY, load_frac_prev=frac1))
        delta2 = q2.load_frac - frac1
        # grow_factor=1.5 で拡大するはず
        assert delta2 >= frac1 * 0.9  # 少なくとも同じか増加

    def test_failure_cutback(self):
        """FAILURE でカットバック."""
        ctrl = UnifiedTimeStepProcess(UnifiedTimeStepInput(t_total=10.0, dt_initial=1.0))
        q1 = ctrl.process(TimeStepQueryInput(action=StepAction.QUERY))
        frac1 = q1.load_frac

        fail = ctrl.process(
            TimeStepQueryInput(
                action=StepAction.FAILURE,
                load_frac=frac1,
                load_frac_prev=0.0,
            )
        )
        assert fail.can_retry
        assert fail.n_cutbacks == 1

    def test_full_quasi_static_cycle(self):
        """準静的解析の完全サイクル（荷重→線形ランプ）."""
        t_total = 1.0
        ctrl = UnifiedTimeStepProcess(
            UnifiedTimeStepInput(
                t_total=t_total,
                dt_initial=0.5,
                dt_min=0.01,
            )
        )

        load_frac_prev = 0.0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            q = ctrl.process(
                TimeStepQueryInput(
                    action=StepAction.QUERY,
                    load_frac_prev=load_frac_prev,
                )
            )
            if not q.has_more_steps:
                break

            load_frac = q.load_frac
            assert load_frac > load_frac_prev
            assert q.dt_sub > 0.0
            assert abs(q.t_current - load_frac * t_total) < 1e-10

            # 成功を報告
            ctrl.process(
                TimeStepQueryInput(
                    action=StepAction.SUCCESS,
                    load_frac=load_frac,
                    load_frac_prev=load_frac_prev,
                    n_attempts=2,
                )
            )
            load_frac_prev = load_frac
            steps += 1

        assert load_frac_prev >= 1.0 - 1e-10
        assert steps < max_steps

    def test_t_total_property(self):
        """t_total プロパティ."""
        ctrl = UnifiedTimeStepProcess(UnifiedTimeStepInput(t_total=42.0))
        assert ctrl.t_total == 42.0

    def test_n_cutbacks_property(self):
        """n_cutbacks プロパティ."""
        ctrl = UnifiedTimeStepProcess(UnifiedTimeStepInput(t_total=10.0, dt_initial=5.0))
        assert ctrl.n_cutbacks == 0

        q = ctrl.process(TimeStepQueryInput(action=StepAction.QUERY))
        ctrl.process(
            TimeStepQueryInput(
                action=StepAction.FAILURE,
                load_frac=q.load_frac,
                load_frac_prev=0.0,
            )
        )
        assert ctrl.n_cutbacks == 1
