"""統一時間増分コントローラ Process.

適応時間増分と適応荷重制御を統合した Process。
準静的/動的を問わず、物理時間ベースの統一インタフェースで
ステップ制御を行う。

準静的解析:
  荷重を自動で時間の線形増加関数に変換:
    F(t) = t / t_total × F_max
  内部では dt ベースの自動時間増分制御で解く。

動的解析:
  物理時間 dt_physical をそのまま使用。
  Generalized-α 時間積分と連携。

統一インタフェース:
  load_frac = t / t_total（0〜1）
  dt_sub = 物理時間増分（準静的: 擬似時間, 動的: 物理時間）
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.contact.solver._adaptive_stepping import (
    AdaptiveStepInput,
    AdaptiveStepOutput,
    AdaptiveSteppingInput,
    AdaptiveSteppingProcess,
    StepAction,
)
from xkep_cae.core import ProcessMeta, SolverProcess

_setattr = object.__setattr__


@dataclass(frozen=True)
class UnifiedTimeStepInput:
    """統一時間増分コントローラの設定.

    物理時間ベースのパラメータで指定する。
    内部で load_frac ベースに変換して AdaptiveSteppingProcess に委譲。

    Args:
        t_total: 総解析時間 [s]（準静的: 擬似総時間, 動的: 物理総時間）
        dt_initial: 初期時間増分 [s]（0 = t_total/20 で自動計算）
        dt_min: 最小時間増分 [s]（0 = dt_initial/32 で自動計算）
        dt_max: 最大時間増分 [s]（0 = dt_initial*4 で自動計算）
        dt_grow_factor: 成功時の増分拡大係数
        dt_shrink_factor: 失敗時の増分縮小係数
        dt_grow_attempt_threshold: 増分拡大の NR 反復数閾値
        dt_shrink_attempt_threshold: 増分縮小の NR 反復数閾値
        dt_contact_change_threshold: 接触状態変化率閾値
    """

    t_total: float
    dt_initial: float = 0.0
    dt_min: float = 0.0
    dt_max: float = 0.0
    dt_grow_factor: float = 1.5
    dt_shrink_factor: float = 0.5
    dt_grow_attempt_threshold: int = 5
    dt_shrink_attempt_threshold: int = 15
    dt_contact_change_threshold: float = 0.3

    def __post_init__(self) -> None:
        """自動パラメータ計算."""
        if self.t_total <= 0.0:
            raise ValueError(f"t_total は正の値が必要: {self.t_total}")
        if self.dt_initial <= 0.0:
            _setattr(self, "dt_initial", self.t_total / 20.0)
        if self.dt_min <= 0.0:
            _setattr(self, "dt_min", self.dt_initial / 32.0)
        if self.dt_max <= 0.0:
            _setattr(self, "dt_max", min(self.dt_initial * 4.0, self.t_total))


@dataclass(frozen=True)
class TimeStepQueryInput:
    """時間増分コントローラへの問い合わせ."""

    action: StepAction
    load_frac: float = 0.0
    load_frac_prev: float = 0.0
    n_attempts: int = 0
    n_active: int = 0
    prev_n_active: int = 0
    diverged: bool = False


@dataclass(frozen=True)
class TimeStepResultOutput:
    """時間増分コントローラの応答."""

    load_frac: float = 0.0
    dt_sub: float = 0.0
    t_current: float = 0.0
    has_more_steps: bool = True
    can_retry: bool = False
    n_cutbacks: int = 0


class UnifiedTimeStepProcess(
    SolverProcess[TimeStepQueryInput, TimeStepResultOutput],
):
    """統一時間増分コントローラ.

    物理時間ベースのインタフェースで、準静的/動的を問わず
    適応時間増分制御を提供する。

    内部で AdaptiveSteppingProcess に委譲し、物理時間と
    load_frac の変換を行う。

    使用方法:
        controller = UnifiedTimeStepProcess(config)

        # ステップループ
        while True:
            result = controller.process(TimeStepQueryInput(action=QUERY))
            if not result.has_more_steps:
                break
            # ... ソルバー実行 ...
            controller.process(TimeStepQueryInput(action=SUCCESS, ...))

    準静的での荷重変換:
        F(t) = load_frac × F_max = (t / t_total) × F_max
    """

    meta = ProcessMeta(
        name="UnifiedTimeStepProcess",
        module="solve",
        version="1.0.0",
        document_path="docs/adaptive_stepping.md",
    )
    uses = [AdaptiveSteppingProcess]

    def __init__(self, config: UnifiedTimeStepInput) -> None:
        self._config = config
        self._t_total = config.t_total

        # 物理時間を load_frac に変換
        dt_initial_frac = config.dt_initial / config.t_total
        dt_min_frac = config.dt_min / config.t_total
        dt_max_frac = config.dt_max / config.t_total

        # 内部の AdaptiveSteppingProcess に委譲
        self._adaptivesteppingprocess = AdaptiveSteppingProcess(
            AdaptiveSteppingInput(
                dt_initial_fraction=dt_initial_frac,
                dt_grow_factor=config.dt_grow_factor,
                dt_shrink_factor=config.dt_shrink_factor,
                dt_grow_attempt_threshold=config.dt_grow_attempt_threshold,
                dt_shrink_attempt_threshold=config.dt_shrink_attempt_threshold,
                dt_contact_change_threshold=config.dt_contact_change_threshold,
                dt_min_fraction=dt_min_frac,
                dt_max_fraction=dt_max_frac,
            )
        )
        self._last_load_frac_prev = 0.0

    def process(self, input_data: TimeStepQueryInput) -> TimeStepResultOutput:
        """統一時間増分制御の1判定."""
        out = self._adaptivesteppingprocess.process(
            AdaptiveStepInput(
                action=input_data.action,
                load_frac=input_data.load_frac,
                load_frac_prev=input_data.load_frac_prev,
                n_attempts=input_data.n_attempts,
                n_active=input_data.n_active,
                prev_n_active=input_data.prev_n_active,
                diverged=input_data.diverged,
            )
        )
        return self._convert(out, input_data)

    def _convert(
        self,
        out: AdaptiveStepOutput,
        query: TimeStepQueryInput,
    ) -> TimeStepResultOutput:
        """AdaptiveStepOutput を物理時間ベースに変換."""
        load_frac = out.next_load_frac
        t_current = load_frac * self._t_total

        # dt_sub = (load_frac - load_frac_prev) * t_total
        load_frac_prev = query.load_frac_prev
        if query.action == StepAction.QUERY:
            load_frac_prev = self._last_load_frac_prev
        dt_sub = (load_frac - load_frac_prev) * self._t_total

        if query.action == StepAction.QUERY and out.has_more_steps:
            self._last_load_frac_prev = load_frac_prev

        return TimeStepResultOutput(
            load_frac=load_frac,
            dt_sub=max(dt_sub, 0.0),
            t_current=t_current,
            has_more_steps=out.has_more_steps,
            can_retry=out.can_retry,
            n_cutbacks=out.n_cutbacks,
        )

    @property
    def t_total(self) -> float:
        """総解析時間."""
        return self._t_total

    @property
    def n_cutbacks(self) -> int:
        """累積カットバック回数."""
        return self._adaptivesteppingprocess._n_cutbacks
