"""適応荷重増分制御（プライベート）.

AdaptiveSteppingProcess を新パッケージに移植。
__xkep_cae_deprecated/process/strategies/adaptive_stepping.py からのコピー。
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass

from xkep_cae.core import ProcessMeta, SolverProcess

_setattr = object.__setattr__


@dataclass(frozen=True)
class AdaptiveSteppingInput:
    """適応荷重増分の設定."""

    dt_initial_fraction: float = 0.0
    dt_grow_factor: float = 1.5
    dt_shrink_factor: float = 0.5
    dt_grow_attempt_threshold: int = 5
    dt_shrink_attempt_threshold: int = 15
    dt_contact_change_threshold: float = 0.3
    dt_min_fraction: float = 0.0
    dt_max_fraction: float = 0.0

    def __post_init__(self) -> None:
        """dt_min/max のデフォルト自動計算."""
        effective_n = max(
            1,
            int(1.0 / self.dt_initial_fraction) if self.dt_initial_fraction > 0 else 1,
        )
        if self.dt_min_fraction <= 0.0:
            _setattr(self, "dt_min_fraction", 1.0 / (effective_n * 32))
        if self.dt_max_fraction <= 0.0:
            _setattr(self, "dt_max_fraction", min(8.0 / effective_n, 1.0))


class StepAction(enum.Enum):
    """適応荷重増分の操作種別."""

    QUERY = "query"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class AdaptiveStepInput:
    """適応荷重増分の入力（前ステップの結果）."""

    action: StepAction
    load_frac: float = 0.0
    load_frac_prev: float = 0.0
    n_attempts: int = 0
    n_active: int = 0
    prev_n_active: int = 0
    diverged: bool = False  # 発散検知フラグ（早期カットバック用）


@dataclass(frozen=True)
class AdaptiveStepOutput:
    """適応荷重増分の出力（次ステップの判定）."""

    next_load_frac: float = 0.0
    has_more_steps: bool = True
    can_retry: bool = False
    n_cutbacks: int = 0  # 累積カットバック回数


class AdaptiveSteppingProcess(SolverProcess[AdaptiveStepInput, AdaptiveStepOutput]):
    """適応荷重増分制御（1判定 = 1 process()）.

    QUERY: 次の load_frac を返す（skip 対象は内部で消化）
    SUCCESS: ステップ成功 → 次ステップ幅を計算してキューに追加
    FAILURE: ステップ失敗 → カットバック可否を判定
    """

    meta = ProcessMeta(
        name="AdaptiveStepping",
        module="solve",
        version="1.0.0",
        document_path="docs/adaptive_stepping.md",
    )
    uses = []

    def __init__(self, config: AdaptiveSteppingInput) -> None:
        self._config = config
        self._queue: deque[float] = deque()
        self._consecutive_good: int = 0
        self._n_cutbacks: int = 0

        base_delta = config.dt_initial_fraction if config.dt_initial_fraction > 0.0 else 1.0
        # dt_max_fraction が設定されている場合、初期ステップもそれに制限
        if config.dt_max_fraction > 0.0:
            base_delta = min(base_delta, config.dt_max_fraction)
        self._queue.append(min(base_delta, 1.0))

    def process(self, input_data: AdaptiveStepInput) -> AdaptiveStepOutput:
        """適応荷重増分の1判定を実行する."""
        if input_data.action == StepAction.QUERY:
            return self._query(input_data.load_frac_prev)
        if input_data.action == StepAction.SUCCESS:
            return self._on_success(input_data)
        if input_data.action == StepAction.FAILURE:
            return self._on_failure(input_data)
        raise ValueError(f"未知のアクション: {input_data.action}")

    def _query(self, load_frac_prev: float) -> AdaptiveStepOutput:
        """次の有効な load_frac を返す（skip 対象は内部で消化）."""
        while self._queue:
            next_frac = self._queue[0]
            if next_frac <= load_frac_prev + 1e-15:
                self._queue.popleft()
                continue
            return AdaptiveStepOutput(next_load_frac=next_frac, has_more_steps=True)
        return AdaptiveStepOutput(has_more_steps=False)

    def _on_success(self, input_data: AdaptiveStepInput) -> AdaptiveStepOutput:
        """ステップ成功時: 消費 + 次ステップ幅決定."""
        if self._queue and self._queue[0] <= input_data.load_frac + 1e-15:
            self._queue.popleft()

        cfg = self._config
        load_frac = input_data.load_frac
        load_frac_prev = input_data.load_frac_prev

        if load_frac >= 1.0 - 1e-12:
            return AdaptiveStepOutput(has_more_steps=len(self._queue) > 0)

        current_delta = load_frac - load_frac_prev
        next_delta = current_delta

        if input_data.n_attempts <= cfg.dt_grow_attempt_threshold:
            self._consecutive_good += 1
            if self._consecutive_good <= 2:
                next_delta = current_delta * cfg.dt_grow_factor
            else:
                _damp = max(0.1, 1.0 / self._consecutive_good)
                next_delta = current_delta * (1.0 + (cfg.dt_grow_factor - 1.0) * _damp)
        elif input_data.n_attempts >= cfg.dt_shrink_attempt_threshold:
            next_delta = current_delta * cfg.dt_shrink_factor
            self._consecutive_good = 0
        else:
            self._consecutive_good = 0

        if input_data.prev_n_active > 0:
            change_rate = abs(input_data.n_active - input_data.prev_n_active) / max(
                input_data.prev_n_active, 1
            )
            if change_rate > cfg.dt_contact_change_threshold:
                next_delta = min(next_delta, current_delta * cfg.dt_shrink_factor)

        next_delta = max(next_delta, cfg.dt_min_fraction)
        next_delta = min(next_delta, cfg.dt_max_fraction)
        next_frac = min(load_frac + next_delta, 1.0)
        if 1.0 - next_frac < cfg.dt_min_fraction * 0.5:
            next_frac = 1.0
        self._queue.appendleft(next_frac)

        return AdaptiveStepOutput(
            next_load_frac=next_frac,
            has_more_steps=True,
        )

    def _on_failure(self, input_data: AdaptiveStepInput) -> AdaptiveStepOutput:
        """ステップ失敗時: カットバック判定."""
        delta = input_data.load_frac - input_data.load_frac_prev
        if delta <= self._config.dt_min_fraction + 1e-15:
            return AdaptiveStepOutput(
                has_more_steps=False,
                can_retry=False,
                n_cutbacks=self._n_cutbacks,
            )

        self._consecutive_good = 0
        self._n_cutbacks += 1

        # 発散検知時はより積極的に縮小（shrink_factor の2乗）
        shrink = self._config.dt_shrink_factor
        if input_data.diverged:
            shrink = shrink * shrink

        mid_frac = input_data.load_frac_prev + delta * shrink
        self._queue.appendleft(input_data.load_frac)
        self._queue.appendleft(mid_frac)

        return AdaptiveStepOutput(
            next_load_frac=mid_frac,
            has_more_steps=True,
            can_retry=True,
            n_cutbacks=self._n_cutbacks,
        )
