"""適応荷重増分制御（プライベート）.

AdaptiveSteppingConfig / AdaptiveLoadController を新パッケージに移植。
xkep_cae_deprecated/process/strategies/adaptive_stepping.py からのコピー。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

_setattr = object.__setattr__


@dataclass(frozen=True)
class AdaptiveSteppingConfig:
    """適応荷重増分の設定."""

    dt_initial_fraction: float = 0.0
    dt_grow_factor: float = 1.5
    dt_shrink_factor: float = 0.5
    dt_grow_iter_threshold: int = 5
    dt_shrink_iter_threshold: int = 15
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


@dataclass(frozen=True)
class AdaptiveLoadController:
    """適応荷重増分のステップキューデータ（純粋データ）."""

    config: AdaptiveSteppingConfig
    _queue: deque[float] = field(default_factory=deque)
    _consecutive_good: int = 0


def _create_load_controller(config: AdaptiveSteppingConfig) -> AdaptiveLoadController:
    """AdaptiveLoadController を初期化する."""
    ctrl = AdaptiveLoadController(config=config)
    base_delta = config.dt_initial_fraction if config.dt_initial_fraction > 0.0 else 1.0
    ctrl._queue.append(min(base_delta, 1.0))
    return ctrl


def _ctrl_has_steps(ctrl: AdaptiveLoadController) -> bool:
    """残りステップがあるか."""
    return len(ctrl._queue) > 0


def _ctrl_peek_next(ctrl: AdaptiveLoadController) -> float:
    """次のステップの load_frac を返す（キューからは取り出さない）."""
    return ctrl._queue[0]


def _ctrl_pop_step(ctrl: AdaptiveLoadController) -> float:
    """次のステップを取り出す."""
    return ctrl._queue.popleft()


def _ctrl_should_skip(
    ctrl: AdaptiveLoadController,
    load_frac: float,
    load_frac_prev: float,
) -> bool:
    """ステップをスキップすべきか."""
    return load_frac <= load_frac_prev + 1e-15


def _ctrl_on_success(
    ctrl: AdaptiveLoadController,
    load_frac: float,
    load_frac_prev: float,
    n_iters: int,
    n_active: int,
    prev_n_active: int,
) -> None:
    """ステップ成功時の次ステップ幅決定."""
    cfg = ctrl.config
    if load_frac >= 1.0 - 1e-12:
        return

    current_delta = load_frac - load_frac_prev
    next_delta = current_delta

    if n_iters <= cfg.dt_grow_iter_threshold:
        _setattr(ctrl, "_consecutive_good", ctrl._consecutive_good + 1)
        if ctrl._consecutive_good <= 2:
            next_delta = current_delta * cfg.dt_grow_factor
        else:
            _damp = max(0.1, 1.0 / ctrl._consecutive_good)
            next_delta = current_delta * (1.0 + (cfg.dt_grow_factor - 1.0) * _damp)
    elif n_iters >= cfg.dt_shrink_iter_threshold:
        next_delta = current_delta * cfg.dt_shrink_factor
        _setattr(ctrl, "_consecutive_good", 0)
    else:
        _setattr(ctrl, "_consecutive_good", 0)

    if prev_n_active > 0:
        change_rate = abs(n_active - prev_n_active) / max(prev_n_active, 1)
        if change_rate > cfg.dt_contact_change_threshold:
            next_delta = min(next_delta, current_delta * cfg.dt_shrink_factor)

    next_delta = max(next_delta, cfg.dt_min_fraction)
    next_delta = min(next_delta, cfg.dt_max_fraction)
    next_frac = min(load_frac + next_delta, 1.0)
    if 1.0 - next_frac < cfg.dt_min_fraction * 0.5:
        next_frac = 1.0
    ctrl._queue.appendleft(next_frac)


def _ctrl_on_failure(
    ctrl: AdaptiveLoadController,
    load_frac: float,
    load_frac_prev: float,
) -> bool:
    """ステップ失敗時のカットバック.

    Returns:
        True: カットバック成功（リトライ可能）
        False: これ以上縮小不可
    """
    delta = load_frac - load_frac_prev
    if delta <= ctrl.config.dt_min_fraction + 1e-15:
        return False

    _setattr(ctrl, "_consecutive_good", 0)
    mid_frac = load_frac_prev + delta * ctrl.config.dt_shrink_factor
    ctrl._queue.appendleft(load_frac)
    ctrl._queue.appendleft(mid_frac)
    return True
