"""U1判断: Process vs Protocol のオーバーヘッド計測.

ProcessMetaclass のラッピング（トレース + プロファイリング）が
Newton反復内の高頻度呼び出しで許容可能かを検証する。

閾値: < 0.1ms/call なら Process として維持
"""

import time

import numpy as np

from xkep_cae.contact.penalty import (
    AutoBeamEIPenalty as AutoBeamEIProcess,
)
from xkep_cae.contact.penalty import (
    PenaltyInput,
)
from xkep_cae.core.base import ProcessMeta
from xkep_cae.core.categories import SolverProcess


# --- Protocol のみの軽量クラス（比較対象）---
class BareProtocolPenalty:
    """Process ラッパーなしの直接実装."""

    def __init__(self, k_pen: float) -> None:
        self._k_pen = k_pen

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        return self._k_pen


# --- SolverProcess 継承クラス（比較対象）---
class DummySolverProcess(SolverProcess):
    meta = ProcessMeta(
        name="DummySolver", module="solve", version="0.1.0", document_path="docs/benchmark.md"
    )

    def __init__(self, k_pen: float) -> None:
        self._k_pen = k_pen

    def process(self, input_data):
        return self._k_pen


def benchmark_call(func, n_calls: int = 100_000) -> float:
    """n_calls 回呼んだ平均時間(μs)を返す."""
    # ウォームアップ
    for _ in range(1000):
        func()

    t0 = time.perf_counter()
    for _ in range(n_calls):
        func()
    elapsed = time.perf_counter() - t0
    return elapsed / n_calls * 1e6  # μs


def main():
    n_calls = 200_000
    print("=== U1判断: Process vs Protocol オーバーヘッド計測 ===")
    print(f"呼び出し回数: {n_calls:,}\n")

    # 1. Protocol (bare) — compute_k_pen 直接呼び出し
    bare = BareProtocolPenalty(k_pen=1000.0)
    t_bare = benchmark_call(lambda: bare.compute_k_pen(0, 10), n_calls)
    print(f"[1] Protocol (bare)       : {t_bare:.3f} μs/call")

    # 2. SolverProcess — compute_k_pen 直接呼び出し（process()経由でない）
    auto = AutoBeamEIProcess(beam_E=200e3, beam_I=1e-4, L_elem=1.0)
    t_strategy_direct = benchmark_call(lambda: auto.compute_k_pen(0, 10), n_calls)
    print(f"[2] SolverProcess (direct): {t_strategy_direct:.3f} μs/call")

    # 3. SolverProcess — process() 経由（メタクラスラッピングあり）
    inp = PenaltyInput(step=0, total_steps=10)
    t_strategy_process = benchmark_call(lambda: auto.process(inp), n_calls)
    print(f"[3] SolverProcess (via process()): {t_strategy_process:.3f} μs/call")

    # 4. DummySolverProcess — process() 経由
    dummy = DummySolverProcess(k_pen=1000.0)
    t_dummy = benchmark_call(lambda: dummy.process(None), n_calls)
    print(f"[4] DummySolverProcess (process()): {t_dummy:.3f} μs/call")

    # 5. numpy 演算との比較（100要素のdot product）
    a = np.random.randn(100)
    b = np.random.randn(100)
    t_numpy = benchmark_call(lambda: np.dot(a, b), n_calls)
    print(f"[5] np.dot(100)           : {t_numpy:.3f} μs/call")

    # 6. numpy 演算との比較（1000要素のdot product）
    c = np.random.randn(1000)
    d = np.random.randn(1000)
    t_numpy_1k = benchmark_call(lambda: np.dot(c, d), n_calls)
    print(f"[6] np.dot(1000)          : {t_numpy_1k:.3f} μs/call")

    print("\n=== オーバーヘッド分析 ===")
    overhead_process = t_strategy_process - t_bare
    overhead_direct = t_strategy_direct - t_bare
    print(f"process() ラッピングオーバーヘッド: {overhead_process:.3f} μs/call")
    print(f"direct呼び出しオーバーヘッド:      {overhead_direct:.3f} μs/call")

    print("\n=== U1判断 ===")
    threshold_us = 100.0  # 0.1ms = 100μs
    if overhead_process < threshold_us:
        print(f"✓ process()オーバーヘッド ({overhead_process:.1f}μs) < 閾値 ({threshold_us}μs)")
        print("→ Strategy は Process として維持してよい")
        print("  ただし、Newton反復内では compute_k_pen() 等の直接呼び出しを推奨")
    else:
        print(f"✗ process()オーバーヘッド ({overhead_process:.1f}μs) >= 閾値 ({threshold_us}μs)")
        print("→ Strategy は Protocol に降格すべき")

    # NR反復での影響推定
    n_nr_iter = 20  # 典型的NR反復回数
    n_strategies = 5  # 5 Strategy 軸
    total_overhead_per_step = overhead_process * n_nr_iter * n_strategies
    print("\n=== NR反復での影響推定 ===")
    print(f"  NR反復/ステップ: {n_nr_iter}")
    print(f"  Strategy数: {n_strategies}")
    print(
        f"  1ステップあたりの追加時間: {total_overhead_per_step:.1f} μs = {total_overhead_per_step / 1000:.3f} ms"
    )
    print("  （典型的NR反復の1ステップ計算時間: 数百ms〜数秒）")


if __name__ == "__main__":
    main()
