"""撚線曲げ揺動ベンチマーク.

z軸上に配置した撚線の一端を固定し、他端の回転角を処方（変位制御）して ~90° 曲げ、
曲がった状態で z 方向にサイクル変位を2周期与える。

実行ロジックは _backend 経由で注入する（C14 準拠）。
データクラス・定数・純粋関数のみをこのモジュールで定義する。

[← README](../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xkep_cae.numerical_tests._backend import backend

# ====================================================================
# ローカルデータクラス
# ====================================================================


@dataclass(frozen=True)
class ContactSolveResult:
    """ソルバー結果."""

    u: np.ndarray
    converged: bool
    n_load_steps: int = 0
    total_newton_iterations: int = 0
    total_outer_iterations: int = 0
    n_active_final: int = 0
    load_history: list = field(default_factory=list)
    displacement_history: list = field(default_factory=list)
    contact_force_history: list = field(default_factory=list)
    graph_history: list = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkTimingCollector:
    """タイミング収集（frozen dataclass）."""

    _records: tuple[dict, ...] = ()

    def record(
        self, phase: int, step: int, iteration: int, label: str, elapsed: float
    ) -> BenchmarkTimingCollector:
        """新しいレコードを追加した新インスタンスを返す."""
        new_record = {
            "phase": phase,
            "step": step,
            "iteration": iteration,
            "label": label,
            "elapsed": elapsed,
        }
        return BenchmarkTimingCollector(_records=(*self._records, new_record))

    def summary_table(self) -> str:
        lines = ["  Timing Summary:"]
        for r in self._records:
            lines.append(f"    {r['label']:30s} {r['elapsed']:.4f} s")
        return "\n".join(lines)


# ====================================================================
# 物理パラメータ（鋼線デフォルト）
# ====================================================================

_DEFAULT_E = 200e3  # MPa（鋼、mm-ton-MPa単位系）
_DEFAULT_NU = 0.3
_WIRE_D = 2.0  # mm 直径
_NDOF_PER_NODE = 6


def _compute_G(E: float, nu: float) -> float:
    """せん断弾性係数を計算."""
    return E / (2.0 * (1.0 + nu))


def _compute_kappa(nu: float) -> float:
    """Cowper (1966) 円形断面のせん断補正係数を計算."""
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


# ====================================================================
# 結果データクラス
# ====================================================================


@dataclass(frozen=True)
class BendingOscillationResult:
    """曲げ揺動ベンチマーク結果.

    Attributes:
        n_strands: 素線本数
        n_elems: 総要素数
        n_nodes: 総節点数
        ndof: 総自由度数
        mesh_length: モデル長さ [mm]
        phase1_converged: Phase 1（曲げ）収束フラグ
        phase2_converged: Phase 2（揺動）収束フラグ
        phase1_result: Phase 1 のソルバー結果
        phase2_results: Phase 2 の各変位ステップのソルバー結果
        timing: 工程別タイミングデータ
        total_time_s: 総計算時間 [s]
        tip_displacement_final: 先端変位 (x, y, z) [mm]
        max_penetration_ratio: 最大貫入比
        n_active_contacts: 最終活性接触ペア数
        displacement_snapshots: GIF用の各ステップ変形座標
        snapshot_labels: 各スナップショットのラベル
    """

    n_strands: int
    n_elems: int
    n_nodes: int
    ndof: int
    mesh_length: float
    phase1_converged: bool
    phase2_converged: bool
    phase1_result: ContactSolveResult
    phase2_results: list[ContactSolveResult] = field(default_factory=list)
    timing: BenchmarkTimingCollector | None = None
    total_time_s: float = 0.0
    tip_displacement_final: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_penetration_ratio: float = 0.0
    n_active_contacts: int = 0
    displacement_snapshots: list[np.ndarray] = field(default_factory=list)
    snapshot_labels: list[str] = field(default_factory=list)


# ====================================================================
# 純粋ヘルパー
# ====================================================================


def _deformed_coords(node_coords_ref: np.ndarray, u: np.ndarray) -> np.ndarray:
    """変形座標を計算 (n_nodes, 3)."""
    n_nodes = node_coords_ref.shape[0]
    coords = node_coords_ref.copy()
    for i in range(n_nodes):
        coords[i, 0] += u[_NDOF_PER_NODE * i]
        coords[i, 1] += u[_NDOF_PER_NODE * i + 1]
        coords[i, 2] += u[_NDOF_PER_NODE * i + 2]
    return coords


# ====================================================================
# レポート出力
# ====================================================================


def _print_benchmark_report(result: BendingOscillationResult) -> str:
    """ベンチマーク結果のフォーマット済みレポートを返す."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  曲げ揺動ベンチマーク結果: {result.n_strands}本")
    lines.append(f"{'=' * 70}")
    lines.append(f"  要素数:       {result.n_elems}")
    lines.append(f"  節点数:       {result.n_nodes}")
    lines.append(f"  自由度数:     {result.ndof}")
    lines.append(f"  モデル長さ:   {result.mesh_length:.1f} mm")
    lines.append(f"  Phase 1 収束: {result.phase1_converged}")
    lines.append(f"  Phase 2 収束: {result.phase2_converged}")
    lines.append(f"  活性接触:     {result.n_active_contacts}")
    lines.append(f"  最大貫入比:   {result.max_penetration_ratio:.6f}")
    lines.append(f"  総計算時間:   {result.total_time_s:.2f} s")

    dx, dy, dz = result.tip_displacement_final
    lines.append(f"  先端変位:     dx={dx:.3f} mm, dy={dy:.3f} mm, dz={dz:.3f} mm")

    total_nr = result.phase1_result.total_newton_iterations
    for r in result.phase2_results:
        total_nr += r.total_newton_iterations
    lines.append(f"  NR反復合計:   {total_nr}")
    lines.append(f"  Phase2ステップ数: {len(result.phase2_results)}")

    if result.timing is not None:
        lines.append("")
        lines.append(result.timing.summary_table())

    report = "\n".join(lines)
    return report


# ====================================================================
# メイン実行関数 — backend 委譲
# ====================================================================


def _run_bending_oscillation(**kwargs: Any) -> BendingOscillationResult:
    """曲げ揺動ベンチマークを実行.

    実行ロジックは backend の bending_oscillation_runner に委譲する。
    backend が未設定の場合は RuntimeError を送出する。

    Args:
        **kwargs: run_bending_oscillation の全引数（n_strands, wire_diameter, etc.）

    Returns:
        BendingOscillationResult
    """
    runner = getattr(backend, "_bending_oscillation_runner", None)
    if runner is None:
        raise RuntimeError(
            "Backend 未設定（bending_oscillation_runner）。"
            "conftest.py で backend._bending_oscillation_runner を注入してください。"
        )
    return runner(**kwargs)


def _run_scaling_benchmark(
    strand_counts: list[int] | None = None,
    **kwargs: Any,
) -> list[BendingOscillationResult]:
    """複数の素線本数で曲げ揺動ベンチマークを実行しスケーリングを分析."""
    if strand_counts is None:
        strand_counts = [7, 19, 37]

    results = []
    for n in strand_counts:
        result = _run_bending_oscillation(n_strands=n, **kwargs)
        results.append(result)

    print(f"\n{'=' * 80}")
    print("  曲げ揺動ベンチマーク スケーリングレポート")
    print(f"{'=' * 80}")
    header = (
        f"{'n_strands':>10} {'n_elems':>8} {'ndof':>8} "
        f"{'P1_conv':>8} {'P2_conv':>8} {'active':>7} "
        f"{'pen_ratio':>10} {'time(s)':>10}"
    )
    print(header)
    print("-" * 80)
    for r in results:
        print(
            f"{r.n_strands:>10} {r.n_elems:>8} {r.ndof:>8} "
            f"{'Y' if r.phase1_converged else 'N':>8} "
            f"{'Y' if r.phase2_converged else 'N':>8} "
            f"{r.n_active_contacts:>7} "
            f"{r.max_penetration_ratio:>10.6f} "
            f"{r.total_time_s:>10.2f}"
        )
    print()

    return results


# 旧名との互換性
WireBendingBenchmarkResult = BendingOscillationResult
run_wire_bending_benchmark = _run_bending_oscillation
