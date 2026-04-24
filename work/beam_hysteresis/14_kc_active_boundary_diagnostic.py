"""Phase C-3' Step 3.1: active 境界下での K_c FD 整合性診断（status-370）.

status-356 で仮説 A + B 同時導入により `test_helical_3d_hermite` の FD
rel_err は **1.795% → 2.18e-07**（機械精度）を達成した。ただし status-357
の 19 本撚線実測では frac=0.3739 と退化し、Phase C-3' の機械精度達成は
**active 集合固定下限定**と判定されている（詳細: `docs/status/status-357.md`,
`xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md`）。

本スクリプトは `14_kc_closest_adj_diagnostic.py` を素体に、

  (1) `gap_target` を深い接触（-1e-2）から active 境界近傍（-1e-6）まで
      sweep し、Hertz + Hermite + 3D ヘリカルでの K_c FD rel_err の
      gap 依存性を測定する（delta_h=0 の C0 Huber ベースライン）。
  (2) `smoothing_delta=2000`（StrandBendingOscillationConfig 既定）を
      有効化し、Huber 平滑化ゾーン `g ∈ [-δ_h/k_pen, +δ_h/k_pen]` を
      跨ぐ gap を測定する（k_pen=1e4 で δ_h=5, gap ゾーン ±5e-4）。

Step 3.1 の成否判定:

  - 結果 A（境界で rel_err 悪化）: gap=-1e-3 以深で rel_err が 2 桁以上
    悪化するなら仮説（active-flip 項の欠落）裏付け。Step 3.2 で新項
    `KcActiveFlipStiffness` を 6 項目化で追加する。
  - 結果 B（境界でも rel_err 健全）: active 判定そのものではなく NR
    履歴・alg側の問題。候補 (g) 再計画。

Usage:
    python work/beam_hysteresis/14_kc_active_boundary_diagnostic.py 2>&1 | \\
        tee /tmp/log-active-boundary-$(date +%s).log

Reference:
- status-370（本 status）
- xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md §3.1
- docs/status/status-356.md（2 経路相殺 machine precision）
- docs/status/status-357.md（19 本退化）
- work/beam_hysteresis/14_kc_closest_adj_diagnostic.py（素体）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# プロジェクトルートを import path に
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xkep_cae.contact.contact_force.strategy import (  # noqa: E402
    HuberContactForceProcess,
)
from xkep_cae.contact.contact_force.tests.test_kc_component_fd import (  # noqa: E402
    _compute_fd_kc,
    _get_analytical_components,
    _make_two_segment_scenario,
)


@dataclass
class DiagnosticRow:
    """1 gap_target / smoothing_delta 組み合わせの測定結果."""

    gap_target: float
    smoothing_delta: float
    delta_h: float
    fd_eps: float
    k_c_norm: float
    fd_norm: float
    diff_norm: float
    rel_err_total: float
    rel_err_aa: float  # active × active
    rel_err_ax_abs: float  # active × adj 絶対 (K_c[ax] は status-356 後も
    # adj へ伸びるので参考値)
    diff_share_aa: float
    diff_share_ax: float
    diff_share_xa: float
    diff_share_xx: float


def run_single(
    gap_target: float,
    smoothing_delta: float,
    k_pen: float = 1e4,
    fd_eps: float = 1e-7,
) -> DiagnosticRow:
    """1 条件で FD vs 解析的 K_c を比較し block 別 rel_err を返す."""
    scenario = _make_two_segment_scenario(
        gap_target=gap_target,
        use_hermite=True,
        penalty_exponent=1.5,
        n_elems=3,
        helical_z=True,
        k_pen=k_pen,
    )

    # smoothing_delta が既定 0 以外のときは strategy を差し替える
    if smoothing_delta > 0.0:
        scenario["strategy"] = HuberContactForceProcess(
            ndof=scenario["ndof"],
            ndof_per_node=scenario["ndof_per_node"],
            penalty_exponent=1.5,
            smoothing_delta=smoothing_delta,
        )

    delta_h = k_pen / smoothing_delta if smoothing_delta > 0 else 0.0

    ndpn = scenario["ndof_per_node"]
    n_nodes = scenario["n_nodes"]
    trans_dofs = np.array(
        [node * ndpn + d for node in range(n_nodes) for d in range(3)],
        dtype=int,
    )

    fd_kc = _compute_fd_kc(scenario, eps=fd_eps, dof_indices=trans_dofs)
    K_mat, K_geo, K_st = _get_analytical_components(scenario)
    K_c = K_mat - K_geo + K_st

    diff = (K_c - fd_kc).toarray()
    fd_d = fd_kc.toarray()
    kc_d = K_c.toarray()

    total_diff = np.linalg.norm(diff)
    total_fd = np.linalg.norm(fd_d)
    total_kc = np.linalg.norm(kc_d)
    rel_total = total_diff / max(total_fd, 1e-30)

    # active / adj ノード（status-355/356 と同じ定義）
    active_nodes = np.array([1, 2, 5, 6])
    adj_nodes = np.array([0, 3, 4, 7])
    active_dofs = np.array(
        [n * ndpn + d for n in active_nodes for d in range(3)], dtype=int
    )
    adj_dofs = np.array(
        [n * ndpn + d for n in adj_nodes for d in range(3)], dtype=int
    )

    diff_aa = diff[np.ix_(active_dofs, active_dofs)]
    diff_ax = diff[np.ix_(active_dofs, adj_dofs)]
    diff_xa = diff[np.ix_(adj_dofs, active_dofs)]
    diff_xx = diff[np.ix_(adj_dofs, adj_dofs)]

    fd_aa = fd_d[np.ix_(active_dofs, active_dofs)]
    n_aa_diff = np.linalg.norm(diff_aa)
    n_ax_diff = np.linalg.norm(diff_ax)
    n_xa_diff = np.linalg.norm(diff_xa)
    n_xx_diff = np.linalg.norm(diff_xx)
    n_aa_fd = np.linalg.norm(fd_aa)

    # シェア（全体 rel_err の源泉分解）
    denom = max(total_diff**2, 1e-60)
    return DiagnosticRow(
        gap_target=gap_target,
        smoothing_delta=smoothing_delta,
        delta_h=delta_h,
        fd_eps=fd_eps,
        k_c_norm=total_kc,
        fd_norm=total_fd,
        diff_norm=total_diff,
        rel_err_total=rel_total,
        rel_err_aa=n_aa_diff / max(n_aa_fd, 1e-30),
        rel_err_ax_abs=n_ax_diff,
        diff_share_aa=(n_aa_diff**2 / denom) * 100.0,
        diff_share_ax=(n_ax_diff**2 / denom) * 100.0,
        diff_share_xa=(n_xa_diff**2 / denom) * 100.0,
        diff_share_xx=(n_xx_diff**2 / denom) * 100.0,
    )


def print_table(title: str, rows: list[DiagnosticRow]) -> None:
    """測定結果を 1 テーブルで出力."""
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    header = (
        f"{'gap_target':>12} {'fd_eps':>9} {'smooth_δ':>9} {'δ_h':>7} "
        f"{'||K_c||':>10} {'||FD||':>10} {'rel_tot':>10} "
        f"{'rel_aa':>10} "
        f"{'aa%':>6} {'ax%':>6} {'xa%':>6} {'xx%':>6}"
    )
    print(header)
    print("-" * 110)
    for r in rows:
        print(
            f"{r.gap_target:>12.3e} "
            f"{r.fd_eps:>9.2e} "
            f"{r.smoothing_delta:>9.1f} "
            f"{r.delta_h:>7.2e} "
            f"{r.k_c_norm:>10.3e} "
            f"{r.fd_norm:>10.3e} "
            f"{r.rel_err_total:>10.3e} "
            f"{r.rel_err_aa:>10.3e} "
            f"{r.diff_share_aa:>6.2f} "
            f"{r.diff_share_ax:>6.2f} "
            f"{r.diff_share_xa:>6.2f} "
            f"{r.diff_share_xx:>6.2f}"
        )


def judge(
    rows_unsmooth: list[DiagnosticRow],
    rows_smooth: list[DiagnosticRow],
    rows_flip: list[DiagnosticRow],
) -> str:
    """Step 3.1 結果 A / B を判定してサマリ文字列を返す."""
    # 基準: status-356 再現の深い接触
    baseline = next(
        (r for r in rows_unsmooth if abs(r.gap_target + 1e-2) < 1e-6), None
    )
    if baseline is None:
        return "判定不能: baseline (gap=-1e-2) 行が見つかりません"

    baseline_err = baseline.rel_err_total

    # 境界近傍: gap_target=-1e-5 or -1e-6 のうちどれか最悪値
    boundary_candidates = [
        r for r in rows_unsmooth if abs(r.gap_target) <= 1e-4 + 1e-9
    ]
    if not boundary_candidates:
        return "判定不能: 境界近傍行が見つかりません"
    worst_boundary = max(boundary_candidates, key=lambda r: r.rel_err_total)
    boundary_err = worst_boundary.rel_err_total

    degradation = boundary_err / max(baseline_err, 1e-30)
    log_deg = np.log10(degradation) if degradation > 0 else -np.inf

    print(f"\n{'=' * 100}")
    print("  Step 3.1 判定サマリ")
    print(f"{'=' * 100}")
    print(f"  baseline rel_err (gap=-1e-2)         = {baseline_err:.3e}")
    print(
        f"  worst boundary rel_err "
        f"(gap={worst_boundary.gap_target:.1e}) = {boundary_err:.3e}"
    )
    print(f"  degradation                          = {degradation:.2f}x ({log_deg:+.2f} 桁)")

    # 平滑化下では gap=0 近傍で FD と解析が両方非零のはず
    smooth_zone_rows = [r for r in rows_smooth if abs(r.gap_target) <= 5e-4 + 1e-9]
    if smooth_zone_rows:
        worst_smooth = max(smooth_zone_rows, key=lambda r: r.rel_err_total)
        print(
            f"  smoothed zone worst rel_err "
            f"(gap={worst_smooth.gap_target:.1e}, δ={worst_smooth.smoothing_delta:.0f}) "
            f"= {worst_smooth.rel_err_total:.3e}"
        )

    # FD 中に active flip を強制発生させたケース
    if rows_flip:
        worst_flip = max(rows_flip, key=lambda r: r.rel_err_total)
        print(
            f"  flip-trigger worst rel_err "
            f"(gap={worst_flip.gap_target:.1e}, fd_eps={worst_flip.fd_eps:.1e}, "
            f"δ_h={worst_flip.delta_h:.1e}) = {worst_flip.rel_err_total:.3e}"
        )

    # 判定閾値: 2 桁以上悪化なら結果 A
    if log_deg >= 2.0:
        verdict = (
            "結果 A: active 境界で rel_err が 2 桁以上悪化。"
            "Step 3.2 で新項 `KcActiveFlipStiffness` 追加に進む。"
        )
    elif log_deg >= 1.0:
        verdict = (
            f"結果 A(弱): rel_err が {degradation:.2f}x 悪化（1 桁相当）。"
            "Step 3.2 実施の必要性は data から再議論。"
        )
    else:
        verdict = (
            "結果 B: active 境界でも rel_err 健全。"
            "項の欠落ではなく NR alg 側（履歴平滑化 / 拡大 Lagrangian 等）"
            "の問題として候補 (g) に再計画。"
        )
    print(f"\n  ▶ 判定: {verdict}")
    return verdict


def main() -> None:
    # ── Block 1: δ_h = 0 (C0 Huber), gap sweep ──
    # gap が 0 に近づくほど p_n→0 で active 境界に近い「深さ」が浅くなる。
    # delta_h=0 なので g>0 は p_n=0 (FD=0) になるため、解析で比較不能。
    # 負側のみを sweep し、active 境界から離れた（深い接触）→境界近傍の傾向を見る。
    gap_sweep_unsmoothed = [
        -1e-2,  # status-356 gate 再現（2.18e-07 期待）
        -5e-3,
        -1e-3,
        -5e-4,
        -1e-4,
        -1e-5,
        -1e-6,  # active 境界直前
    ]

    rows_unsmooth: list[DiagnosticRow] = []
    for gap in gap_sweep_unsmoothed:
        print(f"[unsmoothed] gap_target={gap:+.3e} 測定中 ...", flush=True)
        rows_unsmooth.append(run_single(gap, smoothing_delta=0.0))

    print_table(
        "Block 1: δ_h = 0 (C0 Huber)  gap sweep  "
        "— penalty_exponent=1.5, use_hermite=True, helical_z=True, n_elems=3",
        rows_unsmooth,
    )

    # ── Block 2: δ_h > 0 (StrandBendingOscillationConfig default), gap sweep ──
    # smoothing_delta=2000, k_pen=1e4 → δ_h=5, ゾーン g ∈ [-5e-4, +5e-4]
    # 境界を跨ぐ gap で rel_err を見る。g>0 の inactive 側も含める。
    gap_sweep_smoothed = [
        -1e-3,  # ゾーン外（active 側）
        -5e-4,  # ゾーン境界
        -2e-4,
        -1e-4,
        0.0,  # ゾーン中央
        +1e-4,
        +2e-4,
        +5e-4,  # ゾーン境界
    ]

    rows_smooth: list[DiagnosticRow] = []
    for gap in gap_sweep_smoothed:
        print(f"[smoothed δ=2000] gap_target={gap:+.3e} 測定中 ...", flush=True)
        rows_smooth.append(run_single(gap, smoothing_delta=2000.0))

    print_table(
        "Block 2: δ_h = 5 (smoothing_delta=2000)  gap sweep across active boundary  "
        "— Huber 平滑化ゾーン |g|<5e-4",
        rows_smooth,
    )

    # ── Block 3: FD 中に active flip を強制発生 ──
    # eps を gap_target 絶対値より大きく取ることで、FD 摂動中に gap が符号反転
    # → Huber active 側から inactive 側への flip を経験する。status-356 の
    # 機械精度達成が「FD eps 内で active 側に留まる」ことに依存しているか、
    # それとも境界跨ぎでも成立するかを検定する。
    # 各ケース: (gap_target, fd_eps, smoothing_delta)
    #   - gap=-5e-8, eps=1e-7: 符号跨ぎ（δ_h=0）、unsmoothed で flip 強制
    #   - gap=-1e-5, eps=1e-4: 1 桁強 flip（eps >> |gap|）
    #   - gap=+1e-8, eps=1e-7: 初期 inactive、摂動で一部 active 化
    #   - smoothed δ=2000 版: ゾーン内 flip（解析的に滑らかなので一致期待）
    flip_cases = [
        ("C0 Huber  flip 小", -5e-8, 1e-7, 0.0),
        ("C0 Huber  flip 大", -1e-5, 1e-4, 0.0),
        ("C0 Huber  初期 inactive→flip", +1e-8, 1e-7, 0.0),
        ("Huber δ=2000 ゾーン flip 小", -5e-8, 1e-7, 2000.0),
        ("Huber δ=2000 ゾーン flip 大", -1e-5, 1e-4, 2000.0),
    ]
    rows_flip: list[DiagnosticRow] = []
    for label, gap, eps, sdelta in flip_cases:
        print(
            f"[flip:{label}] gap={gap:+.2e}, eps={eps:.1e}, δ={sdelta:.0f} 測定中 ...",
            flush=True,
        )
        rows_flip.append(run_single(gap, smoothing_delta=sdelta, fd_eps=eps))

    print_table(
        "Block 3: 強制 active flip  "
        "— fd_eps ≥ |gap| で FD 中に符号反転を発生させる",
        rows_flip,
    )

    # ── 判定 ──
    judge(rows_unsmooth, rows_smooth, rows_flip)


if __name__ == "__main__":
    main()
