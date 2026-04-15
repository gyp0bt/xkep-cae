"""work/beam_hysteresis/13_kc_component_fd_19strand.py — 19本撚線 K_c 成分分解 FD 診断.

[← README](README.md) | [← project README](../../README.md)

status-343 推奨アクション 1（本 Process の実運用）を実装。
status-342 で特定された **19本撚線 Type D stall の K_c FD 相対誤差 mean=115% /
x 成分 68% 不整合** の由来を K_mat / K_geo / K_st の部分行列レベルで切り分ける。

**三現主義**: status-342 の `12_kc_fd_diagnostic_19strand.py` は合成接線
K_c = K_mat - K_geo + K_st に対する単一の FD 相対誤差しか取得できない。
本スクリプトは `kc_component_fd_diagnostic=True` フラグを立てて
`ContactKcComponentFDDiagnosticProcess`（status-343 新設）を発火させ、
4 組み合わせ（full / mat_only / mat_geo / mat_st）の FD 相対誤差 + 成分別
不整合シェア + K_i 寄与率を stdout に吐く。本スクリプトはそれを regex parse
して CSV 化する。

仮説 A（status-341 再定義）: x 成分 68% 不整合の primary driver は
K_mat / K_geo / K_st のいずれかの部分行列。
- mat_only の rel_err が小さい → geo/st 追加で悪化 → geo/st が driver
- mat_only の rel_err が大きい → K_mat が driver（材料剛性が z 方向で破綻）

出力 CSV (`/tmp/kc_component_fd_19strand_<timestamp>.csv`):
    trigger_idx,
    full_rel_err, mat_only_rel_err, mat_geo_rel_err, mat_st_rel_err,
    share_mat, share_geo, share_st, best_combo,
    <combo>_comp_x_pct, <combo>_comp_y_pct, <combo>_comp_z_pct,
    <combo>_comp_tx_pct, <combo>_comp_ty_pct, <combo>_comp_tz_pct,
    （combo = full / mat_only / mat_geo / mat_st の 4 種、計 24 列）

実行:
    python work/beam_hysteresis/13_kc_component_fd_19strand.py 2>&1 \\
        | tee /tmp/kc_component_fd_19strand_$(date +%s).log
"""

from __future__ import annotations

import io
import re
import sys
import time
from contextlib import redirect_stdout

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

# ─── K_c 成分分解 FD レポートのパース用正規表現 ──────────────
# "K_c 成分分解 FD 診断（status-342）  [incr=... att=...]" で開始
_RE_KC_BEGIN = re.compile(r"K_c 成分分解 FD 診断")
# 組み合わせ別 rel_err 行
_RE_FULL_REL = re.compile(r"full\s*=\s*([0-9.eE+\-]+)")
_RE_MAT_ONLY_REL = re.compile(r"mat_only\s*=\s*([0-9.eE+\-]+)")
_RE_MAT_GEO_REL = re.compile(r"mat_geo\s*=\s*([0-9.eE+\-]+)")
_RE_MAT_ST_REL = re.compile(r"mat_st\s*=\s*([0-9.eE+\-]+)")
# 寄与率行 "寄与率: mat= X.XX, geo= X.XX, st= X.XX"
_RE_SHARES = re.compile(
    r"寄与率:\s*mat=\s*([0-9.eE+\-]+),\s*geo=\s*([0-9.eE+\-]+),\s*st=\s*([0-9.eE+\-]+)"
)
# 最良組み合わせ
_RE_BEST = re.compile(r"最良組み合わせ:\s*(\w+)")
# comp 別シェア: "  [    full] comp 別不整合: [x= XX.X%, y= XX.X%, z= XX.X%, tx= XX.X%, ty= XX.X%, tz= XX.X%]"
_RE_COMP = re.compile(
    r"\[\s*(full|mat_only|mat_geo|mat_st)\]\s*comp\s*別不整合:\s*\[\s*"
    r"x=\s*([0-9.]+)%,\s*y=\s*([0-9.]+)%,\s*z=\s*([0-9.]+)%,\s*"
    r"tx=\s*([0-9.]+)%,\s*ty=\s*([0-9.]+)%,\s*tz=\s*([0-9.]+)%\s*\]"
)

_COMP_KEYS = ("x", "y", "z", "tx", "ty", "tz")
_COMBOS = ("full", "mat_only", "mat_geo", "mat_st")


def _parse_kc_component_reports(log_text: str) -> list[dict[str, float | str]]:
    """stdout ログから K_c 成分分解 FD 診断レポートを抽出.

    "K_c 成分分解 FD 診断" 行を起点に、次の "=" 区切り行（2 回目）までを
    1 ブロックとする。
    """
    lines = log_text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] | None = None
    eq_count = 0
    for ln in lines:
        if _RE_KC_BEGIN.search(ln):
            if current:
                blocks.append(current)
            current = [ln]
            eq_count = 0
        elif current is not None:
            current.append(ln)
            if re.match(r"^\s*=+\s*$", ln):
                eq_count += 1
                # 始端 "=" + 終端 "=" で 2 回
                if eq_count >= 2:
                    blocks.append(current)
                    current = None
                    eq_count = 0
    if current:
        blocks.append(current)

    records: list[dict[str, float | str]] = []
    for blk in blocks:
        text = "\n".join(blk)
        rec: dict[str, float | str] = {}
        m = _RE_FULL_REL.search(text)
        if m:
            rec["full_rel_err"] = float(m.group(1))
        m = _RE_MAT_ONLY_REL.search(text)
        if m:
            rec["mat_only_rel_err"] = float(m.group(1))
        m = _RE_MAT_GEO_REL.search(text)
        if m:
            rec["mat_geo_rel_err"] = float(m.group(1))
        m = _RE_MAT_ST_REL.search(text)
        if m:
            rec["mat_st_rel_err"] = float(m.group(1))
        m = _RE_SHARES.search(text)
        if m:
            rec["share_mat"] = float(m.group(1))
            rec["share_geo"] = float(m.group(2))
            rec["share_st"] = float(m.group(3))
        m = _RE_BEST.search(text)
        if m:
            rec["best_combo"] = m.group(1)
        # comp 別シェア（4 combo × 6 comp）
        for cm in _RE_COMP.finditer(text):
            combo = cm.group(1)
            for i, ck in enumerate(_COMP_KEYS):
                rec[f"{combo}_comp_{ck}_pct"] = float(cm.group(i + 2))
        if "full_rel_err" in rec:
            records.append(rec)
    return records


def _write_csv(records: list[dict[str, float | str]], path: str) -> None:
    """抽出レコードを CSV に書き出す."""
    cols: list[str] = [
        "trigger_idx",
        "full_rel_err",
        "mat_only_rel_err",
        "mat_geo_rel_err",
        "mat_st_rel_err",
        "share_mat",
        "share_geo",
        "share_st",
        "best_combo",
    ]
    for combo in _COMBOS:
        for ck in _COMP_KEYS:
            cols.append(f"{combo}_comp_{ck}_pct")

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i, rec in enumerate(records):
            row = [str(i)]
            for c in cols[1:]:
                v = rec.get(c, "")
                if isinstance(v, float):
                    row.append(f"{v:.6e}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")


def _summarize(records: list[dict[str, float | str]]) -> None:
    """抽出レコードの要約を stdout に出力."""
    print()
    print("=" * 70)
    print(f"K_c 成分分解 FD 診断レポート抽出: {len(records)} 件")
    print("=" * 70)
    if not records:
        print("  ※ FD 診断レポートが 1 件も抽出できませんでした。")
        print("     (stall/Type D 連続が発火条件に到達しなかった可能性)")
        return

    # rel_err 統計
    print()
    print("  [組み合わせ別 FD 相対誤差]")
    for key in ("full_rel_err", "mat_only_rel_err", "mat_geo_rel_err", "mat_st_rel_err"):
        vals = [r[key] for r in records if key in r and isinstance(r[key], float)]
        if vals:
            arr = np.array(vals)
            print(
                f"    {key:20s}: n={len(arr):3d}, mean={arr.mean():.3e}, "
                f"min={arr.min():.3e}, max={arr.max():.3e}, median={np.median(arr):.3e}"
            )

    # 寄与率統計
    print()
    print("  [K_i 寄与率（||K_i@du||/||K_c@du||）]")
    for key in ("share_mat", "share_geo", "share_st"):
        vals = [r[key] for r in records if key in r and isinstance(r[key], float)]
        if vals:
            arr = np.array(vals)
            print(f"    {key:10s}: mean={arr.mean():.3f}, max={arr.max():.3f}")

    # 最良組み合わせ分布
    print()
    print("  [最良組み合わせ（rel_err 最小）分布]")
    best_counts: dict[str, int] = {}
    for r in records:
        b = r.get("best_combo", "")
        if isinstance(b, str) and b:
            best_counts[b] = best_counts.get(b, 0) + 1
    for combo in _COMBOS:
        n = best_counts.get(combo, 0)
        pct = 100.0 * n / max(len(records), 1)
        print(f"    {combo:10s}: {n:3d} 回 ({pct:5.1f}%)")

    # comp 別シェア（combo 別）
    print()
    print("  [comp 別不整合シェア平均（% 単位、0–100）]")
    for combo in _COMBOS:
        row_parts: list[str] = []
        for ck in _COMP_KEYS:
            key = f"{combo}_comp_{ck}_pct"
            vals = [r[key] for r in records if key in r and isinstance(r[key], float)]
            if vals:
                row_parts.append(f"{ck}={np.mean(vals):5.1f}%")
        if row_parts:
            print(f"    [{combo:>8s}] {'  '.join(row_parts)}")


def main() -> int:
    """19本撚線 + K_c 成分分解 FD 診断実測."""
    print("=" * 70)
    print("19本撚線 K_c 成分分解 FD 診断（status-343 推奨アクション 1）")
    print("仮説 A 最終検証: x 成分 68% 不整合の部分行列由来を切り分け")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,  # status-339 ベースライン
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        tangent_fd_diagnostic=True,  # ストール時 FD 診断（トリガー契機）
        kc_component_fd_diagnostic=True,  # ★ K_c 成分分解 FD 診断を ON
        track_contact_mk=False,
        track_contact_pairs=False,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, κ_max={cfg.bending_curvature}, "
        f"mu={cfg.mu}, n_incr_per_cycle={cfg.n_increments_per_cycle}, "
        f"tangent_fd_diagnostic={cfg.tangent_fd_diagnostic}, "
        f"kc_component_fd_diagnostic={cfg.kc_component_fd_diagnostic}"
    )
    print()

    # ── ソルバー実行 + stdout 捕捉 ──
    captured = io.StringIO()

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                s.write(data)

        def flush(self):
            for s in self._streams:
                s.flush()

    real_stdout = sys.stdout
    tee = _Tee(real_stdout, captured)

    t0 = time.perf_counter()
    with redirect_stdout(tee):
        result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    print()
    print("=" * 70)
    print("ソルバー結果")
    print("=" * 70)
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")

    # ── K_c 成分分解 FD 診断レポート抽出 ──
    log_text = captured.getvalue()
    records = _parse_kc_component_reports(log_text)

    _summarize(records)

    # CSV 書き出し
    csv_path = f"/tmp/kc_component_fd_19strand_{int(time.time())}.csv"
    _write_csv(records, csv_path)
    print(f"\n  → CSV 書き出し: {csv_path}")

    print()
    print("=" * 70)
    print("完了")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
