"""work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py — 19本撚線 K_c FD 診断取得.

[← README](README.md) | [← project README](../../README.md)

status-341 推奨アクション 1（最優先・新規）を実装。

status-339 / 341 で観察された 19本撚線 Type D stall（frac=0.199〜0.48）の
成分別不整合を、ソルバー内蔵の `TangentFDDiagnosticProcess` で直接定量化する。

**三現主義**: status-341 は `[NR診断]` 行の comp 残差（R_u 成分分布）のみを
記録しており、**K_c 自体の FD 相対誤差** および **K_c@du の comp 別不整合**
は未取得。本スクリプトでは `tangent_fd_diagnostic=True` に加え `type_d_auto_fd=True`
（デフォルト）により、FD 診断が stall 時と Type D 連続検知時に自動発火し、
`TangentFDDiagnosticOutput.report` を stdout に吐く。これを解析して CSV
にまとめ、次工程（仮説 A: StJacobian z 成分カップリング検証）のための
根拠固めとする。

出力 CSV (`/tmp/kc_fd_diag_19strand_<timestamp>.csv`):
    trigger_idx, f_c_rel_err,
    full_rel_err, full_comp_x_pct, full_comp_y_pct, full_comp_z_pct,
                  full_comp_tx_pct, full_comp_ty_pct, full_comp_tz_pct,
    fc_comp_x_pct, fc_comp_y_pct, fc_comp_z_pct,
    fc_comp_tx_pct, fc_comp_ty_pct, fc_comp_tz_pct,
    directional_ratio, deriv_agreement

実行:
    python work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py 2>&1 \
        | tee /tmp/kc_fd_diag_19strand_$(date +%s).log
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

# FD 診断レポートのパース用正規表現
# "全体系 K@du: FD vs 解析 相対誤差 = X.XXXXe-XX"
_RE_FULL_REL = re.compile(r"全体系 K@du: FD vs 解析 相対誤差 = ([0-9.eE+\-]+)")
# "f_c FD相対誤差 = X.XXXXe-XX"
_RE_FC_REL = re.compile(r"f_c FD相対誤差 = ([0-9.eE+\-]+)")
# "方向有効性: ||R(u+eps*du)||/||R(u)|| = X"
_RE_DIR_RATIO = re.compile(r"方向有効性: \|\|R\(u\+eps\*du\)\|\|/\|\|R\(u\)\|\| = ([0-9.eE+\-]+)")
# "方向微分整合度: |FD-解析|/max = X"
_RE_DERIV_AGREE = re.compile(r"方向微分整合度: \|FD-解析\|/max = ([0-9.eE+\-]+)")
# "comp別不整合: [x:X(X%), y:X(X%), z:X(X%), θx:X(X%), θy:X(X%), θz:X(X%)]"
_RE_FULL_COMP = re.compile(
    r"comp別不整合: \[x:[^(]+\((\d+)%\), y:[^(]+\((\d+)%\), z:[^(]+\((\d+)%\), "
    r"θx:[^(]+\((\d+)%\), θy:[^(]+\((\d+)%\), θz:[^(]+\((\d+)%\)\]"
)
# "f_c comp別不整合: [x=X/X%, y=X/X%, z=X/X%, θx=X/X%, θy=X/X%, θz=X/X%]"
_RE_FC_COMP = re.compile(
    r"f_c comp別不整合: \[x=[^/]+/(\d+)%, y=[^/]+/(\d+)%, z=[^/]+/(\d+)%, "
    r"θx=[^/]+/(\d+)%, θy=[^/]+/(\d+)%, θz=[^/]+/(\d+)%\]"
)
# "接線剛性FD方向診断"で始まる区切り行
_RE_FD_BEGIN = re.compile(r"接線剛性FD方向診断")


def _parse_fd_reports(log_text: str) -> list[dict[str, float]]:
    """stdout ログから FD 診断レポートを抽出.

    FD レポートは複数行の塊で 1 トリガーごとに出力される。
    "接線剛性FD方向診断" 行を起点に、次の "=" 区切り行まで（もしくは
    次の "接線剛性FD方向診断" 行まで）を 1 ブロックとする。
    """
    lines = log_text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for ln in lines:
        if _RE_FD_BEGIN.search(ln):
            if current:
                blocks.append(current)
            current = [ln]
        elif current is not None:
            current.append(ln)
            # 区切り行 60 文字以上の "=" 連続 → ブロック終端（2 回目で確定）
            if re.match(r"^\s*=+\s*$", ln):
                # "=" は始端と終端の 2 回登場。2 回目を検出したら閉じる
                # 簡易判定: current 長さが 3 以上の場合のみ終端扱い
                if len(current) > 3:
                    blocks.append(current)
                    current = None
    if current:
        blocks.append(current)

    records: list[dict[str, float]] = []
    for blk in blocks:
        text = "\n".join(blk)
        rec: dict[str, float] = {}
        m = _RE_FULL_REL.search(text)
        if m:
            rec["full_rel_err"] = float(m.group(1))
        m = _RE_FC_REL.search(text)
        if m:
            rec["fc_rel_err"] = float(m.group(1))
        m = _RE_DIR_RATIO.search(text)
        if m:
            rec["directional_ratio"] = float(m.group(1))
        m = _RE_DERIV_AGREE.search(text)
        if m:
            rec["deriv_agreement"] = float(m.group(1))
        m = _RE_FULL_COMP.search(text)
        if m:
            for i, k in enumerate(["x", "y", "z", "tx", "ty", "tz"]):
                rec[f"full_comp_{k}_pct"] = float(m.group(i + 1))
        m = _RE_FC_COMP.search(text)
        if m:
            for i, k in enumerate(["x", "y", "z", "tx", "ty", "tz"]):
                rec[f"fc_comp_{k}_pct"] = float(m.group(i + 1))
        # 少なくとも 1 つの数値が取れたら記録
        if rec:
            records.append(rec)
    return records


def _write_csv(records: list[dict[str, float]], path: str) -> None:
    """抽出レコードを CSV に書き出す."""
    cols = [
        "trigger_idx",
        "full_rel_err",
        "fc_rel_err",
        "directional_ratio",
        "deriv_agreement",
        "full_comp_x_pct",
        "full_comp_y_pct",
        "full_comp_z_pct",
        "full_comp_tx_pct",
        "full_comp_ty_pct",
        "full_comp_tz_pct",
        "fc_comp_x_pct",
        "fc_comp_y_pct",
        "fc_comp_z_pct",
        "fc_comp_tx_pct",
        "fc_comp_ty_pct",
        "fc_comp_tz_pct",
    ]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i, rec in enumerate(records):
            row = [str(i)]
            for c in cols[1:]:
                v = rec.get(c, "")
                row.append(f"{v:.6e}" if isinstance(v, float) else str(v))
            f.write(",".join(row) + "\n")


def _summarize(records: list[dict[str, float]]) -> None:
    """抽出レコードの要約を stdout に出力."""
    print()
    print("=" * 70)
    print(f"FD 診断レポート抽出: {len(records)} 件")
    print("=" * 70)
    if not records:
        print("  ※ FD 診断レポートが 1 件も抽出できませんでした。")
        print("     (stall/Type D 連続が発火条件に到達しなかった可能性)")
        return

    # 各スカラーキーの統計
    for key in ("full_rel_err", "fc_rel_err", "directional_ratio", "deriv_agreement"):
        vals = [r[key] for r in records if key in r]
        if vals:
            arr = np.array(vals)
            print(
                f"  {key:22s}: n={len(arr):3d}, mean={arr.mean():.3e}, "
                f"min={arr.min():.3e}, max={arr.max():.3e}, median={np.median(arr):.3e}"
            )

    # comp 分布（K@du 全体系 + f_c 単独）
    print()
    for kind, prefix in (("全体系 K@du comp 別", "full_comp"), ("f_c comp 別", "fc_comp")):
        print(f"  [{kind}] 平均シェア（% 単位）:")
        for c in ("x", "y", "z", "tx", "ty", "tz"):
            vals = [r[f"{prefix}_{c}_pct"] for r in records if f"{prefix}_{c}_pct" in r]
            if vals:
                arr = np.array(vals)
                print(f"    {c:3s}: mean={arr.mean():5.1f}%, max={arr.max():5.1f}%, n={len(arr)}")
        print()


def main() -> int:
    """19本撚線 + K_c FD 診断実測."""
    print("=" * 70)
    print("19本撚線 K_c FD 診断（status-341 推奨アクション 1 — 最優先）")
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
        tangent_fd_diagnostic=True,  # ★ ストール時 FD 診断を明示的に ON
        # type_d_auto_fd は solver default True なので別途不要
        track_contact_mk=False,  # 本セッションは K_c FD 診断のみ収集
        track_contact_pairs=False,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, κ_max={cfg.bending_curvature}, "
        f"mu={cfg.mu}, n_incr_per_cycle={cfg.n_increments_per_cycle}, "
        f"tangent_fd_diagnostic={cfg.tangent_fd_diagnostic}"
    )
    print()

    # ── ソルバー実行 + stdout 捕捉 ──
    # ソルバーは show_progress=True により FD report を stdout に print する。
    # parseで使うため io.StringIO で二重化（tee 的に元 stdout にも流す）。
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

    # ── FD 診断レポート抽出 ──
    log_text = captured.getvalue()
    records = _parse_fd_reports(log_text)

    _summarize(records)

    # CSV 書き出し
    csv_path = f"/tmp/kc_fd_diag_19strand_{int(time.time())}.csv"
    _write_csv(records, csv_path)
    print(f"  → CSV 書き出し: {csv_path}")

    print()
    print("=" * 70)
    print("完了")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
