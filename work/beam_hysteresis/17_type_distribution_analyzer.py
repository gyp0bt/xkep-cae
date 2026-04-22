"""work/beam_hysteresis/17_type_distribution_analyzer.py — NR Type 分布集計ツール.

[← README](README.md) | [← project README](../../README.md)

status-361: 7 本撚線と 19 本撚線の挙動反転（`smoothing_delta=1000` が 7 本で
frac=1.0 大幅改善、19 本で frac=-23.1% 退化）を定量化するため、`[NR診断]`
ログ行から Type 分布を集計する。

仕様:
- 入力: `[NR診断] Incr N (frac=…), … Type分布[A+B:1(9%), A+B+D.div:4(36%), ...], ...`
  形式のログ行
- 出力: (Type キー → 合計 attempts) の集計と、大分類（active_flip=C+E /
  tangent_inconsistent=D / mixed=C+D / others）での構成比

Type 記号の解釈（status-287/288）:
- A/B: 高残差収束経路（通常収束）
- C  : 低残差停滞（active flip の兆候 + 残差振動）
- D.div / D.slow / D.stall: 接線剛性不整合（div=発散, slow=線形低速, stall=停滞）
- E  : active 集合振動（接触 on/off の瞬発振動）

使用方法:
    python work/beam_hysteresis/17_type_distribution_analyzer.py <log_path> [<log_path2> ...]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

_TYPE_LINE = re.compile(r"Type分布\[([^\]]+)\]")
_TYPE_ENTRY = re.compile(r"([A-Z][A-Za-z.+]*):(\d+)\(\d+%\)")


def analyze(log_path: Path) -> dict[str, object]:
    """1 ログから Type 分布を集計."""
    counts: Counter[str] = Counter()
    total_incr = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _TYPE_LINE.search(line)
        if not m:
            continue
        total_incr += 1
        for entry_m in _TYPE_ENTRY.finditer(m.group(1)):
            key = entry_m.group(1)
            n = int(entry_m.group(2))
            counts[key] += n

    total_attempts = sum(counts.values())
    categories = {
        "active_flip (C/E 系)": 0,
        "tangent_inconsistent (D 系単独)": 0,
        "mixed (C+D 系)": 0,
        "high_residual (A/B 系)": 0,
    }
    for key, n in counts.items():
        has_a_or_b = "A" in key or "B" in key
        has_c = "C" in key and "D" not in key.split("+")[-1].replace(".div", "").replace(
            ".slow", ""
        ).replace(".stall", "")
        has_d = ".div" in key or ".slow" in key or ".stall" in key
        has_c_flag = "C" in key
        has_e_flag = "E" in key

        if has_a_or_b:
            categories["high_residual (A/B 系)"] += n
        elif has_d and has_c_flag:
            categories["mixed (C+D 系)"] += n
        elif has_d:
            categories["tangent_inconsistent (D 系単独)"] += n
        elif has_c_flag or has_e_flag:
            categories["active_flip (C/E 系)"] += n
        else:
            # どれにも該当しない（稀）
            categories["active_flip (C/E 系)"] += n
        _ = has_c  # unused silencer

    return {
        "path": str(log_path),
        "total_incr_diag": total_incr,
        "total_attempts": total_attempts,
        "counts": dict(counts),
        "categories": categories,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <log_path> [<log_path2> ...]")
        return 2

    summaries: list[dict[str, object]] = []
    for p in argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"ERROR: {path} が見つかりません", file=sys.stderr)
            continue
        summaries.append(analyze(path))

    if not summaries:
        return 1

    for s in summaries:
        print("=" * 70)
        print(f"  {s['path']}")
        print("=" * 70)
        print(f"  総 Incr 診断行数: {s['total_incr_diag']}")
        print(f"  総 attempts:     {s['total_attempts']}")
        print()
        print("  大分類:")
        total_attempts = int(s["total_attempts"] or 0)
        categories = s["categories"]
        assert isinstance(categories, dict)
        for label, n in categories.items():
            pct = (100.0 * n / total_attempts) if total_attempts > 0 else 0.0
            print(f"    {label:<38s}: {n:>5d} ({pct:5.1f}%)")
        print()
        print("  内訳（Type キーごとの attempts 合計）:")
        counts = s["counts"]
        assert isinstance(counts, dict)
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = (100.0 * n / total_attempts) if total_attempts > 0 else 0.0
            print(f"    {key:<24s}: {n:>5d} ({pct:5.1f}%)")
        print()

    # 比較表（複数ログ時）
    if len(summaries) >= 2:
        print("=" * 70)
        print("  比較表（大分類 %）")
        print("=" * 70)
        labels = list(summaries[0]["categories"].keys())  # type: ignore[index]
        header = f"  {'Category':<38s} " + " ".join(
            f"{Path(str(s['path'])).stem[:18]:>20s}" for s in summaries
        )
        print(header)
        for label in labels:
            pcts = []
            for s in summaries:
                total = int(s["total_attempts"] or 0)
                categories = s["categories"]
                assert isinstance(categories, dict)
                n = categories.get(label, 0)
                pct = (100.0 * n / total) if total > 0 else 0.0
                pcts.append(f"{pct:>19.1f}%")
            print(f"  {label:<38s} " + " ".join(pcts))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
