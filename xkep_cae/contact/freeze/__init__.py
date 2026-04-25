"""Contact Freeze サブパッケージ (status-374).

[← contact README](../__init__.py) | [← project README](../../../README.md)

候補 (g3) pair-wise relaxation Phase 1。status-284 で導入された **全体凍結
モード**（NR 全反復で接触力ベクトル全体を snapshot 値に固定）を pair
granularity に拡張する。19 本 Type D stall（status-339/357/368）の
NR Type 分布が `D+E:67%, E:28%` で active 集合振動支配領域となるが、全体
凍結は不必要な安定 pair まで固定して有効自由度を不必要に削るため、振動
pair のみ freeze + 残り active 維持 が本候補の数理的アイデア。

Phase 1（status-374、本パッケージ）:
- `PairwiseFreezingProcess`: per-pair active flip 履歴 → per-pair 凍結フラグ
  + Type D 単独支配時の グローバル skip 判定（単体 Process、solver 未配線）
- ヘルパ純関数 `_update_pair_active_flips` / `_is_type_d_dominant`（C16
  滅菌のため private、Phase 2 NR ループから `strategy` モジュール直接 import）
- ユニットテストで判定ロジック検証（フラグ整合性 / Type D skip / shape ガード）

Phase 2（status-375 以降予定）:
- `ContactFrictionProcess.freeze_slot` への StrategySlot 配線（optional、
  default OFF）
- `_newton_dynamic.py` の既存全体凍結 (`_freeze_active`/`_freeze_f_c`) を
  pair 単位に拡張: freeze=True ペアのみ snapshot 力に差し替え
- `StrandBendingOscillationConfig.pairwise_freeze_*` 3 field 追加（enabled
  / flip_threshold / skip_when_type_d_dominant）+ NR plumb-through
- 7 本撚線 frac=1.0 維持回帰 + 19 本 Type D stall で frac ≥ 0.6 検証（gate）

数理:
    skip_global := skip_when_type_d_dominant ∧ _is_type_d_dominant(chattering)
    freeze[k]   := (flip_counts[k] ≥ flip_threshold) ∧ is_active_now[k]
                   ∧ ¬ skip_global

    flip_counts は Phase 2 NR ループ側が
    `flip_counts += (is_active_now != is_active_prev)` でインクリメント、
    本 Process は決定論的な判定純計算のみ担う。
"""

from xkep_cae.contact.freeze.strategy import (
    PairwiseFreezingInput,
    PairwiseFreezingOutput,
    PairwiseFreezingProcess,
)

__all__ = [
    "PairwiseFreezingInput",
    "PairwiseFreezingOutput",
    "PairwiseFreezingProcess",
]
