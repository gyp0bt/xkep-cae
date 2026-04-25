"""PairwiseFreezing Strategy — pair 単位接触凍結判定 (status-374 Phase 1).

[← contact freeze README](__init__.py) | [← project README](../../../README.md)

候補 (g3) pair-wise relaxation Phase 1。status-284 で導入された
**全体凍結モード**（NR 全反復で接触力ベクトル全体を snapshot 値に固定）を
**pair granularity** に拡張する。19 本 Type D stall（status-339/357/368）の
NR Type 分布が `D+E:67%, E:28%` で active 集合振動支配領域となるが、
全体凍結は不必要な安定 pair まで固定して有効自由度を不必要に削るため、
**振動 pair のみ freeze + 残り active 維持** が本候補の数理的アイデア。

数理（Phase 1 範囲）:
    入力: per-pair active 切替カウント `flip_counts[k]`、現/前反復 active 状態
    出力: per-pair 凍結フラグ `freeze_flags[k]` + Type D グローバル skip 判定
    判定:
        skip_global := skip_when_type_d_dominant ∧ "D" ∈ chattering_type
                       ∧ "A" ∉ chattering_type ∧ "B" ∉ chattering_type
                       ∧ "E" ∉ chattering_type
        freeze[k]   := (flip_counts[k] ≥ flip_threshold) ∧ is_active_now[k]
                       ∧ ¬ skip_global

Phase 1 は **判定責務のみ**。実際の f_c 部分組立 / NR 残差差し替えは Phase 2
（status-375 以降、`_newton_dynamic.py` への配線時に実施）に分離する。
ContactNormalDamping（status-365）と同じ Phase 1/2 分割パターン。

MCDD 脱法 pattern 対策:
    Phase 1 は frozen dataclass + 純計算で責務単一。`@verified_by` は Phase 2
    の solver 配線時に付与（単体実装時点では `TermExpansionContract` の K_c 5 項に
    所属しない — 凍結は NR 制御ロジックで力法則ではないため、ContactNormalDamping
    と同じく独立系統）。

参照:
    status-284 全体凍結モード初出（`_newton_dynamic.py` の `_freeze_active`/
    `_freeze_f_c`）/ status-368 Case B `chattering_freeze_nr_max=30` で 19 本
    +16.6%（frac=1.0 未達でクローズ）/ status-370 結果 B（K_c 機械精度 → 19 本
    stall は NR alg 側）/ phase_c3prime_19strand_plan.md §3.2 候補 (g3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class PairwiseFreezingInput:
    """pair 単位凍結判定の入力.

    Phase 1 は per-pair 信号から判定のみ行う純計算 Process。Phase 2 で NR
    ループ側が flip_counts を維持し各反復で本 Process を呼ぶ。

    Attributes:
        n_pairs: 接触ペア総数（INACTIVE 含む全ペア）
        pair_active_flip_counts: 各ペアの累積 active flip 回数 (n_pairs,) int.
            Phase 2 NR ループ側で
            `flip_counts += (is_active_now != is_active_prev)` と更新する。
            初期値 0、increment 単位でリセット推奨。
        is_active_now: 現反復での active/inactive 状態 (n_pairs,) bool.
            `pair.state.status != ContactStatus.INACTIVE` から構築。
        flip_threshold: 凍結発動閾値（flip_counts >= threshold で freeze 候補）。
            既定 3（NR 反復 ~3 回連続で振動）。
        chattering_type: グローバル分類文字列（`classify_chattering_type` の
            返り値、例: "D" / "D+E" / "A+B+E" / 空文字）。Phase 2 で NR ループ側
            から渡される。
        skip_when_type_d_dominant: Type D 単独支配時に凍結をスキップ
            （既定 True、status-288 の Type D 対策方針と整合 — D 単独は active
            集合安定なので freeze の意味がない）。
    """

    n_pairs: int
    pair_active_flip_counts: np.ndarray
    is_active_now: np.ndarray
    flip_threshold: int = 3
    chattering_type: str = ""
    skip_when_type_d_dominant: bool = True


@dataclass(frozen=True)
class PairwiseFreezingOutput:
    """pair 単位凍結判定の出力.

    Attributes:
        pair_freeze_flags: 各ペアを freeze するか (n_pairs,) bool.
            Phase 2 NR ループ側がこれを読み、True のペアに対して接触力を
            snapshot 値に差し替える（active 集合に含まれるが力は固定）。
        n_frozen: freeze=True のペア数（診断用）
        n_active_pairs: is_active_now=True のペア数（診断用、INACTIVE 除く）
        skip_freeze_global: True なら全ペア freeze=False。Type D 単独支配時に
            凍結スキップした場合に立つ（NR ループ側は別の対策＝NR 反復拡張等
            にフォールバックする想定、status-288 type_d_extra_attempts と整合）。
        freeze_reasons: 各ペアの判定理由文字列 (n_pairs,) tuple[str, ...]。
            "" / "skip_type_d" / "below_threshold" / "inactive" /
            "freeze_flips=N" のいずれか。診断ログに直接転記可能。
    """

    pair_freeze_flags: np.ndarray
    n_frozen: int
    n_active_pairs: int
    skip_freeze_global: bool
    freeze_reasons: tuple[str, ...]


# ── 純粋ヘルパ関数（Phase 2 から再利用想定） ───────────────


def _update_pair_active_flips(
    flip_counts_prev: np.ndarray,
    is_active_now: np.ndarray,
    is_active_prev: np.ndarray,
) -> np.ndarray:
    """前反復との比較で flip カウントをインクリメントする private 純関数.

    Phase 2 NR ループ側で各反復先頭で呼び出す想定。Phase 1 では本 Process の
    入力として既にインクリメント済みのカウントを受け取るため、本ヘルパは
    Phase 2 配線で利用する（C16 純関数滅菌のため private、`strategy` モジュール
    から直接 import する）。

    Args:
        flip_counts_prev: 前反復までの累積 flip カウント (n_pairs,) int
        is_active_now: 現反復の active 状態 (n_pairs,) bool
        is_active_prev: 前反復の active 状態 (n_pairs,) bool

    Returns:
        flip_counts_new: 現反復までの累積 flip カウント (n_pairs,) int.
            shape mismatch 時は flip_counts_prev をそのまま返す（safety）。
    """
    if flip_counts_prev.shape != is_active_now.shape:
        return flip_counts_prev
    if is_active_now.shape != is_active_prev.shape:
        return flip_counts_prev
    flipped = is_active_now != is_active_prev
    return flip_counts_prev + flipped.astype(flip_counts_prev.dtype)


def _is_type_d_dominant(chattering_type: str) -> bool:
    """Type D 単独支配を判定する private 純関数.

    `classify_chattering_type` の返り値（例 "D" / "D+E" / "A+B+E"）に対し、
    "D" が含まれかつ A/B/E が含まれない場合のみ True を返す（C16 純関数滅菌
    のため private、`strategy` モジュールから直接 import する）。

    Args:
        chattering_type: 分類文字列（"D" / "D+E" / "A+B+E" / "" 等）

    Returns:
        True if Type D 単独（"D" 含む∧A/B/E 全部含まない）
    """
    if not chattering_type or "D" not in chattering_type:
        return False
    for ch in ("A", "B", "E"):
        if ch in chattering_type:
            return False
    return True


# ── Process ────────────────────────────────────────────────


class PairwiseFreezingProcess(
    SolverProcess[PairwiseFreezingInput, PairwiseFreezingOutput],
):
    """pair 単位凍結判定 Process.

    Phase 1 責務: per-pair 入力信号 → per-pair 凍結フラグ + グローバル skip 判定.
    Phase 2 で `ContactFrictionProcess.freeze_slot` 経由で NR ループ
    `_newton_dynamic.py` に配線、freeze=True のペアは接触力を snapshot に
    固定（active 集合からは外さない＝ pair 単位 trust region）。

    判定アルゴリズム:
        1. Type D 単独支配時 (skip_when_type_d_dominant=True) → skip_freeze_global=True、
           全フラグ False
        2. 各ペアについて:
           - is_active_now[k] == False → freeze[k] = False, reason="inactive"
           - flip_counts[k] >= flip_threshold → freeze[k] = True,
             reason=f"freeze_flips={count}"
           - else → freeze[k] = False, reason="below_threshold"

    no-op 経路:
        - n_pairs == 0: 全フィールド空
        - chattering_type 単独 D + skip 有効: 全フラグ False、skip_freeze_global=True

    使用例（Phase 2 NR ループ内）:
        >>> # 各反復先頭で flip_counts 更新
        >>> flip_counts = _update_pair_active_flips(
        ...     flip_counts_prev, is_active_now, is_active_prev
        ... )
        >>> inp = PairwiseFreezingInput(
        ...     n_pairs=len(manager.pairs),
        ...     pair_active_flip_counts=flip_counts,
        ...     is_active_now=is_active_now,
        ...     flip_threshold=cfg.pairwise_freeze_flip_threshold,
        ...     chattering_type=classify_chattering_type(_nr_snap),
        ... )
        >>> out = PairwiseFreezingProcess().process(inp)
        >>> if not out.skip_freeze_global and out.n_frozen > 0:
        ...     # frozen ペアの接触力を snapshot に差し替え（Phase 2 で実装）
        ...     ...
    """

    meta = ProcessMeta(
        name="PairwiseFreezing",
        module="solve",
        version="1.0.0",
        document_path="docs/pairwise_freezing.md",
    )

    def process(self, input_data: PairwiseFreezingInput) -> PairwiseFreezingOutput:
        n = input_data.n_pairs

        # no-op: ペアなし
        if n == 0:
            return PairwiseFreezingOutput(
                pair_freeze_flags=np.zeros(0, dtype=bool),
                n_frozen=0,
                n_active_pairs=0,
                skip_freeze_global=False,
                freeze_reasons=(),
            )

        flip_counts = input_data.pair_active_flip_counts
        is_active = input_data.is_active_now

        # 入力 shape 整合性チェック（責務外だが defensive）
        if flip_counts.shape != (n,) or is_active.shape != (n,):
            raise ValueError(
                f"shape mismatch: n_pairs={n}, "
                f"flip_counts.shape={flip_counts.shape}, "
                f"is_active.shape={is_active.shape}"
            )

        n_active = int(np.count_nonzero(is_active))

        # Type D 単独支配 → グローバル skip
        if input_data.skip_when_type_d_dominant and _is_type_d_dominant(
            input_data.chattering_type,
        ):
            return PairwiseFreezingOutput(
                pair_freeze_flags=np.zeros(n, dtype=bool),
                n_frozen=0,
                n_active_pairs=n_active,
                skip_freeze_global=True,
                freeze_reasons=tuple("skip_type_d" for _ in range(n)),
            )

        # per-pair 判定
        threshold = input_data.flip_threshold
        flags = np.zeros(n, dtype=bool)
        reasons: list[str] = []

        for k in range(n):
            count_k = int(flip_counts[k])
            if not bool(is_active[k]):
                reasons.append("inactive")
                continue
            if count_k >= threshold:
                flags[k] = True
                reasons.append(f"freeze_flips={count_k}")
            else:
                reasons.append("below_threshold")

        return PairwiseFreezingOutput(
            pair_freeze_flags=flags,
            n_frozen=int(np.count_nonzero(flags)),
            n_active_pairs=n_active,
            skip_freeze_global=False,
            freeze_reasons=tuple(reasons),
        )
