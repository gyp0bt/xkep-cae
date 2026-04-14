"""ContactPairAnalysisProcess — 接触ペア履歴の後処理 Process.

status-333 で追加された `contact_pair_history`（各収束インクリメントの活性接触ペア
スナップショット列）を解析し、κ_cr（スティック→スリップ遷移曲率）分布・
素線間累積散逸・活性ペア数推移を抽出する。

status-336 までに整備した M-κ ヒステリシスループ計算を補完し、ファイバー梁
キャリブレーションに必要な **接触レベル** の観測量を提供する。

物理的背景:
    撚線の曲げヒステリシスは、各接触ペアが曲率増大に伴い個別の κ_cr で
    スティック→スリップに遷移することで生じる。ピッチ・曲率振幅・摩擦係数
    により κ_cr 分布が変化し、これが M-κ ループのティアドロップ形状を決定する。

    本 Process は ContactFrictionProcess が出力する履歴から、
    - 各ペア (elem_a, elem_b) の初回スリップ発生時の κ（= κ_cr）
    - 各ペアの最終累積散逸エネルギー
    - インクリメント毎の活性ペア数
    を抽出し、Papailiou モデル / ファイバー梁モデルの校正データとする。

status-337: 接触ペアスナップショット後処理 Process 化。

[← README](../../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from xkep_cae.core import PostProcess, ProcessMeta

if TYPE_CHECKING:
    from xkep_cae.core.data import ContactPairSnapshotEntry

# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class ContactPairAnalysisInput:
    """接触ペア解析入力.

    Attributes:
        contact_pair_history: ((load_frac, (ContactPairSnapshotEntry, ...)), ...) 列。
            `SolverResultData.contact_pair_history` から直接渡す想定。
        moment_curvature_history: ((κ_i, M_i), ...) 列。`contact_pair_history` と
            同じ収束インクリメント順で対応付ける（長さは片方が空でも可）。
        slip_threshold: |(slip_s, slip_t)| > slip_threshold でスリップ検知。
            デフォルト 1e-6 は rz_t の数値雑音を除外するため。
    """

    contact_pair_history: tuple
    moment_curvature_history: tuple = ()
    slip_threshold: float = 1.0e-6


@dataclass(frozen=True)
class ContactPairAnalysisResult:
    """接触ペア解析結果.

    Attributes:
        n_steps: 解析対象のインクリメント数（contact_pair_history のエントリ数）
        n_unique_pairs: 履歴内で1度以上活性化した (elem_a, elem_b) の総数
        n_active_per_step: 各インクリメントの活性ペア数
        load_frac_per_step: 各インクリメントの load_frac
        kappa_cr_per_pair: (elem_a, elem_b) → κ_cr（初回スリップ曲率）。
            スリップ未観測ペアは含まない。moment_curvature_history が空の場合は
            代わりに load_frac を格納する。
        per_pair_dissipation: (elem_a, elem_b) → 最終 dissipation（履歴最終値）
        total_dissipation: 全ペアの最終累積散逸合計
        kappa_cr_mean: κ_cr の平均（スリップ観測ペアのみ）
        kappa_cr_std: κ_cr の標準偏差
        kappa_cr_min: κ_cr の最小値
        kappa_cr_max: κ_cr の最大値
        n_slipped_pairs: スリップが1度以上観測されたペア数
    """

    n_steps: int
    n_unique_pairs: int
    n_active_per_step: tuple[int, ...]
    load_frac_per_step: tuple[float, ...]
    kappa_cr_per_pair: dict = field(default_factory=dict)
    per_pair_dissipation: dict = field(default_factory=dict)
    total_dissipation: float = 0.0
    kappa_cr_mean: float = 0.0
    kappa_cr_std: float = 0.0
    kappa_cr_min: float = 0.0
    kappa_cr_max: float = 0.0
    n_slipped_pairs: int = 0


# ====================================================================
# 純粋ヘルパー関数
# ====================================================================


def _slip_magnitude(entry: ContactPairSnapshotEntry) -> float:
    """接線スリップの大きさ |(slip_s, slip_t)|."""
    return math.hypot(entry.slip_s, entry.slip_t)


def _kappa_for_step(
    mk_history: tuple,
    step_idx: int,
    load_frac: float,
) -> float:
    """step_idx に対応する κ を返す。mk_history が空なら load_frac を返す."""
    if not mk_history:
        return load_frac
    if step_idx < len(mk_history):
        return float(mk_history[step_idx][0])
    # 長さ不一致時は末尾を返す（保守的）
    return float(mk_history[-1][0])


# ====================================================================
# Process
# ====================================================================


class ContactPairAnalysisProcess(
    PostProcess[ContactPairAnalysisInput, ContactPairAnalysisResult],
):
    """接触ペア履歴の後処理 Process.

    `contact_pair_history` を走査し、各ペア (elem_a, elem_b) の:
      - 初回スリップ曲率 κ_cr（または初回スリップ load_frac）
      - 最終累積散逸エネルギー
    を抽出し、併せて活性ペア数の時系列と分布統計を返す。

    κ_cr 判定:
      `stick==False` かつ `|(slip_s, slip_t)| > slip_threshold` を満たす
      最初のインクリメントで記録。以降のスナップショットは無視（最初の遷移のみ）。

    設計意図:
      - M-κ ヒステリシスループ（status-336）は集約量で、微視的な素線間滑りの
        分布は見えない。本 Process は `contact_pair_history` の生データを
        ファイバー梁 / Papailiou モデルの校正指標に変換する。
      - 「κ_cr 分布」は CLAUDE.md の「CR梁接触動解析での直接検証（最優先）」の
        具体アウトプット。
    """

    meta = ProcessMeta(
        name="ContactPairAnalysis",
        module="post",
        version="1.0.0",
        document_path="docs/contact_pair_analysis.md",
    )
    uses = ()

    def process(self, input_data: ContactPairAnalysisInput) -> ContactPairAnalysisResult:
        """接触ペア履歴を解析し、κ_cr 分布・散逸統計を返す."""
        history = input_data.contact_pair_history
        mk_history = input_data.moment_curvature_history
        slip_threshold = input_data.slip_threshold

        n_steps = len(history)

        # 空履歴のショートカット
        if n_steps == 0:
            return ContactPairAnalysisResult(
                n_steps=0,
                n_unique_pairs=0,
                n_active_per_step=(),
                load_frac_per_step=(),
            )

        n_active_per_step: list[int] = []
        load_frac_per_step: list[float] = []
        seen_pairs: set[tuple[int, int]] = set()
        kappa_cr: dict[tuple[int, int], float] = {}
        last_dissipation: dict[tuple[int, int], float] = {}

        for step_idx, (load_frac, entries) in enumerate(history):
            load_frac_per_step.append(float(load_frac))
            n_active_per_step.append(len(entries))
            kappa = _kappa_for_step(mk_history, step_idx, load_frac)

            for entry in entries:
                key = (entry.elem_a, entry.elem_b)
                seen_pairs.add(key)
                last_dissipation[key] = float(entry.dissipation)

                # κ_cr 判定: 初回スリップ時の κ を記録
                if key in kappa_cr:
                    continue
                if entry.stick:
                    continue
                if _slip_magnitude(entry) <= slip_threshold:
                    continue
                kappa_cr[key] = float(kappa)

        # 散逸統計
        total_dissipation = float(sum(last_dissipation.values()))

        # κ_cr 分布統計
        kappa_cr_vals = list(kappa_cr.values())
        n_slipped = len(kappa_cr_vals)
        if n_slipped > 0:
            mean_k = sum(kappa_cr_vals) / n_slipped
            var_k = sum((k - mean_k) ** 2 for k in kappa_cr_vals) / n_slipped
            std_k = math.sqrt(var_k)
            min_k = min(kappa_cr_vals)
            max_k = max(kappa_cr_vals)
        else:
            mean_k = std_k = min_k = max_k = 0.0

        return ContactPairAnalysisResult(
            n_steps=n_steps,
            n_unique_pairs=len(seen_pairs),
            n_active_per_step=tuple(n_active_per_step),
            load_frac_per_step=tuple(load_frac_per_step),
            kappa_cr_per_pair=dict(kappa_cr),
            per_pair_dissipation=dict(last_dissipation),
            total_dissipation=total_dissipation,
            kappa_cr_mean=float(mean_k),
            kappa_cr_std=float(std_k),
            kappa_cr_min=float(min_k),
            kappa_cr_max=float(max_k),
            n_slipped_pairs=n_slipped,
        )
