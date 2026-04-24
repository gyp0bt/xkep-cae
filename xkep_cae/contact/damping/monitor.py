"""ContactDampingEnergyMonitor — 接触法線減衰 E_damp / E_strain 監査.

status-366 Phase 2: 候補 (e) 接触減衰 escape hatch（status-363 §4）用の
PostProcess。ソルバー完了後に `SolverResultData.damping_energy_history` と
`EnergyHistory` を付き合わせ、E_damp_cumulative / E_strain の比を
インクリメント毎に評価する。

評価基準:
    budget_ratio が 0 超で、比がそれを上回るインクリメントが 1 件でもあれば
    `n_violations > 0`、`max_ratio > budget_ratio` で報告する。
    budget_ratio=0（default）ならチェック無効化（`n_violations=0` 固定、max_ratio のみ出力）。

設計指針:
    本 Process は非侵襲監査（読み取り専用）。既存ソルバーの減衰配線は
    `_newton_dynamic.py`（f_damp / K_damp 加算）と `ContactFrictionProcess`
    （E_damp_cumulative 累積）で完結しており、本 Process の出力は
    validation スクリプトが budget 超過を検出するための窓口。

MCDD との関係:
    本 Process は `TermExpansionContract("K_c_term_expansion")` の 5 項分解
    とは独立系統（K_c の解析的接線拡張ではなく Generalized-α の C 行列経路
    の代替）。`@verified_by` 紐付けは不要。
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.core import PostProcess, ProcessMeta


@dataclass(frozen=True)
class ContactDampingEnergyMonitorInput:
    """E_damp 監査の入力.

    Attributes:
        damping_energy_history: `SolverResultData.damping_energy_history` を
            そのまま渡す。各要素は `(load_frac, E_damp_cumulative)` のタプル。
        energy_history: `SolverResultData.energy_history` を渡す。各エントリ
            の `strain_energy` を分母に使用（`_EnergyHistoryAccumulator` 互換）。
        budget_ratio: 許容上限 E_damp_cumulative / E_strain。0 なら検査無効。
        log_every_n_steps: 出力頻度（1=全インクリメント、10=10 step 毎）。
            負値/0 で出力抑制。
    """

    damping_energy_history: tuple
    energy_history: object
    budget_ratio: float = 0.0
    log_every_n_steps: int = 10


@dataclass(frozen=True)
class ContactDampingEnergyMonitorOutput:
    """E_damp 監査の出力.

    Attributes:
        max_ratio: E_damp_cumulative / E_strain の最大値（全インクリメント）
        final_ratio: 最終インクリメントの E_damp_cumulative / E_strain
        final_damping_energy: E_damp_cumulative 最終値
        final_strain_energy: E_strain 最終値
        n_violations: budget_ratio > 0 のとき、比が超過したインクリメント数
        budget_violated: n_violations > 0 と同義（ショートカット用）
        report: 人間向けサマリ（print で出力可能なテキスト）
    """

    max_ratio: float
    final_ratio: float
    final_damping_energy: float
    final_strain_energy: float
    n_violations: int
    budget_violated: bool
    report: str


class ContactDampingEnergyMonitorProcess(
    PostProcess[
        ContactDampingEnergyMonitorInput,
        ContactDampingEnergyMonitorOutput,
    ],
):
    """接触法線減衰のエネルギー消費を監査する PostProcess.

    Phase 2 validation スクリプト（`work/beam_hysteresis/23_contact_damping_*`）
    から呼ばれ、E_damp / E_strain が budget を超えた場合に警告表示する。

    使用例:
        >>> inp = ContactDampingEnergyMonitorInput(
        ...     damping_energy_history=result.damping_energy_history,
        ...     energy_history=result.energy_history,
        ...     budget_ratio=0.10,
        ... )
        >>> out = ContactDampingEnergyMonitorProcess().process(inp)
        >>> print(out.report)
        >>> assert not out.budget_violated, out.report
    """

    meta = ProcessMeta(
        name="ContactDampingEnergyMonitor",
        module="postprocess",
        version="1.0.0",
        document_path="docs/contact_damping.md",
    )

    def process(
        self, input_data: ContactDampingEnergyMonitorInput
    ) -> ContactDampingEnergyMonitorOutput:
        damp_hist = input_data.damping_energy_history
        budget = input_data.budget_ratio

        # 空入力の short-circuit
        if not damp_hist:
            return ContactDampingEnergyMonitorOutput(
                max_ratio=0.0,
                final_ratio=0.0,
                final_damping_energy=0.0,
                final_strain_energy=0.0,
                n_violations=0,
                budget_violated=False,
                report="[DampingMonitor] E_damp_history が空（c_n=0 無効化ケース）",
            )

        # EnergyHistory の step → strain_energy マップを構築
        entries = getattr(input_data.energy_history, "entries", ()) or ()
        strain_map: dict[int, float] = {}
        for e in entries:
            strain_map[getattr(e, "step", 0)] = float(getattr(e, "strain_energy", 0.0))

        # damp_hist 側は (load_frac, E_damp_cum) のみで step 番号を保持しない。
        # インクリメント順が一致するので zip で並列走査、超過分の strain は
        # strain 履歴の最終値で代用する（E_damp と E_strain は同じ step で対応）。
        strain_list: list[float] = [float(getattr(e, "strain_energy", 0.0)) for e in entries]
        max_ratio = 0.0
        n_violations = 0
        lines: list[str] = []

        # strain が不足している時は最後の値をパディング（E_strain は単調に
        # 変動するため近傍値で十分な精度）。
        last_strain_fallback = strain_list[-1] if strain_list else 0.0

        n_hist = len(damp_hist)
        log_every = input_data.log_every_n_steps
        for idx, (load_frac, e_damp) in enumerate(damp_hist):
            if idx < len(strain_list):
                e_strain = strain_list[idx]
            else:
                e_strain = last_strain_fallback
            e_strain_abs = abs(e_strain)
            ratio = e_damp / e_strain_abs if e_strain_abs > 1e-30 else 0.0
            if ratio > max_ratio:
                max_ratio = ratio
            violated = budget > 0.0 and ratio > budget
            if violated:
                n_violations += 1
            do_log = log_every > 0 and (idx % log_every == 0 or idx == n_hist - 1 or violated)
            if do_log:
                tag = "!" if violated else " "
                lines.append(
                    f"  [DampingMonitor]{tag} frac={load_frac:.4f} "
                    f"E_damp={e_damp:.3e} E_strain={e_strain:.3e} "
                    f"ratio={ratio:.3e}"
                )

        final_frac, final_damp = damp_hist[-1]
        final_strain = strain_list[-1] if strain_list else 0.0
        final_ratio = final_damp / abs(final_strain) if abs(final_strain) > 1e-30 else 0.0

        header = (
            f"[DampingMonitor] summary: n_steps={n_hist}, "
            f"max_ratio={max_ratio:.3e}, final_ratio={final_ratio:.3e}, "
            f"budget={budget:.3e}, violations={n_violations}"
        )
        report = "\n".join([header, *lines])

        return ContactDampingEnergyMonitorOutput(
            max_ratio=max_ratio,
            final_ratio=final_ratio,
            final_damping_energy=final_damp,
            final_strain_energy=final_strain,
            n_violations=n_violations,
            budget_violated=n_violations > 0,
            report=report,
        )
