"""ステップエネルギー診断 Process.

動的解析のステップごとのエネルギー収支を計算する。
運動エネルギー・ひずみエネルギー・外力仕事・接触散逸を追跡し、
エネルギー保存性を定量評価する。

設計仕様:
  KE = 0.5 * v^T M v
  SE = 0.5 * u^T f_int  (内力仕事近似)
  W_ext = f_ext · u
  W_contact = f_c · u
  energy_balance = |KE + SE - W_ext - W_contact| / max(KE + SE, 1e-30)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import ProcessMeta, SolverProcess


@dataclass(frozen=True)
class StepEnergyInput:
    """ステップエネルギー診断の入力."""

    u: np.ndarray
    velocity: np.ndarray
    mass_matrix: sp.spmatrix | np.ndarray
    f_int: np.ndarray
    f_ext: np.ndarray
    f_c: np.ndarray
    dt: float
    step: int


@dataclass(frozen=True)
class StepEnergyOutput:
    """ステップエネルギー診断の出力."""

    kinetic_energy: float
    strain_energy: float
    external_work: float
    contact_work: float
    total_energy: float
    energy_ratio: float
    step: int


class StepEnergyDiagnosticsProcess(
    SolverProcess[StepEnergyInput, StepEnergyOutput],
):
    """ステップごとのエネルギー収支診断.

    動的解析における各ステップのエネルギー項を計算し、
    エネルギー保存性を定量評価する。

    エネルギー比率 energy_ratio は 1.0 に近いほどエネルギー保存が良好。
    数値減衰（rho_inf < 1）の場合は energy_ratio < 1.0 が正常。
    """

    meta = ProcessMeta(
        name="StepEnergyDiagnostics",
        module="solve",
        version="1.0.0",
        document_path="docs/energy_diagnostics.md",
    )

    def process(self, input_data: StepEnergyInput) -> StepEnergyOutput:
        """エネルギー収支を計算する."""
        u = input_data.u
        v = input_data.velocity
        M = input_data.mass_matrix

        # 運動エネルギー: KE = 0.5 * v^T M v
        if sp.issparse(M):
            Mv = M.dot(v)
        else:
            Mv = M @ v
        ke = 0.5 * float(np.dot(v, Mv))

        # ひずみエネルギー（内力仕事近似）: SE = 0.5 * u^T f_int
        se = 0.5 * float(np.dot(u, input_data.f_int))

        # 外力仕事: W_ext = f_ext · u
        w_ext = float(np.dot(input_data.f_ext, u))

        # 接触仕事: W_contact = f_c · u
        w_contact = float(np.dot(input_data.f_c, u))

        # 総エネルギー（保存量の近似）
        total = ke + se

        # エネルギー比率: 初期エネルギーに対する現在のエネルギー
        # 外力なし自由振動: total/initial ≈ 1.0 (数値減衰分だけ < 1.0)
        energy_ratio = total / max(abs(w_ext) + abs(w_contact) + abs(total), 1e-30)

        return StepEnergyOutput(
            kinetic_energy=ke,
            strain_energy=se,
            external_work=w_ext,
            contact_work=w_contact,
            total_energy=total,
            energy_ratio=energy_ratio,
            step=input_data.step,
        )


@dataclass(frozen=True)
class EnergyHistoryEntryOutput:
    """エネルギー履歴の1エントリ."""

    step: int
    time: float
    kinetic_energy: float
    strain_energy: float
    external_work: float
    contact_work: float
    total_energy: float
    energy_ratio: float


# C17: エネルギー履歴蓄積器は可変状態を持つため Process 外部ユーティリティ
# __init__.py で re-export しないプライベートクラスとする
class _EnergyHistoryAccumulator:
    """エネルギー履歴の蓄積器（プライベート）."""

    def __init__(self) -> None:
        self.entries: list[EnergyHistoryEntryOutput] = []
        self.initial_energy: float = 0.0

    def append(self, entry: EnergyHistoryEntryOutput) -> None:
        """エントリを追加."""
        if len(self.entries) == 0:
            self.initial_energy = entry.total_energy
        self.entries.append(entry)

    @property
    def decay_ratio(self) -> float:
        """エネルギー減衰率: E_final / E_initial."""
        if len(self.entries) < 2 or abs(self.initial_energy) < 1e-30:
            return 1.0
        return self.entries[-1].total_energy / self.initial_energy

    def summary(self) -> str:
        """エネルギー履歴のサマリ文字列."""
        if not self.entries:
            return "エネルギー履歴: なし"
        e0 = self.entries[0]
        ef = self.entries[-1]
        lines = [
            "=" * 50,
            "  エネルギー収支サマリ",
            "=" * 50,
            f"  ステップ数: {len(self.entries)}",
            f"  初期 KE: {e0.kinetic_energy:.6e}",
            f"  初期 SE: {e0.strain_energy:.6e}",
            f"  初期 Total: {e0.total_energy:.6e}",
            f"  最終 KE: {ef.kinetic_energy:.6e}",
            f"  最終 SE: {ef.strain_energy:.6e}",
            f"  最終 Total: {ef.total_energy:.6e}",
            f"  エネルギー減衰率: {self.decay_ratio:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# 後方互換エイリアス
EnergyHistoryEntry = EnergyHistoryEntryOutput
EnergyHistory = _EnergyHistoryAccumulator
