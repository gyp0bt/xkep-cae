"""Contact Damping サブパッケージ (status-365).

接触ペア単位の法線減衰力 `f_damp = -c_n * v_n * n̂` を組み立てる Process。
候補 (e) 接触減衰 escape hatch（status-363 §4、MCDD 凍結解除待機中課題 19本
frac=1.0 完走のための最後の escape hatch）の Phase 1 インフラ。

Phase 1（status-365、本パッケージ）:
- `ContactNormalDampingProcess`: ペア単位 f_damp + K_damp (= c_n * c1 * n̂⊗n̂)
  + E_damp 消散率の組み立て（単体 Process、solver 未配線）
- ユニットテストで物理整合性検証（単一ペア解析解 / ゼロ減衰 / 複数ペア重畳 /
  エネルギー消散率正値性）

Phase 2（status-366 予定）:
- `ContactFrictionProcess` への StrategySlot 追加（optional、default OFF）
- `_newton_dynamic.py` で tangent_components() と並行に f_damp/K_damp 加算
- `StrandBendingOscillationConfig.contact_damping_coefficient` を NR に連結
- `ContactDampingEnergyMonitorProcess` で 10 step 毎に E_damp/E_strain 出力
- 7本撚線で 1/2/5/10/20% 減衰 budget 実測 → 19本 Type D stall 検証

数理:
    f_damp[k,α] = -c_n * v_n * coeff_k * n̂[α]  （k ∈ A0,A1,B0,B1、α ∈ {x,y,z}）
    v_n = g_shape · v_local = Σ coeff_k * n̂[α] * v[node_k, α]
    K_damp[k,l]_{αβ} = c_n * c1 * coeff_k * coeff_l * n̂[α] * n̂[β]
                    = c_n * c1 * (g_shape ⊗ g_shape)
    E_damp_rate = Σ c_n * v_n² （単位時間あたり消散エネルギー、常に ≥ 0）

coeff_k = [(1-s), s, -(1-t), -t]（線形形状関数）は `HuberContactForceProcess.
_contact_shape_vector` と同符号（ペア A→B に対し n̂ が外向き）。

c1 = γ/(β·dt) は Generalized-α の速度-変位感度で、effective_stiffness()
での C 寄与係数と整合する（docs/contact_friction.md の時間積分節参照）。
"""

from xkep_cae.contact.damping.strategy import (
    ContactNormalDampingInput,
    ContactNormalDampingOutput,
    ContactNormalDampingProcess,
)

__all__ = [
    "ContactNormalDampingInput",
    "ContactNormalDampingOutput",
    "ContactNormalDampingProcess",
]
