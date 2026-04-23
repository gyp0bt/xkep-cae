"""ContactNormalDamping Strategy — 接触法線減衰 (status-365 Phase 1).

接触ペア p = (elem_a, elem_b, nodes_a=[A0, A1], nodes_b=[B0, B1]) について、
法線方向の相対速度 v_n を数値的に減衰させる力 `f_damp = -c_n * v_n * n̂` を
組み立てる Process。

背景:
    status-363 の 4 ケース感度掃引で「BT line search は active 集合振動を
    根本抑制できない」と確定した。次候補 (e) は Type D stall の震源である
    active×mixed 領域に対し **escape hatch** として微小な粘性を入れる手法。
    Generalized-α の C 行列に直接足すのではなく、接触ペア単位で組み立てた
    `f_damp + K_damp` を NR 残差/接線剛性に加算する（時間積分モジュール無変更）。

数理:
    形状係数 coeff = [(1-s), s, -(1-t), -t]   （HuberContactForce と同一）
    g_shape (12,) = [coeff_0·n̂, coeff_1·n̂, coeff_2·n̂, coeff_3·n̂]
    v_n = g_shape · v_local
    f_damp_local = -c_n * v_n * g_shape                        (12,)
    K_damp_local = c_n * c1 * (g_shape ⊗ g_shape)              (12, 12)
    E_damp_rate  = Σ_active c_n * v_n²                         (>=0)

c1 = γ/(β·dt) は Generalized-α 速度-変位感度（`effective_stiffness` で
`(1 - α_f) * c1 * C` と組み合わされる係数）。呼び出し側が dt 反映済みで
渡す設計にして、本 Process は時間積分スキームに依存しない。

MCDD 脱法 pattern 対策:
- Phase 1 本 Process は frozen dataclass + 純計算で責務単一
- `@verified_by` は Phase 2 の solver 配線時に付与（単体実装時点では
  term 契約に所属しない — TermExpansionContract の K_c 5 項は解析的接線の
  材料・幾何項であり、減衰項はパイプライン上別系統のため）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._types import ContactStatus
from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class ContactNormalDampingInput:
    """接触法線減衰の入力.

    Attributes:
        pairs: 接触ペアのリスト（`_ContactPairOutput` 互換、status フィールド
            が `ContactStatus.INACTIVE` 以外のものだけ減衰力を組み立てる）
        velocity: 全体速度ベクトル (ndof_total,)。Generalized-α `vel` 属性を
            そのまま渡す。
        c_n: 法線減衰係数（c_n >= 0、0 なら no-op）
        stiffness_factor: c1 = γ/(β·dt) — Generalized-α の速度-変位感度。
            呼び出し側で time_integration_strategy から取得して渡す。
        ndof_total: 全体 DOF 数
        ndof_per_node: ノードあたり DOF 数（撚線梁は 6、変位は先頭 3）
    """

    pairs: list
    velocity: np.ndarray
    c_n: float
    stiffness_factor: float
    ndof_total: int
    ndof_per_node: int = 6


@dataclass(frozen=True)
class ContactNormalDampingOutput:
    """接触法線減衰の出力.

    Attributes:
        f_damp: 全体減衰力ベクトル (ndof_total,)。NR 残差に **加算** する
            （`R_eff += f_damp` の向き。物理符号は負で `-c_n v_n n̂` だが、
            本フィールドにはその符号で格納済み）
        K_damp: 全体減衰接線剛性 (ndof_total, ndof_total) CSR。NR K に
            **加算** する（ `K_eff += K_damp`）。常に対称半正定値。
        energy_rate: 瞬時消散エネルギー率 Σ c_n v_n² [力·距離/時間] = [エネルギー/時間]。
            dt 乗算は呼び出し側（Phase 2 の Energy Monitor）で行う。
        n_active_pairs: 組み立て対象となった active ペア数（診断用）
    """

    f_damp: np.ndarray
    K_damp: sp.csr_matrix
    energy_rate: float
    n_active_pairs: int


# ── Process ────────────────────────────────────────────────


class ContactNormalDampingProcess(
    SolverProcess[ContactNormalDampingInput, ContactNormalDampingOutput],
):
    """接触法線減衰力 f_damp + K_damp を組み立てる Process.

    線形形状関数 (1-s), s, -(1-t), -t を使用。Hermite 高次基底は Phase 2
    以降で拡張（K_damp の (g_shape ⊗ g_shape) 構造は基底選択に依らない）。

    no-op 条件:
    - `c_n == 0`: f_damp / K_damp ともゼロ、energy_rate = 0、n_active_pairs = 0
    - active ペアなし: 同上

    使用例（Phase 2 NR ループ内）:
        >>> inp = ContactNormalDampingInput(
        ...     pairs=manager.pairs,
        ...     velocity=time_integ.vel,
        ...     c_n=cfg.contact_damping_coefficient,
        ...     stiffness_factor=time_integ.gamma / (time_integ.beta * dt),
        ...     ndof_total=len(u),
        ... )
        >>> out = ContactNormalDampingProcess().process(inp)
        >>> R_eff += out.f_damp
        >>> K_eff += out.K_damp
    """

    meta = ProcessMeta(
        name="ContactNormalDamping",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_damping.md",
    )

    def process(self, input_data: ContactNormalDampingInput) -> ContactNormalDampingOutput:
        ndof = input_data.ndof_total
        ndpn = input_data.ndof_per_node
        c_n = input_data.c_n
        c1 = input_data.stiffness_factor

        f_damp = np.zeros(ndof)

        # no-op 経路: c_n=0 は必ず空の CSR を返す（K_damp.nnz == 0）
        if c_n == 0.0 or not input_data.pairs:
            K_empty = sp.csr_matrix((ndof, ndof))
            return ContactNormalDampingOutput(
                f_damp=f_damp,
                K_damp=K_empty,
                energy_rate=0.0,
                n_active_pairs=0,
            )

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []
        energy_rate = 0.0
        n_active = 0

        vel = input_data.velocity

        for p in input_data.pairs:
            if p.state.status == ContactStatus.INACTIVE:
                continue

            s = p.state.s
            t = p.state.t
            n_hat = np.asarray(p.state.normal, dtype=float)

            # 形状係数: A0, A1, B0, B1（HuberContactForce と同符号、n̂ は A→B 外向き）
            coeffs = np.array([1.0 - s, s, -(1.0 - t), -t], dtype=float)

            # g_shape (12,) = [coeff_k * n̂ for k in {A0,A1,B0,B1}]
            g_shape = np.empty(12, dtype=float)
            g_shape[0:3] = coeffs[0] * n_hat
            g_shape[3:6] = coeffs[1] * n_hat
            g_shape[6:9] = coeffs[2] * n_hat
            g_shape[9:12] = coeffs[3] * n_hat

            # 各ノードの先頭 3 DOF（変位成分）を拾う
            nodes = (
                int(p.nodes_a[0]),
                int(p.nodes_a[1]),
                int(p.nodes_b[0]),
                int(p.nodes_b[1]),
            )
            global_dofs = np.empty(12, dtype=np.int64)
            for k in range(4):
                base = nodes[k] * ndpn
                global_dofs[k * 3 + 0] = base + 0
                global_dofs[k * 3 + 1] = base + 1
                global_dofs[k * 3 + 2] = base + 2

            v_local = vel[global_dofs]
            v_n = float(g_shape @ v_local)

            # f_damp_local = -c_n * v_n * g_shape
            f_damp_local = (-c_n * v_n) * g_shape
            np.add.at(f_damp, global_dofs, f_damp_local)

            # K_damp_local (12,12) = c_n * c1 * (g_shape ⊗ g_shape)
            if c1 != 0.0:
                K_local = (c_n * c1) * np.outer(g_shape, g_shape)
                # COO 構築: np.repeat / np.tile で 12x12 ブロック
                rows_block = np.repeat(global_dofs, 12)
                cols_block = np.tile(global_dofs, 12)
                vals_block = K_local.reshape(-1)
                rows_list.append(rows_block)
                cols_list.append(cols_block)
                vals_list.append(vals_block)

            # E_damp_rate: c_n * v_n² （瞬時消散率、dt は呼び出し側で乗算）
            energy_rate += c_n * v_n * v_n
            n_active += 1

        if rows_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            vals = np.concatenate(vals_list)
            K_damp = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
        else:
            K_damp = sp.csr_matrix((ndof, ndof))

        return ContactNormalDampingOutput(
            f_damp=f_damp,
            K_damp=K_damp,
            energy_rate=energy_rate,
            n_active_pairs=n_active,
        )
