"""ContactKcComponentFDDiagnosticProcess — K_c を K_mat / K_geo / K_st に分解した FD 診断.

status-342 推奨アクション 1（最優先）として新設。

`TangentFDDiagnosticProcess`（status-256/290）は K_c = K_mat - K_geo + K_st
の合成接線に対する単一の FD 相対誤差しか報告しない。一方、
`ContactForceStrategy.tangent_components()` は K_mat / K_geo / K_st を個別に
返せるため、本 Process では FD `∂f_c/∂u·du` と各部分行列 `K_i @ du` を
組み合わせごとに比較し、**x 成分 68% 不整合（status-342）がどの部分行列に
由来するか**を切り分ける。

**設計意図**
- status-295 の手続き（`K_c_adj mat-only` vs `K_st_adj` ON の比較）を
  任意の (u, du) 断面で自動化する
- mat-only / mat-geo / mat-st / full の 4 組み合わせを FD と突合
- 各組み合わせで comp 別（x/y/z/θx/θy/θz）不整合シェアを報告

**利用例**（work スクリプトや solver フック経由）::

    proc = ContactKcComponentFDDiagnosticProcess()
    K_mat, K_geo, K_st = strategy.tangent_components(u, manager, k_pen, ...)
    out = proc.process(
        ContactKcComponentFDDiagnosticInput(
            u=u,
            du=du,
            compute_contact_force=lambda uu: strategy.evaluate(uu, manager, k_pen)[0],
            K_mat=K_mat, K_geo=K_geo, K_st=K_st,
            eps=1e-7,
        )
    )
    print(out.report)

設計仕様: docs/verify.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import ProcessMeta, SolverProcess

# 成分名（6DOF/node: 並進3 + 回転3）
_COMP_NAMES = ("x", "y", "z", "tx", "ty", "tz")
_NDPN = 6  # ndof_per_node


@dataclass(frozen=True)
class ContactKcComponentFDDiagnosticInput:
    """K_c 成分分解 FD 診断の入力.

    Attributes:
        u: 現在の変位ベクトル（shape (ndof,)）
        du: NR 方向ベクトル（shape (ndof,)）
        compute_contact_force: u → f_c（接触力ベクトル、shape (ndof,)）を返す callable
        K_mat: 材料剛性 K_mat（shape (ndof, ndof), sparse）
        K_geo: 幾何剛性 K_geo（shape (ndof, ndof), sparse）。K_c = K_mat - K_geo + K_st
        K_st: 滑り剛性 K_st（shape (ndof, ndof), sparse）
        eps: FD 摂動幅
        label: 報告用ラベル（任意、trigger idx 等）
    """

    u: np.ndarray
    du: np.ndarray
    compute_contact_force: Callable[[np.ndarray], np.ndarray]
    K_mat: sp.spmatrix
    K_geo: sp.spmatrix
    K_st: sp.spmatrix
    eps: float = 1e-7
    label: str = ""


@dataclass(frozen=True)
class ContactKcComponentFDDiagnosticOutput:
    """K_c 成分分解 FD 診断の出力.

    各 rel_err は ||dfc_FD - K_combo @ du|| / max(||dfc_FD||, ||K_combo @ du||).
    -1.0 は未計算（入力不足等）を示す。

    share_{mat,geo,st} は ||K_i @ du|| / ||K_c @ du|| で定義される寄与率。
    comp_share_* は成分（x/y/z/θx/θy/θz）別の不整合 L2 ノルムシェア（%単位、
    各成分 0〜100%）。既存 TangentFDDiagnosticProcess の表記と整合。
    """

    full_rel_err: float = -1.0  # K_c = K_mat - K_geo + K_st
    mat_only_rel_err: float = -1.0
    mat_geo_rel_err: float = -1.0  # K_mat - K_geo （K_st 除外）
    mat_st_rel_err: float = -1.0  # K_mat + K_st （K_geo 除外）
    share_mat: float = 0.0
    share_geo: float = 0.0
    share_st: float = 0.0
    # status-345: 部分行列作用ベクトルの絶対 L2 ノルム（share 分母ドリフト排除）
    mat_du_norm: float = 0.0
    geo_du_norm: float = 0.0
    st_du_norm: float = 0.0
    full_du_norm: float = 0.0
    dfc_fd_norm: float = 0.0
    comp_share_full: dict[str, float] = field(default_factory=dict)
    comp_share_mat_only: dict[str, float] = field(default_factory=dict)
    comp_share_mat_geo: dict[str, float] = field(default_factory=dict)
    comp_share_mat_st: dict[str, float] = field(default_factory=dict)
    report: str = ""


class ContactKcComponentFDDiagnosticProcess(
    SolverProcess[ContactKcComponentFDDiagnosticInput, ContactKcComponentFDDiagnosticOutput],
):
    """K_c を K_mat / K_geo / K_st に分解した FD 診断 Process.

    status-342 で特定された K_c x 成分 68% 不整合の由来を、
    部分行列 (mat / geo / st) レベルで切り分ける。

    検証手順:
    1. ``dfc_fd = (compute_contact_force(u+eps*du) - compute_contact_force(u)) / eps``
    2. 4 組み合わせ {mat-only, mat-geo, mat-st, full} で ``K_combo @ du`` を計算
    3. 各組み合わせの FD 相対誤差と comp 別（x/y/z/θ）不整合シェアを返す
    4. ``||K_i @ du||`` の寄与率（share_mat/geo/st）も併記

    frozen=True の Input/Output により不変性を担保し、診断ログの再現性を保証する。
    """

    meta = ProcessMeta(
        name="ContactKcComponentFDDiagnostic",
        module="verify",
        version="1.0.0",
        document_path="docs/verify.md",
    )

    def process(
        self,
        input_data: ContactKcComponentFDDiagnosticInput,
    ) -> ContactKcComponentFDDiagnosticOutput:
        inp = input_data
        lines: list[str] = []
        lines.append("=" * 60)
        hdr = "K_c 成分分解 FD 診断（status-342）"
        if inp.label:
            hdr += f"  [{inp.label}]"
        lines.append(hdr)
        lines.append("=" * 60)

        du_norm = float(np.linalg.norm(inp.du))
        lines.append(f"  ||du|| = {du_norm:.4e}, eps = {inp.eps:.2e}")

        if du_norm < 1e-30:
            lines.append("  du がゼロ — 診断スキップ")
            lines.append("=" * 60)
            return ContactKcComponentFDDiagnosticOutput(report="\n".join(lines))

        # ── FD: ∂f_c/∂u · du ──
        fc_base = np.asarray(inp.compute_contact_force(inp.u)).ravel()
        fc_pert = np.asarray(inp.compute_contact_force(inp.u + inp.eps * inp.du)).ravel()
        dfc_fd = (fc_pert - fc_base) / inp.eps
        dfc_norm = float(np.linalg.norm(dfc_fd))
        lines.append(f"  ||dfc (FD)|| = {dfc_norm:.4e}")

        # ── 解析: 各部分行列 @ du ──
        mat_du = np.asarray(inp.K_mat @ inp.du).ravel()
        geo_du = np.asarray(inp.K_geo @ inp.du).ravel()
        st_du = np.asarray(inp.K_st @ inp.du).ravel()

        # K_c = K_mat - K_geo + K_st（strategy.tangent_components の規約）
        full_du = mat_du - geo_du + st_du
        mat_geo_du = mat_du - geo_du
        mat_st_du = mat_du + st_du

        mat_du_norm = float(np.linalg.norm(mat_du))
        geo_du_norm = float(np.linalg.norm(geo_du))
        st_du_norm = float(np.linalg.norm(st_du))
        full_du_norm = float(np.linalg.norm(full_du))
        lines.append(
            f"  ||K_mat@du|| = {mat_du_norm:.4e}, "
            f"||K_geo@du|| = {geo_du_norm:.4e}, "
            f"||K_st@du||  = {st_du_norm:.4e}"
        )
        lines.append(f"  ||K_c@du||   = {full_du_norm:.4e}")

        # ── 寄与率（Kc に対するシェア） ──
        # status-345: 19本撚線で K_geo share が 1e-3 スケールの微小値を
        # 取りうるため、`{:5.2f}` では 0.00 に丸められて捕捉できない。
        # 科学表記（`{:.3e}`）で精度を保全し、下流 CSV パースでも復元可能に。
        ref_share = max(full_du_norm, 1e-30)
        share_mat = mat_du_norm / ref_share
        share_geo = geo_du_norm / ref_share
        share_st = st_du_norm / ref_share
        lines.append(f"  寄与率: mat={share_mat:.3e}, geo={share_geo:.3e}, st={share_st:.3e}")

        # ── 4 組み合わせの rel_err + comp 分解 ──
        combos = {
            "full": full_du,
            "mat_only": mat_du,
            "mat_geo": mat_geo_du,
            "mat_st": mat_st_du,
        }
        rel_errs: dict[str, float] = {}
        comp_shares: dict[str, dict[str, float]] = {}
        for name, K_du in combos.items():
            diff = dfc_fd - K_du
            diff_norm = float(np.linalg.norm(diff))
            ref = max(dfc_norm, float(np.linalg.norm(K_du)), 1e-30)
            rel_errs[name] = diff_norm / ref
            comp_shares[name] = _compute_comp_shares(diff)

        lines.append("-" * 60)
        lines.append("  組み合わせ別 FD 相対誤差（K_c = K_mat - K_geo + K_st 合意）:")
        lines.append(f"    full     = {rel_errs['full']:.4e}")
        lines.append(f"    mat_only = {rel_errs['mat_only']:.4e}  (geo/st 除外)")
        lines.append(f"    mat_geo  = {rel_errs['mat_geo']:.4e}  (st 除外)")
        lines.append(f"    mat_st   = {rel_errs['mat_st']:.4e}   (geo 除外)")

        # 最良組み合わせを報告
        best_name = min(rel_errs, key=lambda k: rel_errs[k])
        lines.append(f"  最良組み合わせ: {best_name} (rel_err={rel_errs[best_name]:.4e})")

        lines.append("-" * 60)
        for name in ("full", "mat_only", "mat_geo", "mat_st"):
            parts = ", ".join(f"{c}={comp_shares[name].get(c, 0.0):5.1f}%" for c in _COMP_NAMES)
            lines.append(f"  [{name:>8s}] comp 別不整合: [{parts}]")

        # ── warn if K_st is driving the mismatch ──
        if rel_errs["full"] > 0.1:
            delta_add_st = rel_errs["full"] - rel_errs["mat_geo"]
            delta_add_geo = rel_errs["mat_st"] - rel_errs["mat_only"]
            if delta_add_st > 0.05:
                lines.append(
                    f"  ⚠ K_st 足し込みで rel_err が {delta_add_st:+.4e} 悪化"
                    f" — K_st が primary driver の可能性"
                )
            if delta_add_geo > 0.05:
                lines.append(
                    f"  ⚠ K_geo 足し込みで rel_err が {delta_add_geo:+.4e} 悪化"
                    f" — K_geo が primary driver の可能性"
                )

        lines.append("=" * 60)

        return ContactKcComponentFDDiagnosticOutput(
            full_rel_err=rel_errs["full"],
            mat_only_rel_err=rel_errs["mat_only"],
            mat_geo_rel_err=rel_errs["mat_geo"],
            mat_st_rel_err=rel_errs["mat_st"],
            share_mat=share_mat,
            share_geo=share_geo,
            share_st=share_st,
            mat_du_norm=mat_du_norm,
            geo_du_norm=geo_du_norm,
            st_du_norm=st_du_norm,
            full_du_norm=full_du_norm,
            dfc_fd_norm=dfc_norm,
            comp_share_full=comp_shares["full"],
            comp_share_mat_only=comp_shares["mat_only"],
            comp_share_mat_geo=comp_shares["mat_geo"],
            comp_share_mat_st=comp_shares["mat_st"],
            report="\n".join(lines),
        )


def _compute_comp_shares(diff: np.ndarray) -> dict[str, float]:
    """DOF 差分ベクトルを成分別（x/y/z/θx/θy/θz）不整合シェア（%）に分解.

    各成分の L2 ノルムを全体 L2 ノルムで割った百分率（各成分 0〜100%）。
    Parseval より各成分ノルム二乗の和は全体ノルム二乗に等しいが、
    L2 ノルム自体の線形和は sqrt(n_comp) 倍まで膨らみうるため、
    本ヘルパーの返り値の総和は必ずしも 100% にはならない。
    既存 `TangentFDDiagnosticProcess` の comp 別シェア表記と整合。
    """
    arr = np.asarray(diff).ravel()
    if arr.size < _NDPN:
        return {}
    total = float(np.linalg.norm(arr))
    if total < 1e-30:
        return {c: 0.0 for c in _COMP_NAMES}
    shares: dict[str, float] = {}
    for c_idx, c_name in enumerate(_COMP_NAMES):
        idxs = np.arange(c_idx, arr.size, _NDPN)
        comp_norm = float(np.linalg.norm(arr[idxs]))
        shares[c_name] = 100.0 * comp_norm / total
    return shares
