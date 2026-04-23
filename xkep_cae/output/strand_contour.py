"""Strand3DContourProcess — 撚線 3D パイプコンターレンダリング.

設計仕様: docs/strand_contour.md

19本撚線等の撚線モデルで「接触状態 / 接触力 / 軸応力 / 曲率 / チャタリング」
を 3D パイプとして可視化する PostProcess。

`SolverResultData` + `MeshData` を受け取り、各フィールドごとに PNG を生成。
`contact_pair_history` が存在する場合はチャタリング解析も追加。

出力ビュー（各 PNG で 2 サブプロット）:
  - side view (XZ 平面投影)
  - 3D oblique (3次元視点)

対象フィールド（`requested_fields` で選択、デフォルト全て）:
  - contact: 接触/非接触の binary（赤/青）
  - contact_force: 接触力ノルム p_n（hot colormap）
  - stress: 要素軸応力 σ=Eε（coolwarm colormap）
  - curvature: 要素曲率 κ（viridis colormap）
  - chatter_binary: チャタリング検出 binary（pair_history 必須）
  - chatter_score: チャタリング強度 score（pair_history 必須、magma colormap）

status-362 の仮説 C 候補 (c) 実機検証の視覚化インフラとして新設。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta, SolverResultData

# 対応フィールド一覧（`requested_fields` の値として使用）
_ALL_FIELDS: tuple[str, ...] = (
    "contact",
    "contact_force",
    "stress",
    "curvature",
    "chatter_binary",
    "chatter_score",
)


@dataclass(frozen=True)
class Strand3DContourConfig:
    """撚線 3D コンターレンダリングの設定."""

    solver_result: SolverResultData
    mesh: MeshData
    output_dir: str = "docs/verification/strand_3d"
    prefix: str = "strand"
    ndof_per_node: int = 6
    young: float = 130.0e3  # MPa（軸応力計算用）
    elev: float = 15.0  # 3D 視点の仰角
    azim: float = -60.0  # 3D 視点の方位角
    requested_fields: tuple[str, ...] = _ALL_FIELDS
    suffix: str = ""  # 空なら frac 値から自動生成


@dataclass(frozen=True)
class Strand3DContourResult:
    """撚線 3D コンターレンダリングの結果."""

    image_paths: tuple[str, ...] = field(default_factory=tuple)
    field_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    n_chattering_elements: int = 0
    n_contact_elements: int = 0


# ====================================================================
# 純粋ヘルパー関数
# ====================================================================


def _compute_deformed(
    node_coords: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int,
) -> np.ndarray:
    """参照座標 + 並進変位から変形座標を計算."""
    n_nodes = node_coords.shape[0]
    deformed = node_coords.copy()
    for i in range(n_nodes):
        for d in range(3):
            idx = i * ndof_per_node + d
            if idx < len(u):
                deformed[i, d] += u[idx]
    return deformed


def _element_contact_mask(
    n_elems: int,
    manager: object,
) -> tuple[np.ndarray, np.ndarray]:
    """manager.pairs から要素ごとの接触マスク + 接触力マップを算出."""
    is_contact = np.zeros(n_elems, dtype=bool)
    force_per_elem = np.zeros(n_elems)
    if manager is None or not hasattr(manager, "pairs"):
        return is_contact, force_per_elem
    for p in manager.pairs:
        if not hasattr(p, "state"):
            continue
        if p.state.p_n <= 0.0:
            continue
        elem_a = int(p.elem_a)
        elem_b = int(p.elem_b)
        if 0 <= elem_a < n_elems:
            is_contact[elem_a] = True
            force_per_elem[elem_a] += float(p.state.p_n)
        if 0 <= elem_b < n_elems:
            is_contact[elem_b] = True
            force_per_elem[elem_b] += float(p.state.p_n)
    return is_contact, force_per_elem


def _element_curvature(
    deformed: np.ndarray,
    connectivity: np.ndarray,
    strand_ids: np.ndarray,
) -> np.ndarray:
    """同素線隣接要素の接線ベクトル差から要素曲率を近似計算."""
    n_elems = connectivity.shape[0]
    tangents = np.zeros((n_elems, 3))
    lengths = np.zeros(n_elems)
    for e in range(n_elems):
        n0, n1 = connectivity[e]
        v = deformed[n1] - deformed[n0]
        L = float(np.linalg.norm(v))
        lengths[e] = L
        tangents[e] = v / max(L, 1e-30)

    # 要素素線 ID 判定
    if strand_ids.shape[0] == n_elems:
        elem_strand = strand_ids.astype(int)
    else:
        elem_strand = np.zeros(n_elems, dtype=int)
        for e in range(n_elems):
            n0 = int(connectivity[e, 0])
            if n0 < strand_ids.shape[0]:
                elem_strand[e] = int(strand_ids[n0])

    curvature = np.zeros(n_elems)
    for e in range(n_elems):
        strand = elem_strand[e]
        max_angle_per_L = 0.0
        for ne in range(n_elems):
            if ne == e or elem_strand[ne] != strand:
                continue
            if connectivity[e, 0] in connectivity[ne] or connectivity[e, 1] in connectivity[ne]:
                dot = float(np.clip(tangents[e] @ tangents[ne], -1.0, 1.0))
                angle = float(np.arccos(dot))
                L_avg = 0.5 * (lengths[e] + lengths[ne])
                if L_avg > 1e-30:
                    max_angle_per_L = max(max_angle_per_L, angle / L_avg)
        curvature[e] = max_angle_per_L
    return curvature


def _element_axial_stress(
    deformed: np.ndarray,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    young: float,
) -> np.ndarray:
    """要素軸ひずみ × Young 率から軸応力."""
    n_elems = connectivity.shape[0]
    stress = np.zeros(n_elems)
    for e in range(n_elems):
        n0, n1 = connectivity[e]
        L0 = float(np.linalg.norm(node_coords_ref[n1] - node_coords_ref[n0]))
        L1 = float(np.linalg.norm(deformed[n1] - deformed[n0]))
        if L0 > 1e-30:
            eps = (L1 - L0) / L0
            stress[e] = young * eps
    return stress


def _element_chattering_score(
    n_elems: int,
    contact_pair_history: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """pair_history から要素チャタリングスコア.

    各ペア (elem_a, elem_b) で
      - activation_flip: active(p_n>0) ↔ inactive 状態遷移回数
      - stick_slide_flip: 両 increment active 時の stick↔slide 遷移回数
    の合計を increments で正規化した score を、関連要素に加算する。
    """
    n_increments = len(contact_pair_history)
    if n_increments < 2:
        return np.zeros(n_elems), np.zeros(n_elems, dtype=bool)

    # (elem_a, elem_b) → per-increment (is_active, is_stick) 履歴構築
    per_incr_states: list[dict[tuple[int, int], tuple[bool, bool]]] = []
    all_keys: set[tuple[int, int]] = set()
    for _frac, entries in contact_pair_history:
        state_map: dict[tuple[int, int], tuple[bool, bool]] = {}
        for e in entries:
            key = (int(e.elem_a), int(e.elem_b))
            state_map[key] = (float(e.p_n) > 0.0, bool(e.stick))
            all_keys.add(key)
        per_incr_states.append(state_map)

    pair_history_by_id: dict[tuple[int, int], list[tuple[bool, bool]]] = defaultdict(list)
    for key in all_keys:
        hist = []
        for state_map in per_incr_states:
            if key in state_map:
                hist.append(state_map[key])
            else:
                hist.append((False, False))
        pair_history_by_id[key] = hist

    chatter_score = np.zeros(n_elems)
    for key, hist in pair_history_by_id.items():
        elem_a, elem_b = key
        activation_flips = 0
        stick_slide_flips = 0
        for i in range(1, len(hist)):
            prev_active, prev_stick = hist[i - 1]
            curr_active, curr_stick = hist[i]
            if prev_active != curr_active:
                activation_flips += 1
            if prev_active and curr_active and prev_stick != curr_stick:
                stick_slide_flips += 1
        pair_score = (activation_flips + stick_slide_flips) / max(n_increments, 1)
        if 0 <= elem_a < n_elems:
            chatter_score[elem_a] += pair_score
        if 0 <= elem_b < n_elems:
            chatter_score[elem_b] += pair_score
    return chatter_score, chatter_score > 0.0


def _resolve_element_radii(radii: Any, connectivity: np.ndarray, n_elems: int) -> np.ndarray:
    """要素半径配列を解決（スカラー / 要素長 / ノード長いずれも対応）."""
    if np.isscalar(radii):
        return np.full(n_elems, float(radii))
    arr = np.asarray(radii, dtype=float)
    if arr.shape[0] == n_elems:
        return arr
    # ノード配列と仮定して連結性経由（範囲外は平均で埋める）
    r_elem = np.full(n_elems, float(arr.mean()))
    for e in range(n_elems):
        n0 = int(connectivity[e, 0])
        n1 = int(connectivity[e, 1])
        if n0 < arr.shape[0] and n1 < arr.shape[0]:
            r_elem[e] = 0.5 * (arr[n0] + arr[n1])
    return r_elem


def _render_single(  # noqa: PLR0913
    deformed: np.ndarray,
    connectivity: np.ndarray,
    r_arr: np.ndarray,
    field_values: np.ndarray,
    title: str,
    output_path: Path,
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    binary: bool = False,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    """1 フィールドを 2 サブプロット（XZ / 3D oblique）で PNG 保存."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(14, 6))
    n_elems = connectivity.shape[0]
    r_max = max(float(r_arr.max()), 1e-30)

    if binary:
        colors = np.where(field_values.astype(bool), "red", "blue")
        norm = None
        cmap_obj = None
    else:
        if vmin is None:
            vmin = float(np.nanmin(field_values))
        if vmax is None:
            vmax = float(np.nanmax(field_values))
        if vmax - vmin < 1e-30:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

    # Subplot 1: XZ projection
    ax_xz = fig.add_subplot(1, 2, 1)
    for e in range(n_elems):
        n0, n1 = connectivity[e]
        if binary:
            color = colors[e]
        else:
            color = cmap_obj(norm(float(field_values[e])))
        ax_xz.plot(
            [deformed[n0, 0], deformed[n1, 0]],
            [deformed[n0, 2], deformed[n1, 2]],
            "-",
            color=color,
            linewidth=2.0 * r_arr[e] / r_max + 0.5,
        )
    ax_xz.set_xlabel("X [mm]")
    ax_xz.set_ylabel("Z [mm]")
    ax_xz.set_title(f"{title} — side view (XZ)")
    ax_xz.set_aspect("equal")
    ax_xz.grid(True, alpha=0.3)

    # Subplot 2: 3D oblique
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    for e in range(n_elems):
        n0, n1 = connectivity[e]
        if binary:
            color = colors[e]
        else:
            color = cmap_obj(norm(float(field_values[e])))
        ax_3d.plot(
            [deformed[n0, 0], deformed[n1, 0]],
            [deformed[n0, 1], deformed[n1, 1]],
            [deformed[n0, 2], deformed[n1, 2]],
            "-",
            color=color,
            linewidth=2.0 * r_arr[e] / r_max + 0.5,
        )
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title(f"{title} — 3D oblique")
    ax_3d.view_init(elev=elev, azim=azim)
    try:
        ax_3d.set_box_aspect(
            [
                float(deformed[:, 0].ptp()),
                max(float(deformed[:, 1].ptp()), 1e-6),
                float(deformed[:, 2].ptp()),
            ]
        )
    except Exception:
        pass

    # Legend / colorbar
    if binary:
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color="red", linewidth=3, label="true"),
                plt.Line2D([0], [0], color="blue", linewidth=3, label="false"),
            ],
            loc="lower center",
            ncol=2,
        )
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=fig.axes, shrink=0.8, pad=0.02)

    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# PostProcess
# ====================================================================


class Strand3DContourProcess(PostProcess[Strand3DContourConfig, Strand3DContourResult]):
    """撚線 3D パイプコンターレンダリング Process.

    `SolverResultData` + `MeshData` から接触 / 力 / 応力 / 曲率 /
    チャタリング等のフィールドを 3D パイプで可視化し PNG 出力する。

    チャタリング解析は `solver_result.contact_pair_history` が非空の場合に
    有効（`StrandBendingOscillationConfig.track_contact_pairs=True` 等で記録）。
    """

    meta = ProcessMeta(
        name="Strand3DContour",
        module="post",
        version="1.0.0",
        document_path="docs/strand_contour.md",
    )
    uses = ()

    def process(self, input_data: Strand3DContourConfig) -> Strand3DContourResult:
        cfg = input_data
        result = cfg.solver_result
        mesh = cfg.mesh

        node_coords = np.asarray(mesh.node_coords, dtype=float)
        connectivity = np.asarray(mesh.connectivity, dtype=int)
        strand_ids = np.asarray(mesh.strand_ids, dtype=int)
        n_elems = connectivity.shape[0]
        deformed = _compute_deformed(node_coords, result.u, cfg.ndof_per_node)
        r_arr = _resolve_element_radii(mesh.radii, connectivity, n_elems)

        # 共通フィールド計算
        is_contact, force_per_elem = _element_contact_mask(n_elems, result.final_contact_manager)
        curvature = _element_curvature(deformed, connectivity, strand_ids)
        stress = _element_axial_stress(deformed, node_coords, connectivity, cfg.young)

        # チャタリング（pair_history 依存）
        pair_history = result.contact_pair_history
        chatter_score, has_chatter = _element_chattering_score(n_elems, pair_history)

        # 最新 frac 取得
        frac = result.load_history[-1] if result.load_history else 0.0
        suffix = cfg.suffix if cfg.suffix else f"frac{frac:.3f}"

        out_dir = Path(cfg.output_dir)
        image_paths: list[str] = []
        field_stats: dict[str, dict[str, float]] = {}

        # フィールド定義: (name, values, kwargs)
        field_specs = {
            "contact": (
                is_contact,
                f"{cfg.prefix} — contact status (frac={frac:.4f})",
                {"binary": True},
            ),
            "contact_force": (
                force_per_elem,
                f"{cfg.prefix} — contact force norm p_n (frac={frac:.4f})",
                {"cmap": "hot", "vmin": 0.0},
            ),
            "stress": (
                stress,
                f"{cfg.prefix} — axial stress σ=Eε [MPa] (frac={frac:.4f})",
                {"cmap": "coolwarm"},
            ),
            "curvature": (
                curvature,
                f"{cfg.prefix} — curvature κ [1/mm] (frac={frac:.4f})",
                {"cmap": "viridis", "vmin": 0.0},
            ),
            "chatter_binary": (
                has_chatter,
                (
                    f"{cfg.prefix} — chattering detected (frac={frac:.4f}, "
                    f"{int(has_chatter.sum())}/{n_elems})"
                ),
                {"binary": True},
            ),
            "chatter_score": (
                chatter_score,
                (
                    f"{cfg.prefix} — chattering intensity score "
                    f"(frac={frac:.4f}, max={float(chatter_score.max()):.3f})"
                ),
                {"cmap": "magma", "vmin": 0.0},
            ),
        }

        for name in cfg.requested_fields:
            if name not in field_specs:
                continue
            # pair_history 必須フィールドの skip
            if name in ("chatter_binary", "chatter_score") and len(pair_history) < 2:
                continue
            values, title, kwargs = field_specs[name]
            path = out_dir / f"{cfg.prefix}_{name}_{suffix}.png"
            _render_single(
                deformed,
                connectivity,
                r_arr,
                np.asarray(values),
                title=title,
                output_path=path,
                elev=cfg.elev,
                azim=cfg.azim,
                **kwargs,
            )
            image_paths.append(str(path))
            # 統計
            if values.dtype == bool:
                field_stats[name] = {"n_true": float(int(values.sum()))}
            else:
                field_stats[name] = {
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "mean": float(np.nanmean(values)),
                }

        return Strand3DContourResult(
            image_paths=tuple(image_paths),
            field_stats=field_stats,
            n_chattering_elements=int(has_chatter.sum()),
            n_contact_elements=int(is_contact.sum()),
        )
