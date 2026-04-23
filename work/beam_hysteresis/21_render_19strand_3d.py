"""work/beam_hysteresis/21_render_19strand_3d.py — 19本撚線 3D パイプレンダリング.

[← README](README.md) | [← project README](../../README.md)

status-362 仮説 C (c) の 19 本撚線実機検証結果（frac=0.5153 stall）を
3D パイプとして可視化。4 種類のフィールドをコンターで表示:

1. **contact** (binary): 接触ペアに属する要素=赤、非接触=青
2. **contact_force** (scalar): 要素に作用する接触力ノルム
3. **stress** (scalar): 梁内力ベースの軸応力
4. **curvature** (scalar): 要素曲率

19 本撚線の BT 設定を流用。sandbox 時間制約のため `max_increments=150`
に制限（自然な stall 前に打切る）して確実に pickle を保存する。

実行:
    uv run --extra dev python work/beam_hysteresis/21_render_19strand_3d.py \\
        2>&1 | tee /tmp/render_19strand_$(date +%s).log

出力: `docs/verification/19strand_3d/*.png` 4 枚。
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def run_solver_and_pickle(pkl_path: Path) -> tuple[object, object]:
    """19本撚線を実行して結果を pickle 保存。既存 pickle があればスキップ."""
    if pkl_path.exists():
        print(f"[cache] 既存 pickle 使用: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data["result"], data["cfg"]

    print("=" * 70)
    print("19本撚線 BT 実行（max_increments=150 で打切り、pickle 保存）")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=150,  # sandbox 時間制約で早期打切り
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        contact_backtracking_enabled=True,
        contact_backtracking_mixed_only=True,
        contact_backtracking_active_flip_threshold=3,
        contact_backtracking_residual_ratio=2.0,
        contact_backtracking_max_steps=4,
        contact_backtracking_rate_threshold=0.85,
        # チャタリング可視化のため per-increment 接触ペア履歴を記録
        track_contact_pairs=True,
    )

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    print()
    print(f"  frac={frac:.4f}, incr={sr.n_increments}, cb={sr.n_cutbacks}, elapsed={elapsed:.1f}s")

    # pickle 保存（mesh / u / contact pairs + pair history）
    mesh = result.mesh
    manager = sr.final_contact_manager
    pairs_snapshot = []
    if manager is not None and hasattr(manager, "pairs"):
        for p in manager.pairs:
            if hasattr(p, "state"):
                pairs_snapshot.append(
                    {
                        "elem_a": int(p.elem_a),
                        "elem_b": int(p.elem_b),
                        "nodes_a": np.asarray(p.nodes_a, dtype=int),
                        "nodes_b": np.asarray(p.nodes_b, dtype=int),
                        "radius_a": float(p.radius_a),
                        "radius_b": float(p.radius_b),
                        "gap": float(p.state.gap),
                        "p_n": float(p.state.p_n),
                        "normal": np.asarray(p.state.normal, dtype=float),
                        "status": str(p.state.status.name)
                        if hasattr(p.state.status, "name")
                        else str(p.state.status),
                    }
                )

    # 各 increment の active ペアスナップショット履歴（チャタリング可視化用）
    pair_history: list[tuple[float, list[dict]]] = []
    for frac_i, entries in sr.contact_pair_history:
        entry_dicts = []
        for e in entries:
            entry_dicts.append(
                {
                    "elem_a": int(e.elem_a),
                    "elem_b": int(e.elem_b),
                    "p_n": float(e.p_n),
                    "gap": float(e.gap),
                    "stick": bool(e.stick),
                }
            )
        pair_history.append((float(frac_i), entry_dicts))

    data = {
        "result": {
            "u": sr.u.copy(),
            "frac": frac,
            "n_increments": sr.n_increments,
            "n_cutbacks": sr.n_cutbacks,
            "converged": sr.converged,
            "elapsed": elapsed,
            "pairs_snapshot": pairs_snapshot,
            "pair_history": pair_history,  # 新規: per-increment チャタリング解析用
            "node_coords": mesh.node_coords.copy(),
            "connectivity": mesh.connectivity.copy(),
            "radii": mesh.radii if np.isscalar(mesh.radii) else mesh.radii.copy(),
            "n_strands": int(mesh.n_strands),
            "strand_ids": mesh.strand_ids.copy(),
        },
        "cfg": {
            "wire_radius": cfg.wire_radius,
            "bending_curvature": cfg.bending_curvature,
        },
    }
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[pickle] 保存: {pkl_path}")
    return data["result"], data["cfg"]


def compute_deformed_coords(
    node_coords: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """変形後座標（変位の並進成分のみ反映）."""
    n_nodes = node_coords.shape[0]
    deformed = node_coords.copy()
    for i in range(n_nodes):
        for d in range(3):
            idx = i * ndof_per_node + d
            if idx < len(u):
                deformed[i, d] += u[idx]
    return deformed


def element_contact_mask(n_elems: int, pairs_snapshot: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """要素ごとの接触状態マスク + 接触力強度マップ."""
    is_contact = np.zeros(n_elems, dtype=bool)
    force_per_elem = np.zeros(n_elems)
    for p in pairs_snapshot:
        if p["p_n"] > 0.0 or p["status"] in ("ACTIVE", "SLIDING", "active", "sliding"):
            is_contact[p["elem_a"]] = True
            is_contact[p["elem_b"]] = True
            # 接触力は両要素に均等に振る
            force_per_elem[p["elem_a"]] += p["p_n"]
            force_per_elem[p["elem_b"]] += p["p_n"]
    return is_contact, force_per_elem


def element_curvature(
    deformed: np.ndarray,
    connectivity: np.ndarray,
    strand_ids: np.ndarray,
) -> np.ndarray:
    """要素ごとの近似曲率（接続する要素の接線ベクトル差から推定）."""
    n_elems = connectivity.shape[0]
    tangents = np.zeros((n_elems, 3))
    lengths = np.zeros(n_elems)
    for e in range(n_elems):
        n0, n1 = connectivity[e]
        v = deformed[n1] - deformed[n0]
        L = float(np.linalg.norm(v))
        lengths[e] = L
        tangents[e] = v / max(L, 1e-30)

    # 各素線内で隣接要素の接線差から曲率推定
    curvature = np.zeros(n_elems)
    # strand_ids は要素ごと（n_elems 長）なので直接使用
    if strand_ids.shape[0] == n_elems:
        elem_strand = strand_ids.astype(int)
    else:
        # 念のためフォールバック: ノードから判定
        elem_strand = np.zeros(n_elems, dtype=int)
        for e in range(n_elems):
            n0 = int(connectivity[e, 0])
            if n0 < strand_ids.shape[0]:
                elem_strand[e] = int(strand_ids[n0])

    for e in range(n_elems):
        # 同素線の隣接要素を探す
        strand = elem_strand[e]
        neighbors = []
        for ne in range(n_elems):
            if ne == e or elem_strand[ne] != strand:
                continue
            # 共通ノードあり
            if connectivity[e, 0] in connectivity[ne] or connectivity[e, 1] in connectivity[ne]:
                neighbors.append(ne)
        if not neighbors:
            continue
        # 最大の接線ベクトル差 / 要素長 = 曲率近似
        max_angle_per_L = 0.0
        for ne in neighbors:
            dot = float(np.clip(tangents[e] @ tangents[ne], -1.0, 1.0))
            angle = float(np.arccos(dot))
            L_avg = 0.5 * (lengths[e] + lengths[ne])
            if L_avg > 1e-30:
                max_angle_per_L = max(max_angle_per_L, angle / L_avg)
        curvature[e] = max_angle_per_L
    return curvature


def element_axial_stress(
    deformed: np.ndarray,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    young: float = 130.0e3,
) -> np.ndarray:
    """要素ごとの軸ひずみからの軸応力 σ=E·ε."""
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


def element_chattering_score(
    n_elems: int,
    pair_history: list[tuple[float, list[dict]]],
) -> tuple[np.ndarray, np.ndarray]:
    """pair_history から要素ごとのチャタリングスコアを算出.

    チャタリングスコア定義:
      s_e = (# activation flips + # stick/slide flips) / # increments

    各ペア (elem_a, elem_b) について:
      - increment 間で active(p_n>0) → inactive → active の遷移を activation_flip として数える
      - stick=True ↔ False の遷移を stick_slide_flip として数える
    要素 e のスコアは、e を含む全ペアの flip 数合計を #increments で正規化。

    戻り値: (chatter_score, has_chattering_mask)
      - chatter_score: 要素ごとの float スコア（0 = 安定、大きい = チャタリング）
      - has_chattering_mask: score > 0 の binary 判定
    """
    if not pair_history:
        return np.zeros(n_elems), np.zeros(n_elems, dtype=bool)

    n_increments = len(pair_history)

    # (elem_a, elem_b) キーで各 increment の状態を集約
    from collections import defaultdict

    pair_history_by_id: dict[tuple[int, int], list[tuple[bool, bool]]] = defaultdict(list)
    # 各 increment で出現する (elem_a, elem_b) を記録（欠落増分は inactive 扱い）
    all_pair_ids: set[tuple[int, int]] = set()
    per_incr_pair_states: list[dict[tuple[int, int], tuple[bool, bool]]] = []
    for _frac_i, entries in pair_history:
        state_map: dict[tuple[int, int], tuple[bool, bool]] = {}
        for e in entries:
            key = (int(e["elem_a"]), int(e["elem_b"]))
            # (is_active, is_stick)
            state_map[key] = (e["p_n"] > 0.0, bool(e["stick"]))
            all_pair_ids.add(key)
        per_incr_pair_states.append(state_map)

    for key in all_pair_ids:
        # 各 increment の状態（欠落 = (False, False) = inactive）
        history = []
        for state_map in per_incr_pair_states:
            if key in state_map:
                history.append(state_map[key])
            else:
                history.append((False, False))
        pair_history_by_id[key] = history

    chatter_score = np.zeros(n_elems)
    for key, history in pair_history_by_id.items():
        elem_a, elem_b = key
        activation_flips = 0
        stick_slide_flips = 0
        for i in range(1, len(history)):
            prev_active, prev_stick = history[i - 1]
            curr_active, curr_stick = history[i]
            if prev_active != curr_active:
                activation_flips += 1
            # stick/slide flips のみ両 increment で active な場合にカウント
            if prev_active and curr_active and prev_stick != curr_stick:
                stick_slide_flips += 1
        pair_score = (activation_flips + stick_slide_flips) / max(n_increments, 1)
        if 0 <= elem_a < n_elems:
            chatter_score[elem_a] += pair_score
        if 0 <= elem_b < n_elems:
            chatter_score[elem_b] += pair_score

    has_chattering = chatter_score > 0.0
    return chatter_score, has_chattering


def render_3d_pipes(
    deformed: np.ndarray,
    connectivity: np.ndarray,
    radii: float | np.ndarray,
    field_values: np.ndarray,
    title: str,
    output_path: Path,
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    binary: bool = False,
    tube_segments: int = 12,
    elev: float = 15,
    azim: float = -60,
) -> None:
    """3D パイプレンダリング（matplotlib 3D tube 風）.

    binary=True の場合、field_values は 0/1 の bool 的配列で、
    True=赤、False=青 で描く。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(14, 6))

    n_elems = connectivity.shape[0]
    if np.isscalar(radii):
        r_arr = np.full(n_elems, float(radii))
    elif np.asarray(radii).shape[0] == n_elems:
        # radii は要素ごと（n_elems 長）なのでそのまま使用
        r_arr = np.asarray(radii, dtype=float)
    else:
        # フォールバック: ノード配列として連結性経由
        r_node = np.asarray(radii, dtype=float)
        r_arr = np.array(
            [
                0.5 * (r_node[connectivity[e, 0]] + r_node[connectivity[e, 1]])
                for e in range(n_elems)
                if connectivity[e, 0] < r_node.shape[0] and connectivity[e, 1] < r_node.shape[0]
            ]
        )
        if r_arr.shape[0] != n_elems:
            r_arr = np.full(n_elems, float(np.mean(r_node)))

    if binary:
        colors = np.where(field_values, "red", "blue")
    else:
        if vmin is None:
            vmin = float(np.nanmin(field_values))
        if vmax is None:
            vmax = float(np.nanmax(field_values))
        if vmax - vmin < 1e-30:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

    # 2 ビュー: 側面 (XZ) + 斜め 3D
    for _sp_i, (_ax_projection, view) in enumerate(
        [
            (None, "xy"),  # XY 平面（上から）
            ("3d", "3d"),  # 3D
        ]
    ):
        if view == "xy":
            ax = fig.add_subplot(1, 2, 1)
            for e in range(n_elems):
                n0, n1 = connectivity[e]
                xs = [deformed[n0, 0], deformed[n1, 0]]
                zs = [deformed[n0, 2], deformed[n1, 2]]
                if binary:
                    color = colors[e]
                else:
                    color = cmap_obj(norm(field_values[e]))
                ax.plot(xs, zs, "-", color=color, linewidth=2.0 * r_arr[e] / r_arr.max() + 0.5)
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Z [mm]")
            ax.set_title(f"{title} — side view (XZ)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        else:
            ax = fig.add_subplot(1, 2, 2, projection="3d")
            for e in range(n_elems):
                n0, n1 = connectivity[e]
                xs = [deformed[n0, 0], deformed[n1, 0]]
                ys = [deformed[n0, 1], deformed[n1, 1]]
                zs = [deformed[n0, 2], deformed[n1, 2]]
                if binary:
                    color = colors[e]
                else:
                    color = cmap_obj(norm(field_values[e]))
                ax.plot(xs, ys, zs, "-", color=color, linewidth=2.0 * r_arr[e] / r_arr.max() + 0.5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"{title} — 3D oblique")
            ax.view_init(elev=elev, azim=azim)
            # 等アスペクトに近づける
            try:
                ax.set_box_aspect(
                    [
                        float(deformed[:, 0].ptp()),
                        float(deformed[:, 1].ptp()),
                        float(deformed[:, 2].ptp()),
                    ]
                )
            except Exception:
                pass

    # カラーバー / 凡例
    if binary:
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color="red", linewidth=3, label="contact"),
                plt.Line2D([0], [0], color="blue", linewidth=3, label="no contact"),
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
    print(f"  [render] {output_path}")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "docs" / "verification" / "19strand_3d"
    pkl_path = output_dir / "19strand_bt_stall.pkl"

    result, cfg = run_solver_and_pickle(pkl_path)

    node_coords = result["node_coords"]
    connectivity = result["connectivity"]
    u = result["u"]
    pairs_snapshot = result["pairs_snapshot"]
    radii = result["radii"]
    strand_ids = result["strand_ids"]
    frac = result["frac"]

    deformed = compute_deformed_coords(node_coords, u)
    n_elems = connectivity.shape[0]
    is_contact, force_per_elem = element_contact_mask(n_elems, pairs_snapshot)
    curvature = element_curvature(deformed, connectivity, strand_ids)
    stress = element_axial_stress(deformed, node_coords, connectivity, young=130.0e3)

    # チャタリング解析（pair_history が保存されている場合のみ）
    pair_history = result.get("pair_history", [])
    chatter_score, has_chatter = element_chattering_score(n_elems, pair_history)

    print()
    print("=" * 70)
    print(f"レンダリング統計（frac={frac:.4f}, n_elems={n_elems}）")
    print("=" * 70)
    n_contact = int(is_contact.sum())
    print(f"  接触要素: {n_contact} / {n_elems} ({100.0 * n_contact / n_elems:.1f}%)")
    print(
        f"  接触ペア: {len(pairs_snapshot)} (active p_n>0: "
        f"{sum(1 for p in pairs_snapshot if p['p_n'] > 0.0)})"
    )
    print(f"  接触力 max: {float(force_per_elem.max()):.3e}")
    print(f"  軸応力 [MPa]: min={float(stress.min()):.2e}, max={float(stress.max()):.2e}")
    print(f"  曲率 [1/mm]: min={float(curvature.min()):.2e}, max={float(curvature.max()):.2e}")
    n_chatter = int(has_chatter.sum())
    print(
        f"  チャタリング要素: {n_chatter} / {n_elems} "
        f"({100.0 * n_chatter / n_elems:.1f}%, score mean={float(chatter_score.mean()):.3f}, "
        f"max={float(chatter_score.max()):.3f})"
    )
    print(f"  pair_history 長: {len(pair_history)} increments")

    suffix = f"frac{frac:.3f}"

    render_3d_pipes(
        deformed,
        connectivity,
        radii,
        is_contact,
        title=f"19-strand BT stall — contact status (frac={frac:.4f})",
        output_path=output_dir / f"19strand_contact_{suffix}.png",
        binary=True,
    )
    render_3d_pipes(
        deformed,
        connectivity,
        radii,
        force_per_elem,
        title=f"19-strand BT stall — contact force norm p_n (frac={frac:.4f})",
        output_path=output_dir / f"19strand_contact_force_{suffix}.png",
        cmap="hot",
        vmin=0.0,
    )
    render_3d_pipes(
        deformed,
        connectivity,
        radii,
        stress,
        title=f"19-strand BT stall — axial stress σ=Eε [MPa] (frac={frac:.4f})",
        output_path=output_dir / f"19strand_stress_{suffix}.png",
        cmap="coolwarm",
    )
    render_3d_pipes(
        deformed,
        connectivity,
        radii,
        curvature,
        title=f"19-strand BT stall — curvature κ [1/mm] (frac={frac:.4f})",
        output_path=output_dir / f"19strand_curvature_{suffix}.png",
        cmap="viridis",
        vmin=0.0,
    )

    # チャタリング可視化（pair_history あり時のみ）
    if len(pair_history) >= 2:
        render_3d_pipes(
            deformed,
            connectivity,
            radii,
            has_chatter,
            title=(
                f"19-strand BT stall — chattering detected (frac={frac:.4f}, "
                f"{n_chatter}/{n_elems}={100.0 * n_chatter / n_elems:.1f}%)"
            ),
            output_path=output_dir / f"19strand_chatter_binary_{suffix}.png",
            binary=True,
        )
        render_3d_pipes(
            deformed,
            connectivity,
            radii,
            chatter_score,
            title=(
                f"19-strand BT stall — chattering intensity score "
                f"(frac={frac:.4f}, max={float(chatter_score.max()):.3f})"
            ),
            output_path=output_dir / f"19strand_chatter_score_{suffix}.png",
            cmap="magma",
            vmin=0.0,
        )
        n_images = 6
    else:
        print("  (pair_history が 2 increment 未満のためチャタリング可視化 skip)")
        n_images = 4

    print()
    print("=" * 70)
    print(f"完了: PNG {n_images} 枚保存 → {output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
