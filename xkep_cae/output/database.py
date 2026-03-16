"""出力データベース（OutputDatabase）.

過渡応答解析の全結果を Step/Increment/Frame 階層で管理し、
ソルバー結果からの構築機能と、エクスポート機能を提供する。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from xkep_cae.output.request import (
    ENERGY_VARIABLES,
    NODAL_VARIABLES,
)
from xkep_cae.output.step import Frame, IncrementResult, Step, StepResult


@dataclass
class OutputDatabase:
    """全ステップの出力データベース.

    Attributes:
        step_results: ステップ結果のリスト
        node_coords: (n_nodes, ndim) 節点座標
        connectivity: 要素接続情報 [(vtk_cell_type, node_indices_array), ...]
        ndof_per_node: 1節点あたりの自由度数
        node_sets: 名前付き節点集合 {名前: 節点インデックス配列}
        fixed_dofs: 拘束DOFインデックス
    """

    step_results: list[StepResult] = field(default_factory=list)
    node_coords: np.ndarray | None = None
    connectivity: list[tuple[int, np.ndarray]] | None = None
    ndof_per_node: int = 2
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    fixed_dofs: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        """ステップ数."""
        return len(self.step_results)

    @property
    def n_nodes(self) -> int:
        """節点数."""
        if self.node_coords is None:
            return 0
        return self.node_coords.shape[0]

    @property
    def ndim(self) -> int:
        """空間次元数."""
        if self.node_coords is None:
            return 0
        return self.node_coords.shape[1]

    def all_frames(self) -> list[tuple[int, Frame]]:
        """全ステップの全フレームを (step_index, Frame) のリストで返す."""
        result = []
        for sr in self.step_results:
            for frame in sr.frames:
                result.append((sr.step_index, frame))
        return result

    def total_time(self) -> float:
        """全ステップの合計時間."""
        return sum(sr.step.total_time for sr in self.step_results)


def build_output_database(
    steps: list[Step],
    solver_results: list[Any],
    *,
    node_coords: np.ndarray | None = None,
    connectivity: list[tuple[int, np.ndarray]] | None = None,
    ndof_per_node: int = 2,
    node_sets: dict[str, np.ndarray] | None = None,
    fixed_dofs: np.ndarray | None = None,
    M: np.ndarray | sp.spmatrix | None = None,
    K: np.ndarray | sp.spmatrix | None = None,
    f_ext_funcs: list[Callable[[float], np.ndarray]] | None = None,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray] | None = None,
    element_data_func: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
) -> OutputDatabase:
    """既存のソルバー結果から OutputDatabase を構築する.

    ソルバー結果（TransientResult / NonlinearTransientResult 等）を
    Step/Increment/Frame 階層に変換する。

    Args:
        steps: ステップ定義のリスト
        solver_results: 各ステップのソルバー結果
            TransientResult, NonlinearTransientResult, CentralDifferenceResult
        node_coords: (n_nodes, ndim) 節点座標
        connectivity: 要素接続 [(vtk_cell_type, node_indices), ...]
        ndof_per_node: 1節点あたりのDOF数
        node_sets: 名前付き節点集合
        fixed_dofs: 拘束DOF
        M: 質量行列（ALLKE 計算用）
        K: 剛性行列（ALLIE 計算用）
        f_ext_funcs: 各ステップの外力関数（CF 出力用）
        assemble_internal_force: 非線形内力コールバック u → f_int(u)。
            指定時は反力を RF = f_int(u) + M·a で計算する（線形の K·u + M·a より正確）。
        element_data_func: 変位ベクトルから要素データを計算するコールバック。
            u → {"stress_xx": ndarray, ...}。フレーム生成時に呼ばれる。

    Returns:
        OutputDatabase
    """
    if len(steps) != len(solver_results):
        raise ValueError(
            f"steps ({len(steps)}) と solver_results ({len(solver_results)}) の数が一致しない"
        )

    nsets = node_sets if node_sets is not None else {}
    db = OutputDatabase(
        node_coords=node_coords,
        connectivity=connectivity,
        ndof_per_node=ndof_per_node,
        node_sets={k: np.asarray(v, dtype=int) for k, v in nsets.items()},
        fixed_dofs=np.asarray(fixed_dofs, dtype=int) if fixed_dofs is not None else None,
    )

    cumulative_time = 0.0

    for step_idx, (step, result) in enumerate(zip(steps, solver_results, strict=True)):
        sr = _build_step_result(
            step=step,
            step_index=step_idx,
            start_time=cumulative_time,
            solver_result=result,
            ndof_per_node=ndof_per_node,
            node_sets=nsets,
            fixed_dofs=fixed_dofs,
            M=M,
            K=K,
            f_ext_func=f_ext_funcs[step_idx] if f_ext_funcs is not None else None,
            assemble_internal_force=assemble_internal_force,
            element_data_func=element_data_func,
        )
        db.step_results.append(sr)
        cumulative_time += step.total_time

    return db


def _build_step_result(
    step: Step,
    step_index: int,
    start_time: float,
    solver_result: Any,
    ndof_per_node: int,
    node_sets: dict[str, np.ndarray | list[int]],
    fixed_dofs: np.ndarray | None,
    M: np.ndarray | sp.spmatrix | None,
    K: np.ndarray | sp.spmatrix | None,
    f_ext_func: Callable[[float], np.ndarray] | None,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray] | None = None,
    element_data_func: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
) -> StepResult:
    """1ステップのソルバー結果を StepResult に変換する."""
    # ソルバー結果から配列を取得
    time_arr = solver_result.time
    disp_hist = solver_result.displacement
    vel_hist = solver_result.velocity
    acc_hist = solver_result.acceleration
    n_steps_solver = len(time_arr) - 1

    # 収束情報
    converged = True
    if hasattr(solver_result, "converged"):
        converged = solver_result.converged
    iter_per_step = None
    if hasattr(solver_result, "iterations_per_step"):
        iter_per_step = solver_result.iterations_per_step

    # --- インクリメント結果の構築 ---
    increments: list[IncrementResult] = []
    for i in range(n_steps_solver):
        dt_actual = float(time_arr[i + 1] - time_arr[i])
        n_iter = iter_per_step[i] if iter_per_step and i < len(iter_per_step) else 1
        inc = IncrementResult(
            increment_index=i,
            time=float(time_arr[i + 1]),
            dt=dt_actual,
            displacement=disp_hist[i + 1].copy(),
            velocity=vel_hist[i + 1].copy(),
            acceleration=acc_hist[i + 1].copy(),
            converged=True,
            iterations=n_iter,
        )
        increments.append(inc)

    # --- フィールド出力フレームの構築 ---
    frames: list[Frame] = []
    if step.field_output is not None:
        fo = step.field_output
        frame_times = np.linspace(0.0, step.total_time, fo.num + 1)[1:]  # 0 は除外
        # 初期状態もフレーム0として追加
        elem_data_0 = element_data_func(disp_hist[0]) if element_data_func is not None else {}
        frames.append(
            Frame(
                frame_index=0,
                time=start_time,
                displacement=disp_hist[0].copy(),
                velocity=vel_hist[0].copy() if "V" in fo.variables else None,
                acceleration=acc_hist[0].copy() if "A" in fo.variables else None,
                element_data=elem_data_0,
            )
        )
        for fi, ft in enumerate(frame_times):
            # 最も近いインクリメントを探す or 内挿
            frame_data = _interpolate_at_time(ft, time_arr, disp_hist, vel_hist, acc_hist)
            elem_data = element_data_func(frame_data[0]) if element_data_func is not None else {}
            frames.append(
                Frame(
                    frame_index=fi + 1,
                    time=start_time + ft,
                    displacement=frame_data[0],
                    velocity=frame_data[1] if "V" in fo.variables else None,
                    acceleration=frame_data[2] if "A" in fo.variables else None,
                    element_data=elem_data,
                )
            )

    # --- ヒストリ出力の構築 ---
    history: dict[str, dict[str, np.ndarray]] = {}
    history_times = np.array([])
    if step.history_output is not None:
        ho = step.history_output
        # ヒストリ出力時刻を生成
        n_history = int(np.ceil(step.total_time / ho.dt))
        history_times_relative = np.linspace(0.0, step.total_time, n_history + 1)
        history_times = start_time + history_times_relative

        # 各 node_set に対してヒストリデータを記録
        for nset_name, nset_nodes in ho.node_sets.items():
            nset_nodes_arr = np.asarray(nset_nodes, dtype=int)
            history[nset_name] = {}

            for var in ho.variables:
                if var in NODAL_VARIABLES:
                    data = _extract_history_nodal(
                        var,
                        history_times_relative,
                        time_arr,
                        disp_hist,
                        vel_hist,
                        acc_hist,
                        nset_nodes_arr,
                        ndof_per_node,
                        fixed_dofs=fixed_dofs,
                        M=M,
                        K=K,
                        f_ext_func=f_ext_func,
                        assemble_internal_force=assemble_internal_force,
                    )
                    history[nset_name][var] = data
                elif var in ENERGY_VARIABLES:
                    data = _extract_history_energy(
                        var,
                        history_times_relative,
                        time_arr,
                        disp_hist,
                        vel_hist,
                        M=M,
                        K=K,
                    )
                    history[nset_name][var] = data

    return StepResult(
        step=step,
        step_index=step_index,
        start_time=start_time,
        increments=increments,
        frames=frames,
        history=history,
        history_times=history_times,
        converged=converged,
    )


def _interpolate_at_time(
    t: float,
    time_arr: np.ndarray,
    disp_hist: np.ndarray,
    vel_hist: np.ndarray,
    acc_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """指定時刻のデータを線形内挿で取得する."""
    if t <= time_arr[0]:
        return disp_hist[0].copy(), vel_hist[0].copy(), acc_hist[0].copy()
    if t >= time_arr[-1]:
        return disp_hist[-1].copy(), vel_hist[-1].copy(), acc_hist[-1].copy()

    # 二分探索で区間を特定
    idx = int(np.searchsorted(time_arr, t, side="right")) - 1
    idx = max(0, min(idx, len(time_arr) - 2))

    t0, t1 = time_arr[idx], time_arr[idx + 1]
    dt = t1 - t0
    if dt < 1e-30:
        alpha = 0.0
    else:
        alpha = (t - t0) / dt

    u = (1.0 - alpha) * disp_hist[idx] + alpha * disp_hist[idx + 1]
    v = (1.0 - alpha) * vel_hist[idx] + alpha * vel_hist[idx + 1]
    a = (1.0 - alpha) * acc_hist[idx] + alpha * acc_hist[idx + 1]

    return u, v, a


def _extract_history_nodal(
    var: str,
    history_times: np.ndarray,
    time_arr: np.ndarray,
    disp_hist: np.ndarray,
    vel_hist: np.ndarray,
    acc_hist: np.ndarray,
    node_indices: np.ndarray,
    ndof_per_node: int,
    *,
    fixed_dofs: np.ndarray | None = None,
    M: np.ndarray | sp.spmatrix | None = None,
    K: np.ndarray | sp.spmatrix | None = None,
    f_ext_func: Callable[[float], np.ndarray] | None = None,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """節点変数のヒストリデータを抽出する.

    Args:
        var: 変数名 ("U", "V", "A", "RF", "CF")
        history_times: ヒストリ出力時刻（ステップ相対）
        time_arr: ソルバーの時刻配列
        disp_hist, vel_hist, acc_hist: ソルバーの変位/速度/加速度履歴
        node_indices: 対象節点インデックス
        ndof_per_node: 1節点あたりのDOF数
        fixed_dofs: 拘束DOF
        M: 質量行列（RF計算用）
        K: 剛性行列（RF計算用）
        f_ext_func: 外力関数（CF出力用）
        assemble_internal_force: 非線形内力コールバック u → f_int(u)。
            指定時は RF = f_int(u) + M·a で計算（線形の K·u + M·a より正確）。

    Returns:
        (n_times, n_nodes * ndof_per_node) のデータ配列
    """
    n_times = len(history_times)
    dof_indices = _node_to_dof_indices(node_indices, ndof_per_node)
    n_dofs = len(dof_indices)
    data = np.zeros((n_times, n_dofs), dtype=float)

    for i, t in enumerate(history_times):
        u_full, v_full, a_full = _interpolate_at_time(t, time_arr, disp_hist, vel_hist, acc_hist)

        if var == "U":
            data[i] = u_full[dof_indices]
        elif var == "V":
            data[i] = v_full[dof_indices]
        elif var == "A":
            data[i] = a_full[dof_indices]
        elif var == "RF":
            # 反力計算
            if M is not None:
                M_d = _to_dense(M)
                if assemble_internal_force is not None:
                    # 非線形: RF = f_int(u) + M·a
                    f_int = assemble_internal_force(u_full)
                    rf_full = f_int + M_d @ a_full
                elif K is not None:
                    # 線形: RF = K·u + M·a
                    K_d = _to_dense(K)
                    rf_full = K_d @ u_full + M_d @ a_full
                else:
                    rf_full = M_d @ a_full
                # 拘束DOFのみ反力を持つ
                if fixed_dofs is not None:
                    mask = np.zeros(len(u_full), dtype=bool)
                    mask[fixed_dofs] = True
                    rf_filtered = np.where(mask, rf_full, 0.0)
                    data[i] = rf_filtered[dof_indices]
                else:
                    data[i] = rf_full[dof_indices]
            else:
                data[i] = 0.0
        elif var == "CF":
            # 集中外力
            if f_ext_func is not None:
                f = f_ext_func(t)
                data[i] = f[dof_indices]
            else:
                data[i] = 0.0

    return data


def _extract_history_energy(
    var: str,
    history_times: np.ndarray,
    time_arr: np.ndarray,
    disp_hist: np.ndarray,
    vel_hist: np.ndarray,
    *,
    M: np.ndarray | sp.spmatrix | None = None,
    K: np.ndarray | sp.spmatrix | None = None,
) -> np.ndarray:
    """エネルギー変数のヒストリデータを抽出する.

    Args:
        var: "ALLIE" or "ALLKE"

    Returns:
        (n_times,) のエネルギー時刻歴
    """
    n_times = len(history_times)
    data = np.zeros(n_times, dtype=float)

    for i, t in enumerate(history_times):
        u, v, _ = _interpolate_at_time(t, time_arr, disp_hist, vel_hist, np.zeros_like(disp_hist))

        if var == "ALLKE" and M is not None:
            M_d = _to_dense(M)
            data[i] = 0.5 * float(v @ M_d @ v)
        elif var == "ALLIE" and K is not None:
            K_d = _to_dense(K)
            data[i] = 0.5 * float(u @ K_d @ u)

    return data


def _node_to_dof_indices(node_indices: np.ndarray, ndof_per_node: int) -> np.ndarray:
    """節点インデックスをDOFインデックスに展開する."""
    nodes = np.asarray(node_indices, dtype=int)
    dofs = np.empty(len(nodes) * ndof_per_node, dtype=int)
    for i, n in enumerate(nodes):
        for d in range(ndof_per_node):
            dofs[i * ndof_per_node + d] = n * ndof_per_node + d
    return dofs


def _to_dense(A: np.ndarray | sp.spmatrix) -> np.ndarray:
    """疎行列を密行列に変換する."""
    if sp.issparse(A):
        return A.toarray()  # type: ignore[union-attr]
    return np.asarray(A, dtype=float)


def mesh_from_abaqus_inp(
    filepath: str,
) -> dict[str, Any]:
    """Abaqus .inp ファイルからメッシュ情報を OutputDatabase 用の形式で読み込む.

    read_abaqus_inp() のパース結果を、build_output_database() / export_vtk() で
    使用できる形式（node_coords, connectivity, node_sets）に変換するブリッジ関数。

    Args:
        filepath: Abaqus .inp ファイルのパス

    Returns:
        辞書 {
            "node_coords": ndarray (n_nodes, 2 or 3),
            "connectivity": [(vtk_cell_type, node_index_array), ...],
            "node_sets": {name: ndarray of node indices},
            "mesh": AbaqusMesh（元のパース結果）,
        }
    """
    from xkep_cae.io.abaqus_inp import read_abaqus_inp

    mesh = read_abaqus_inp(filepath)

    # ノードラベル → 0始まりインデックスへのマッピング
    label_to_idx: dict[int, int] = {}
    for i, node in enumerate(mesh.nodes):
        label_to_idx[node.label] = i

    # 節点座標
    has_z = any(abs(n.z) > 1e-30 for n in mesh.nodes)
    ndim = 3 if has_z else 2
    n_nodes = len(mesh.nodes)
    node_coords = np.zeros((n_nodes, ndim), dtype=float)
    for i, node in enumerate(mesh.nodes):
        node_coords[i, 0] = node.x
        node_coords[i, 1] = node.y
        if ndim == 3:
            node_coords[i, 2] = node.z

    # VTK セルタイプマッピング
    _ELEM_TYPE_TO_VTK: dict[str, int] = {
        "CPS3": 5,  # VTK_TRIANGLE
        "CPE3": 5,
        "CPS4": 9,  # VTK_QUAD
        "CPS4R": 9,
        "CPE4": 9,
        "CPE4R": 9,
        "CPS6": 22,  # VTK_QUADRATIC_TRIANGLE
        "CPE6": 22,
        "B21": 3,  # VTK_LINE
        "B22": 3,
        "B31": 3,
        "B32": 3,
    }

    connectivity: list[tuple[int, np.ndarray]] = []
    for group in mesh.element_groups:
        elem_type_upper = group.elem_type.upper()
        vtk_type = _ELEM_TYPE_TO_VTK.get(elem_type_upper, 3)

        if not group.elements:
            continue

        # 各要素の接続を 0-based インデックスに変換
        elem_rows = []
        for _label, nodes in group.elements:
            row = [label_to_idx.get(n, n - 1) for n in nodes]
            elem_rows.append(row)

        conn_arr = np.array(elem_rows, dtype=int)
        connectivity.append((vtk_type, conn_arr))

    # ノードセット（ラベル → 0-based インデックスに変換）
    node_sets: dict[str, np.ndarray] = {}
    for nset_name, labels in mesh.nsets.items():
        indices = [label_to_idx.get(lbl, lbl - 1) for lbl in labels]
        node_sets[nset_name] = np.array(indices, dtype=int)

    return {
        "node_coords": node_coords,
        "connectivity": connectivity,
        "node_sets": node_sets,
        "mesh": mesh,
    }


def run_transient_steps(
    steps: list[Step],
    M: np.ndarray | sp.spmatrix,
    K: np.ndarray | sp.spmatrix,
    f_ext_funcs: list[Callable[[float], np.ndarray]],
    u0: np.ndarray,
    v0: np.ndarray,
    *,
    C: np.ndarray | sp.spmatrix | None = None,
    fixed_dofs: np.ndarray | None = None,
    node_coords: np.ndarray | None = None,
    connectivity: list[tuple[int, np.ndarray]] | None = None,
    ndof_per_node: int = 2,
    node_sets: dict[str, np.ndarray] | None = None,
    solver: str = "newmark",
    alpha_hht: float = 0.0,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray] | None = None,
) -> OutputDatabase:
    """ステップ列を順次実行し、前ステップの終状態を自動引き継ぎする.

    各ステップのソルバーを実行し、前ステップの最終変位・速度を
    次ステップの初期条件として自動的に引き継ぐ。
    結果は OutputDatabase としてまとめて返す。

    Args:
        steps: ステップ定義のリスト
        M: 質量行列
        K: 剛性行列
        f_ext_funcs: 各ステップの外力関数リスト（len == len(steps)）
        u0: 初期変位ベクトル
        v0: 初期速度ベクトル
        C: 減衰行列（None = 減衰なし）
        fixed_dofs: 拘束 DOF
        node_coords: 節点座標
        connectivity: 要素接続情報
        ndof_per_node: 1節点あたりの DOF 数
        node_sets: 名前付き節点集合
        solver: ソルバー種別 ("newmark" / "central_difference")
        alpha_hht: HHT-α パラメータ（newmark のみ）
        assemble_internal_force: 非線形内力コールバック（RF 計算用）

    Returns:
        OutputDatabase
    """
    from xkep_cae.dynamics import (
        CentralDifferenceConfig,
        TransientConfig,
        solve_central_difference,
        solve_transient,
    )

    if len(steps) != len(f_ext_funcs):
        raise ValueError(
            f"steps ({len(steps)}) と f_ext_funcs ({len(f_ext_funcs)}) の数が一致しない"
        )

    ndof = len(u0)
    C_mat = C if C is not None else np.zeros((ndof, ndof), dtype=float)

    solver_results: list[Any] = []
    u_curr = u0.copy()
    v_curr = v0.copy()

    for step, f_ext in zip(steps, f_ext_funcs, strict=True):
        n_steps_solver = step.n_increments

        if solver == "newmark":
            config = TransientConfig(
                dt=step.dt,
                n_steps=n_steps_solver,
                alpha_hht=alpha_hht,
            )
            result = solve_transient(
                M, C_mat, K, f_ext, u_curr, v_curr, config, fixed_dofs=fixed_dofs
            )
        elif solver == "central_difference":
            config_cd = CentralDifferenceConfig(dt=step.dt, n_steps=n_steps_solver)
            result = solve_central_difference(
                M, C_mat, K, f_ext, u_curr, v_curr, config_cd, fixed_dofs=fixed_dofs
            )
        else:
            raise ValueError(f"未対応のソルバー: {solver}（対応: newmark, central_difference）")

        solver_results.append(result)

        # 次ステップの初期状態 = 今ステップの終状態
        u_curr = result.displacement[-1].copy()
        v_curr = result.velocity[-1].copy()

    return build_output_database(
        steps=steps,
        solver_results=solver_results,
        node_coords=node_coords,
        connectivity=connectivity,
        ndof_per_node=ndof_per_node,
        node_sets=node_sets,
        fixed_dofs=fixed_dofs,
        M=M,
        K=K,
        f_ext_funcs=f_ext_funcs,
        assemble_internal_force=assemble_internal_force,
    )


__all__ = [
    "OutputDatabase",
    "build_output_database",
    "mesh_from_abaqus_inp",
    "run_transient_steps",
]
