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
        frames.append(
            Frame(
                frame_index=0,
                time=start_time,
                displacement=disp_hist[0].copy(),
                velocity=vel_hist[0].copy() if "V" in fo.variables else None,
                acceleration=acc_hist[0].copy() if "A" in fo.variables else None,
            )
        )
        for fi, ft in enumerate(frame_times):
            # 最も近いインクリメントを探す or 内挿
            frame_data = _interpolate_at_time(ft, time_arr, disp_hist, vel_hist, acc_hist)
            frames.append(
                Frame(
                    frame_index=fi + 1,
                    time=start_time + ft,
                    displacement=frame_data[0],
                    velocity=frame_data[1] if "V" in fo.variables else None,
                    acceleration=frame_data[2] if "A" in fo.variables else None,
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
            # 反力 = f_ext - f_int = f_ext - K*u (線形) or M*a + C*v
            # 簡易計算: RF = K*u の拘束DOF成分
            if K is not None and M is not None:
                K_d = _to_dense(K)
                M_d = _to_dense(M)
                rf_full = K_d @ u_full + M_d @ a_full
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


__all__ = [
    "OutputDatabase",
    "build_output_database",
]
