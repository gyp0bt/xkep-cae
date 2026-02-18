"""過渡応答出力インターフェースのテスト.

Step / Increment / Frame / OutputDatabase / Export のテスト。
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from xkep_cae.output import (
    VTK_LINE,
    FieldOutputRequest,
    Frame,
    HistoryOutputRequest,
    IncrementResult,
    InitialConditions,
    OutputDatabase,
    Step,
    StepResult,
    build_output_database,
    export_frames_csv,
    export_history_csv,
    export_json,
    export_vtk,
)

# ====================================================================
# ヘルパー: ダミーのソルバー結果を生成
# ====================================================================


class _DummySolverResult:
    """ソルバー結果のモック（TransientResult 等と同等のインターフェース）."""

    def __init__(
        self,
        time: np.ndarray,
        displacement: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        converged: bool = True,
        iterations_per_step: list[int] | None = None,
    ):
        self.time = time
        self.displacement = displacement
        self.velocity = velocity
        self.acceleration = acceleration
        self.converged = converged
        self.iterations_per_step = iterations_per_step


def _make_sdof_result(
    omega: float = 10.0, n_steps: int = 100, dt: float = 0.01
) -> tuple[_DummySolverResult, np.ndarray, np.ndarray]:
    """SDOF 自由振動の解析解からダミー結果を生成.

    u(t) = A*cos(ω*t), v(t) = -A*ω*sin(ω*t), a(t) = -A*ω²*cos(ω*t)

    Returns:
        (solver_result, M, K) 質量行列と剛性行列も返す
    """
    A = 1.0
    time = np.linspace(0, dt * n_steps, n_steps + 1)
    u = A * np.cos(omega * time)
    v = -A * omega * np.sin(omega * time)
    a = -A * omega**2 * np.cos(omega * time)

    disp = u.reshape(-1, 1)  # (n+1, 1)
    vel = v.reshape(-1, 1)
    acc = a.reshape(-1, 1)

    M = np.array([[1.0]])
    K = np.array([[omega**2]])

    return _DummySolverResult(time, disp, vel, acc), M, K


def _make_beam_result(
    n_nodes: int = 11, n_steps: int = 50, dt: float = 0.01
) -> tuple[_DummySolverResult, np.ndarray, np.ndarray]:
    """梁要素のダミー結果を生成（2D、3DOF/node）.

    Returns:
        (solver_result, node_coords, connectivity)
    """
    ndof_per_node = 3
    ndof = n_nodes * ndof_per_node
    time = np.linspace(0, dt * n_steps, n_steps + 1)

    # 正弦波変位
    disp = np.zeros((n_steps + 1, ndof))
    vel = np.zeros((n_steps + 1, ndof))
    acc = np.zeros((n_steps + 1, ndof))

    for i, t in enumerate(time):
        for node in range(n_nodes):
            x = node / (n_nodes - 1)
            # y方向変位: sin(π*x)*sin(ω*t)
            omega = 2 * np.pi
            uy = np.sin(np.pi * x) * np.sin(omega * t) * 0.1
            vy = np.sin(np.pi * x) * omega * np.cos(omega * t) * 0.1
            ay = -np.sin(np.pi * x) * omega**2 * np.sin(omega * t) * 0.1

            disp[i, node * ndof_per_node + 1] = uy
            vel[i, node * ndof_per_node + 1] = vy
            acc[i, node * ndof_per_node + 1] = ay

    node_coords = np.zeros((n_nodes, 2))
    node_coords[:, 0] = np.linspace(0, 1, n_nodes)

    # 要素接続（2節点梁）
    conn = np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])

    return (
        _DummySolverResult(time, disp, vel, acc),
        node_coords,
        conn,
    )


# ====================================================================
# Step データモデルのテスト
# ====================================================================


class TestStep:
    """Step データモデルのテスト."""

    def test_basic_creation(self):
        step = Step(name="step-1", total_time=1.0, dt=0.01)
        assert step.name == "step-1"
        assert step.total_time == 1.0
        assert step.dt == 0.01
        assert step.n_increments == 100

    def test_n_increments_ceil(self):
        step = Step(name="test", total_time=1.0, dt=0.03)
        assert step.n_increments == 34  # ceil(1.0/0.03) = 34

    def test_invalid_total_time(self):
        with pytest.raises(ValueError, match="total_time"):
            Step(name="bad", total_time=0.0, dt=0.01)

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt は正値"):
            Step(name="bad", total_time=1.0, dt=-0.01)

    def test_dt_exceeds_total_time(self):
        with pytest.raises(ValueError, match="dt.*total_time"):
            Step(name="bad", total_time=0.1, dt=0.5)

    def test_with_output_requests(self):
        ho = HistoryOutputRequest(dt=0.01, variables=["U", "RF"])
        fo = FieldOutputRequest(num=10, variables=["U", "V"])
        step = Step(
            name="step-1",
            total_time=1.0,
            dt=0.01,
            history_output=ho,
            field_output=fo,
        )
        assert step.history_output is not None
        assert step.field_output is not None
        assert step.field_output.num == 10


class TestIncrementResult:
    """IncrementResult のテスト."""

    def test_creation(self):
        inc = IncrementResult(
            increment_index=0,
            time=0.01,
            dt=0.01,
            displacement=np.zeros(4),
            velocity=np.zeros(4),
            acceleration=np.zeros(4),
        )
        assert inc.increment_index == 0
        assert inc.converged is True
        assert inc.iterations == 1


class TestFrame:
    """Frame のテスト."""

    def test_creation(self):
        frame = Frame(
            frame_index=0,
            time=0.0,
            displacement=np.zeros(4),
        )
        assert frame.frame_index == 0
        assert frame.velocity is None

    def test_with_velocity(self):
        frame = Frame(
            frame_index=1,
            time=0.5,
            displacement=np.ones(4),
            velocity=np.ones(4) * 0.5,
        )
        assert frame.velocity is not None


# ====================================================================
# OutputRequest のテスト
# ====================================================================


class TestHistoryOutputRequest:
    """HistoryOutputRequest のテスト."""

    def test_basic(self):
        ho = HistoryOutputRequest(
            dt=0.01,
            variables=["U", "RF", "ALLIE", "ALLKE"],
            node_sets={"refmove": [0, 5, 10]},
        )
        assert ho.dt == 0.01
        assert len(ho.variables) == 4
        assert "refmove" in ho.node_sets

    def test_invalid_variable(self):
        with pytest.raises(ValueError, match="未対応"):
            HistoryOutputRequest(dt=0.01, variables=["INVALID"])

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt は正値"):
            HistoryOutputRequest(dt=0.0, variables=["U"])


class TestFieldOutputRequest:
    """FieldOutputRequest のテスト."""

    def test_basic(self):
        fo = FieldOutputRequest(num=15, variables=["U", "V", "A"])
        assert fo.num == 15
        assert fo.node_sets is None

    def test_invalid_num(self):
        with pytest.raises(ValueError, match="num は1以上"):
            FieldOutputRequest(num=0)


# ====================================================================
# InitialConditions のテスト
# ====================================================================


class TestInitialConditions:
    """InitialConditions のテスト."""

    def test_build_velocity(self):
        ic = InitialConditions()
        ic.add(type="velocity", node_indices=[0, 1, 2], dof=0, value=10.0)

        u0, v0 = ic.build_initial_vectors(ndof_total=9, ndof_per_node=3)

        assert np.allclose(u0, 0.0)
        assert v0[0] == 10.0  # node 0, dof 0
        assert v0[3] == 10.0  # node 1, dof 0
        assert v0[6] == 10.0  # node 2, dof 0
        assert v0[1] == 0.0  # node 0, dof 1

    def test_build_displacement(self):
        ic = InitialConditions()
        ic.add(type="displacement", node_indices=[1], dof=1, value=0.5)

        u0, v0 = ic.build_initial_vectors(ndof_total=6, ndof_per_node=2)

        assert u0[3] == 0.5  # node 1, dof 1
        assert np.allclose(v0, 0.0)

    def test_multiple_entries(self):
        ic = InitialConditions()
        ic.add(type="velocity", node_indices=[0], dof=0, value=5.0)
        ic.add(type="displacement", node_indices=[1], dof=0, value=1.0)

        u0, v0 = ic.build_initial_vectors(ndof_total=4, ndof_per_node=2)

        assert v0[0] == 5.0
        assert u0[2] == 1.0

    def test_out_of_range(self):
        ic = InitialConditions()
        ic.add(type="velocity", node_indices=[10], dof=0, value=1.0)

        with pytest.raises(ValueError, match="範囲外"):
            ic.build_initial_vectors(ndof_total=6, ndof_per_node=2)


# ====================================================================
# OutputDatabase + build のテスト
# ====================================================================


class TestBuildOutputDatabase:
    """build_output_database のテスト."""

    def test_single_step_history(self):
        """1ステップのヒストリ出力."""
        result, M, K = _make_sdof_result(omega=10.0, n_steps=100, dt=0.01)

        step = Step(
            name="step-1",
            total_time=1.0,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.1,
                variables=["U", "ALLKE", "ALLIE"],
                node_sets={"monitor": [0]},
            ),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_sets={"monitor": np.array([0])},
            M=M,
            K=K,
        )

        assert db.n_steps == 1
        sr = db.step_results[0]
        assert sr.converged is True
        assert len(sr.increments) == 100

        # ヒストリデータが存在
        assert "monitor" in sr.history
        assert "U" in sr.history["monitor"]
        assert "ALLKE" in sr.history["monitor"]
        assert "ALLIE" in sr.history["monitor"]

        # ヒストリ時刻の数
        assert len(sr.history_times) == 11  # 0.0, 0.1, ..., 1.0

    def test_single_step_field(self):
        """1ステップのフィールド出力."""
        result, M, K = _make_sdof_result(omega=10.0, n_steps=100, dt=0.01)

        step = Step(
            name="step-1",
            total_time=1.0,
            dt=0.01,
            field_output=FieldOutputRequest(
                num=5,
                variables=["U", "V", "A"],
            ),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
            M=M,
            K=K,
        )

        sr = db.step_results[0]
        # 初期フレーム + 5フレーム = 6
        assert len(sr.frames) == 6
        assert sr.frames[0].time == 0.0
        assert sr.frames[-1].time == pytest.approx(1.0, abs=1e-10)

    def test_energy_conservation(self):
        """ALLKE + ALLIE ≈ const（SDOF自由振動）."""
        omega = 10.0
        result, M, K = _make_sdof_result(omega=omega, n_steps=200, dt=0.005)

        step = Step(
            name="step-1",
            total_time=1.0,
            dt=0.005,
            history_output=HistoryOutputRequest(
                dt=0.05,
                variables=["ALLKE", "ALLIE"],
                node_sets={"all": [0]},
            ),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            M=M,
            K=K,
            node_sets={"all": np.array([0])},
        )

        sr = db.step_results[0]
        ke = sr.history["all"]["ALLKE"]
        ie = sr.history["all"]["ALLIE"]
        total = ke + ie

        # 全エネルギーが保存（解析解なので厳密一致）
        assert np.allclose(total, total[0], rtol=1e-6)

    def test_multi_step(self):
        """2ステップの連結."""
        result1, M, K = _make_sdof_result(omega=10.0, n_steps=50, dt=0.01)
        result2, _, _ = _make_sdof_result(omega=10.0, n_steps=50, dt=0.01)

        steps = [
            Step(name="step-1", total_time=0.5, dt=0.01),
            Step(name="step-2", total_time=0.5, dt=0.01),
        ]

        db = build_output_database(
            steps=steps,
            solver_results=[result1, result2],
            ndof_per_node=1,
        )

        assert db.n_steps == 2
        assert db.total_time() == pytest.approx(1.0)
        assert db.step_results[0].start_time == 0.0
        assert db.step_results[1].start_time == pytest.approx(0.5)

    def test_mismatched_steps_results(self):
        """ステップ数と結果数の不一致."""
        result, _, _ = _make_sdof_result()
        with pytest.raises(ValueError, match="一致しない"):
            build_output_database(
                steps=[Step(name="s1", total_time=1.0, dt=0.01)],
                solver_results=[result, result],
            )


# ====================================================================
# CSV エクスポートのテスト
# ====================================================================


class TestExportCSV:
    """CSV エクスポートのテスト."""

    def test_history_csv(self, tmp_path):
        """ヒストリ出力 CSV の生成."""
        result, M, K = _make_sdof_result(n_steps=100, dt=0.01)
        step = Step(
            name="step1",
            total_time=1.0,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.1,
                variables=["U", "ALLKE"],
                node_sets={"monitor": [0]},
            ),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_sets={"monitor": np.array([0])},
            M=M,
            K=K,
        )

        files = export_history_csv(db, tmp_path)
        assert len(files) == 1
        assert Path(files[0]).exists()

        # CSV の中身を確認
        content = Path(files[0]).read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 12  # header + 11 data rows
        assert "time" in lines[0]
        assert "U_0_d1" in lines[0]
        assert "ALLKE" in lines[0]

    def test_frames_csv(self, tmp_path):
        """フレーム出力 CSV の生成."""
        beam_result, node_coords, conn = _make_beam_result(n_nodes=5, n_steps=20)
        step = Step(
            name="step1",
            total_time=0.2,
            dt=0.01,
            field_output=FieldOutputRequest(num=4, variables=["U", "V"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[beam_result],
            ndof_per_node=3,
            node_coords=node_coords,
        )

        files = export_frames_csv(db, tmp_path)
        # summary + 5 frame files (0 initial + 4)
        assert len(files) == 6
        for f in files:
            assert Path(f).exists()


# ====================================================================
# JSON エクスポートのテスト
# ====================================================================


class TestExportJSON:
    """JSON エクスポートのテスト."""

    def test_basic_json(self, tmp_path):
        """基本的な JSON エクスポート."""
        result, M, K = _make_sdof_result(n_steps=50, dt=0.01)
        step = Step(
            name="step1",
            total_time=0.5,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.1,
                variables=["U"],
                node_sets={"monitor": [0]},
            ),
            field_output=FieldOutputRequest(num=3, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
            node_sets={"monitor": np.array([0])},
            M=M,
            K=K,
        )

        filepath = export_json(db, tmp_path)
        assert Path(filepath).exists()

        with open(filepath) as fh:
            data = json.load(fh)

        assert data["metadata"]["n_steps"] == 1
        assert data["metadata"]["ndof_per_node"] == 1
        assert len(data["steps"]) == 1
        assert data["steps"][0]["name"] == "step1"
        assert "frames" in data["steps"][0]
        assert "history" in data["steps"][0]


# ====================================================================
# VTK エクスポートのテスト
# ====================================================================


class TestExportVTK:
    """VTK/VTU エクスポートのテスト."""

    def test_beam_vtk(self, tmp_path):
        """梁要素の VTK 出力."""
        beam_result, node_coords, conn = _make_beam_result(n_nodes=5, n_steps=20)
        step = Step(
            name="step1",
            total_time=0.2,
            dt=0.01,
            field_output=FieldOutputRequest(num=4, variables=["U", "V", "A"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[beam_result],
            ndof_per_node=3,
            node_coords=node_coords,
            connectivity=[(VTK_LINE, conn)],
        )

        pvd_path = export_vtk(db, tmp_path)
        assert Path(pvd_path).exists()

        # .pvd ファイルの検証
        tree = ET.parse(pvd_path)
        root = tree.getroot()
        assert root.tag == "VTKFile"
        assert root.get("type") == "Collection"

        datasets = root.findall(".//DataSet")
        assert len(datasets) == 5  # 1 initial + 4 frames

        # .vtu ファイルの検証
        for ds in datasets:
            vtu_name = ds.get("file")
            vtu_path = tmp_path / vtu_name
            assert vtu_path.exists()

            vtu_tree = ET.parse(vtu_path)
            vtu_root = vtu_tree.getroot()
            assert vtu_root.get("type") == "UnstructuredGrid"

            piece = vtu_root.find(".//Piece")
            assert piece is not None
            assert piece.get("NumberOfPoints") == "5"
            assert piece.get("NumberOfCells") == "4"

            # PointData の検証
            point_data = piece.find("PointData")
            assert point_data is not None
            data_arrays = point_data.findall("DataArray")
            names = {da.get("Name") for da in data_arrays}
            assert "U" in names
            assert "U_magnitude" in names
            assert "V" in names
            assert "A" in names

    def test_vtk_without_coords_raises(self, tmp_path):
        """node_coords なしで VTK 出力するとエラー."""
        db = OutputDatabase()
        with pytest.raises(ValueError, match="node_coords"):
            export_vtk(db, tmp_path)

    def test_vtk_timesteps(self, tmp_path):
        """VTK タイムステップの正確性."""
        result, _, _ = _make_sdof_result(n_steps=10, dt=0.1)
        step = Step(
            name="s1",
            total_time=1.0,
            dt=0.1,
            field_output=FieldOutputRequest(num=5, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
            connectivity=[(VTK_LINE, np.array([[0, 0]]))],
        )

        pvd_path = export_vtk(db, tmp_path)
        tree = ET.parse(pvd_path)
        datasets = tree.findall(".//DataSet")

        timesteps = [float(ds.get("timestep")) for ds in datasets]
        assert timesteps[0] == pytest.approx(0.0)
        assert timesteps[-1] == pytest.approx(1.0)


# ====================================================================
# StepResult のテスト
# ====================================================================


class TestStepResult:
    """StepResult のテスト."""

    def test_creation(self):
        step = Step(name="test", total_time=1.0, dt=0.01)
        sr = StepResult(step=step, step_index=0)
        assert sr.converged is True
        assert len(sr.increments) == 0
        assert len(sr.frames) == 0


# ====================================================================
# OutputDatabase のテスト
# ====================================================================


class TestOutputDatabase:
    """OutputDatabase のテスト."""

    def test_empty(self):
        db = OutputDatabase()
        assert db.n_steps == 0
        assert db.n_nodes == 0
        assert db.total_time() == 0.0

    def test_all_frames(self):
        """全フレームの取得."""
        result, _, _ = _make_sdof_result(n_steps=20, dt=0.05)
        step = Step(
            name="s1",
            total_time=1.0,
            dt=0.05,
            field_output=FieldOutputRequest(num=3, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
        )

        all_frames = db.all_frames()
        assert len(all_frames) == 4  # initial + 3

    def test_properties(self):
        db = OutputDatabase(
            node_coords=np.zeros((5, 3)),
            ndof_per_node=6,
        )
        assert db.n_nodes == 5
        assert db.ndim == 3


# ====================================================================
# 統合テスト: dynamics ソルバーとの連携
# ====================================================================


class TestDynamicsIntegration:
    """dynamics モジュールのソルバー結果との統合テスト."""

    def test_with_solve_transient(self):
        """solve_transient の結果から OutputDatabase を構築."""
        from xkep_cae.dynamics import TransientConfig, solve_transient

        # SDOF: m*a + k*u = 0, u(0)=1, v(0)=0
        m, k = 1.0, 100.0
        M = np.array([[m]])
        K = np.array([[k]])
        C = np.zeros((1, 1))

        config = TransientConfig(dt=0.01, n_steps=100)
        result = solve_transient(M, C, K, np.zeros(1), np.array([1.0]), np.array([0.0]), config)

        step = Step(
            name="free_vibration",
            total_time=1.0,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.05,
                variables=["U", "V", "ALLKE", "ALLIE"],
                node_sets={"all": [0]},
            ),
            field_output=FieldOutputRequest(num=10, variables=["U", "V"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
            node_sets={"all": np.array([0])},
            M=M,
            K=K,
        )

        # 検証: エネルギー保存
        sr = db.step_results[0]
        ke = sr.history["all"]["ALLKE"]
        ie = sr.history["all"]["ALLIE"]
        total = ke + ie
        assert np.allclose(total, total[0], rtol=1e-3)

        # フレーム数
        assert len(sr.frames) == 11  # initial + 10

    def test_with_central_difference(self):
        """Central Difference の結果から OutputDatabase を構築."""
        from xkep_cae.dynamics import (
            CentralDifferenceConfig,
            critical_time_step,
            solve_central_difference,
        )

        m, k = 1.0, 100.0
        M = np.array([[m]])
        K = np.array([[k]])
        C = np.zeros((1, 1))

        dt_cr = critical_time_step(M, K)
        dt = dt_cr * 0.5
        n_steps = int(1.0 / dt)

        config = CentralDifferenceConfig(dt=dt, n_steps=n_steps)
        result = solve_central_difference(
            M, C, K, np.zeros(1), np.array([1.0]), np.array([0.0]), config
        )

        step = Step(
            name="explicit",
            total_time=dt * n_steps,
            dt=dt,
            field_output=FieldOutputRequest(num=5, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
        )

        assert db.n_steps == 1
        assert len(db.step_results[0].frames) == 6


# ====================================================================
# 実用的なワークフローのテスト
# ====================================================================


class TestWorkflow:
    """実際の利用シナリオに近いワークフローのテスト."""

    def test_full_workflow_with_export(self, tmp_path):
        """完全なワークフロー: ソルバー実行 → DB構築 → 3形式エクスポート."""
        from xkep_cae.dynamics import TransientConfig, solve_transient

        # 2-DOF 系
        M = np.diag([1.0, 1.0])
        K = np.array([[200.0, -100.0], [-100.0, 100.0]])
        C = np.zeros((2, 2))

        config = TransientConfig(dt=0.005, n_steps=200)
        u0 = np.array([1.0, 0.0])
        v0 = np.zeros(2)
        result = solve_transient(M, C, K, np.zeros(2), u0, v0, config)

        step = Step(
            name="step1",
            total_time=1.0,
            dt=0.005,
            history_output=HistoryOutputRequest(
                dt=0.05,
                variables=["U", "ALLKE", "ALLIE"],
                node_sets={"nodes": [0, 1]},
            ),
            field_output=FieldOutputRequest(num=10, variables=["U", "V"]),
        )

        node_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0, 1]])

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=node_coords,
            connectivity=[(VTK_LINE, conn)],
            node_sets={"nodes": np.array([0, 1])},
            M=M,
            K=K,
        )

        # CSV エクスポート
        csv_files = export_history_csv(db, tmp_path / "csv")
        assert len(csv_files) > 0

        frame_files = export_frames_csv(db, tmp_path / "csv")
        assert len(frame_files) > 0

        # JSON エクスポート
        json_file = export_json(db, tmp_path / "json")
        assert Path(json_file).exists()

        # VTK エクスポート
        pvd_file = export_vtk(db, tmp_path / "vtk")
        assert Path(pvd_file).exists()

    def test_initial_conditions_workflow(self):
        """初期条件を使ったワークフロー."""
        from xkep_cae.dynamics import TransientConfig, solve_transient

        # 初期条件を設定
        ic = InitialConditions()
        ic.add(type="velocity", node_indices=[0], dof=0, value=5.0)

        ndof_per_node = 1
        ndof_total = 2
        u0, v0 = ic.build_initial_vectors(ndof_total, ndof_per_node)

        assert v0[0] == 5.0
        assert v0[1] == 0.0

        # ソルバー実行
        M = np.diag([1.0, 1.0])
        K = np.array([[100.0, -50.0], [-50.0, 50.0]])
        C = np.zeros((2, 2))

        config = TransientConfig(dt=0.01, n_steps=100)
        result = solve_transient(M, C, K, np.zeros(2), u0, v0, config)

        assert result.displacement.shape == (101, 2)

    def test_multi_step_workflow(self, tmp_path):
        """複数ステップの連結ワークフロー."""
        from xkep_cae.dynamics import TransientConfig, solve_transient

        M = np.array([[1.0]])
        K = np.array([[100.0]])
        C = np.zeros((1, 1))

        # Step 1: 自由振動
        config1 = TransientConfig(dt=0.01, n_steps=50)
        result1 = solve_transient(M, C, K, np.zeros(1), np.array([1.0]), np.array([0.0]), config1)

        # Step 2: 初期条件を step1 の終了状態から（ここでは簡略化）
        config2 = TransientConfig(dt=0.005, n_steps=100)
        result2 = solve_transient(
            M, C, K, np.zeros(1), result1.displacement[-1], result1.velocity[-1], config2
        )

        steps = [
            Step(
                name="step1",
                total_time=0.5,
                dt=0.01,
                field_output=FieldOutputRequest(num=5, variables=["U"]),
            ),
            Step(
                name="step2",
                total_time=0.5,
                dt=0.005,
                field_output=FieldOutputRequest(num=5, variables=["U"]),
            ),
        ]

        db = build_output_database(
            steps=steps,
            solver_results=[result1, result2],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
            connectivity=[(VTK_LINE, np.array([[0, 0]]))],
        )

        assert db.n_steps == 2
        assert db.step_results[1].start_time == pytest.approx(0.5)

        # VTK出力
        pvd_path = export_vtk(db, tmp_path)
        assert Path(pvd_path).exists()

        # 全フレーム数
        all_frames = db.all_frames()
        assert len(all_frames) == 12  # (1+5) + (1+5)


# ====================================================================
# run_transient_steps のテスト
# ====================================================================


class TestRunTransientSteps:
    """run_transient_steps() のテスト."""

    def test_single_step(self):
        """1ステップの自動実行."""
        from xkep_cae.output import run_transient_steps

        M = np.array([[1.0]])
        K = np.array([[100.0]])
        u0 = np.array([1.0])
        v0 = np.array([0.0])

        step = Step(
            name="step1",
            total_time=0.5,
            dt=0.01,
            field_output=FieldOutputRequest(num=5, variables=["U"]),
        )

        db = run_transient_steps(
            steps=[step],
            M=M,
            K=K,
            f_ext_funcs=[np.zeros(1)],
            u0=u0,
            v0=v0,
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
        )

        assert db.n_steps == 1
        assert len(db.step_results[0].frames) == 6  # initial + 5

    def test_multi_step_state_carryover(self):
        """2ステップの状態引き継ぎ."""
        from xkep_cae.output import run_transient_steps

        M = np.array([[1.0]])
        K = np.array([[100.0]])
        u0 = np.array([1.0])
        v0 = np.array([0.0])

        steps = [
            Step(name="s1", total_time=0.5, dt=0.01),
            Step(name="s2", total_time=0.5, dt=0.01),
        ]

        db = run_transient_steps(
            steps=steps,
            M=M,
            K=K,
            f_ext_funcs=[np.zeros(1), np.zeros(1)],
            u0=u0,
            v0=v0,
            ndof_per_node=1,
        )

        assert db.n_steps == 2
        assert db.step_results[1].start_time == pytest.approx(0.5)

        # 連続性の確認: step2 は step1 の終了状態から始まるので、連続的に変化する
        s2_first = db.step_results[1].increments[0].displacement
        assert not np.allclose(s2_first, 0.0)

    def test_mismatched_steps_f_ext_raises(self):
        """steps と f_ext_funcs の数不一致でエラー."""
        from xkep_cae.output import run_transient_steps

        M = np.array([[1.0]])
        K = np.array([[100.0]])

        with pytest.raises(ValueError, match="一致しない"):
            run_transient_steps(
                steps=[Step(name="s1", total_time=0.5, dt=0.01)],
                M=M,
                K=K,
                f_ext_funcs=[np.zeros(1), np.zeros(1)],
                u0=np.array([0.0]),
                v0=np.array([0.0]),
            )

    def test_central_difference_solver(self):
        """陽解法での実行."""
        from xkep_cae.dynamics import critical_time_step
        from xkep_cae.output import run_transient_steps

        M = np.array([[1.0]])
        K = np.array([[100.0]])
        dt_cr = critical_time_step(M, K)
        dt = dt_cr * 0.5

        step = Step(name="explicit", total_time=0.5, dt=dt)

        db = run_transient_steps(
            steps=[step],
            M=M,
            K=K,
            f_ext_funcs=[np.zeros(1)],
            u0=np.array([1.0]),
            v0=np.array([0.0]),
            solver="central_difference",
            ndof_per_node=1,
        )

        assert db.n_steps == 1
        assert db.step_results[0].converged is True

    def test_invalid_solver_raises(self):
        """未対応ソルバーでエラー."""
        from xkep_cae.output import run_transient_steps

        M = np.array([[1.0]])
        K = np.array([[100.0]])

        with pytest.raises(ValueError, match="未対応のソルバー"):
            run_transient_steps(
                steps=[Step(name="s1", total_time=0.5, dt=0.01)],
                M=M,
                K=K,
                f_ext_funcs=[np.zeros(1)],
                u0=np.array([0.0]),
                v0=np.array([0.0]),
                solver="rk4",
            )

    def test_with_history_output(self):
        """ヒストリ出力付きの run_transient_steps."""
        from xkep_cae.output import run_transient_steps

        M = np.array([[1.0]])
        K = np.array([[100.0]])

        step = Step(
            name="step1",
            total_time=1.0,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.1,
                variables=["U", "ALLKE", "ALLIE"],
                node_sets={"all": [0]},
            ),
        )

        db = run_transient_steps(
            steps=[step],
            M=M,
            K=K,
            f_ext_funcs=[np.zeros(1)],
            u0=np.array([1.0]),
            v0=np.array([0.0]),
            ndof_per_node=1,
            node_sets={"all": np.array([0])},
        )

        sr = db.step_results[0]
        assert "all" in sr.history
        ke = sr.history["all"]["ALLKE"]
        ie = sr.history["all"]["ALLIE"]
        total = ke + ie
        # エネルギー保存（Newmark β=1/4 は暗黙的にエネルギー保存）
        assert np.allclose(total, total[0], rtol=1e-3)


# ====================================================================
# 非線形反力計算 (assemble_internal_force) のテスト
# ====================================================================


class TestNonlinearReactionForce:
    """assemble_internal_force を使った非線形反力計算のテスト."""

    def test_linear_vs_nonlinear_rf(self):
        """線形ケースで K·u と assemble_internal_force が一致."""
        result, M, K = _make_sdof_result(omega=10.0, n_steps=100, dt=0.01)

        step = Step(
            name="step1",
            total_time=1.0,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.1,
                variables=["RF"],
                node_sets={"fix": [0]},
            ),
        )

        # 線形 RF (K·u + M·a)
        db_linear = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_sets={"fix": np.array([0])},
            fixed_dofs=np.array([0]),
            M=M,
            K=K,
        )

        # 非線形 RF (f_int(u) + M·a) — 線形の場合 f_int(u) = K·u
        def f_int_linear(u: np.ndarray) -> np.ndarray:
            return K @ u

        db_nonlinear = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_sets={"fix": np.array([0])},
            fixed_dofs=np.array([0]),
            M=M,
            K=K,
            assemble_internal_force=f_int_linear,
        )

        rf_lin = db_linear.step_results[0].history["fix"]["RF"]
        rf_nl = db_nonlinear.step_results[0].history["fix"]["RF"]

        np.testing.assert_allclose(rf_lin, rf_nl, rtol=1e-10)

    def test_nonlinear_rf_without_K(self):
        """K なしでも assemble_internal_force があれば RF 計算可能."""
        result, M, K = _make_sdof_result(omega=10.0, n_steps=50, dt=0.01)

        step = Step(
            name="step1",
            total_time=0.5,
            dt=0.01,
            history_output=HistoryOutputRequest(
                dt=0.1,
                variables=["RF"],
                node_sets={"all": [0]},
            ),
        )

        # 非線形内力: f_int(u) = 2*K*u（意図的に K と異なる非線形応答）
        def f_int_nonlinear(u: np.ndarray) -> np.ndarray:
            return 2.0 * K @ u

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_sets={"all": np.array([0])},
            M=M,
            assemble_internal_force=f_int_nonlinear,
        )

        rf = db.step_results[0].history["all"]["RF"]
        assert rf.shape[0] == 6  # 0.0, 0.1, ..., 0.5
        # f_int(u) + M·a = 2K·u + M·a ≠ 0（K != 線形の場合）
        # 自由振動: M·a = -K·u なので RF = 2K·u - K·u = K·u ≠ 0
        assert np.any(np.abs(rf) > 1e-3)


# ====================================================================
# VTK バイナリ出力のテスト
# ====================================================================


class TestExportVTKBinary:
    """VTK バイナリ出力モードのテスト."""

    def test_binary_output(self, tmp_path):
        """バイナリ VTK 出力."""
        beam_result, node_coords, conn = _make_beam_result(n_nodes=5, n_steps=20)
        step = Step(
            name="step1",
            total_time=0.2,
            dt=0.01,
            field_output=FieldOutputRequest(num=2, variables=["U", "V"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[beam_result],
            ndof_per_node=3,
            node_coords=node_coords,
            connectivity=[(VTK_LINE, conn)],
        )

        pvd_path = export_vtk(db, tmp_path, binary=True)
        assert Path(pvd_path).exists()

        # VTU ファイルの検証
        tree = ET.parse(pvd_path)
        datasets = tree.findall(".//DataSet")
        assert len(datasets) == 3  # initial + 2

        # binary format のチェック
        first_vtu = tmp_path / datasets[0].get("file")
        vtu_tree = ET.parse(first_vtu)
        data_arrays = vtu_tree.findall(".//DataArray")
        for da in data_arrays:
            assert da.get("format") == "binary"

    def test_binary_data_decodable(self, tmp_path):
        """バイナリデータが正しく base64 デコードできる."""
        import base64

        result, M, K = _make_sdof_result(n_steps=10, dt=0.1)
        step = Step(
            name="s1",
            total_time=1.0,
            dt=0.1,
            field_output=FieldOutputRequest(num=2, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=1,
            node_coords=np.array([[0.0]]),
            connectivity=[(VTK_LINE, np.array([[0, 0]]))],
        )

        pvd_path = export_vtk(db, tmp_path, binary=True)
        tree = ET.parse(pvd_path)
        first_vtu = tmp_path / tree.findall(".//DataSet")[0].get("file")

        vtu_tree = ET.parse(first_vtu)
        for da in vtu_tree.findall(".//DataArray"):
            text = da.text.strip()
            # base64 デコードが成功する
            decoded = base64.b64decode(text)
            assert len(decoded) > 0


# ====================================================================
# 要素データ出力 (CellData) のテスト
# ====================================================================


class TestElementData:
    """Frame の element_data と VTK CellData 出力のテスト."""

    def test_frame_element_data(self):
        """Frame に element_data を設定."""
        frame = Frame(
            frame_index=0,
            time=0.0,
            displacement=np.zeros(4),
            element_data={"stress_xx": np.array([1.0, 2.0])},
        )
        assert "stress_xx" in frame.element_data
        assert frame.element_data["stress_xx"].shape == (2,)

    def test_element_data_in_vtk(self, tmp_path):
        """element_data が VTK CellData として出力される."""
        beam_result, node_coords, conn = _make_beam_result(n_nodes=5, n_steps=10)
        n_elems = 4  # 5 nodes - 1

        def elem_data_func(u: np.ndarray) -> dict[str, np.ndarray]:
            return {"stress_xx": np.ones(n_elems) * np.max(np.abs(u))}

        step = Step(
            name="step1",
            total_time=0.1,
            dt=0.01,
            field_output=FieldOutputRequest(num=2, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[beam_result],
            ndof_per_node=3,
            node_coords=node_coords,
            connectivity=[(VTK_LINE, conn)],
            element_data_func=elem_data_func,
        )

        # element_data がフレームに設定されているか
        for frame in db.step_results[0].frames:
            assert "stress_xx" in frame.element_data
            assert frame.element_data["stress_xx"].shape == (n_elems,)

        # VTK 出力
        pvd_path = export_vtk(db, tmp_path)
        tree = ET.parse(pvd_path)
        first_vtu = tmp_path / tree.findall(".//DataSet")[0].get("file")
        vtu_tree = ET.parse(first_vtu)

        # CellData の検証
        cell_data = vtu_tree.find(".//CellData")
        assert cell_data is not None
        stress_arr = cell_data.find("DataArray[@Name='stress_xx']")
        assert stress_arr is not None

    def test_element_data_multi_component(self, tmp_path):
        """多成分要素データ（応力テンソル等）."""
        beam_result, node_coords, conn = _make_beam_result(n_nodes=5, n_steps=10)
        n_elems = 4

        def elem_data_func(u: np.ndarray) -> dict[str, np.ndarray]:
            return {"stress": np.zeros((n_elems, 3))}

        step = Step(
            name="step1",
            total_time=0.1,
            dt=0.01,
            field_output=FieldOutputRequest(num=1, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[beam_result],
            ndof_per_node=3,
            node_coords=node_coords,
            connectivity=[(VTK_LINE, conn)],
            element_data_func=elem_data_func,
        )

        pvd_path = export_vtk(db, tmp_path)
        tree = ET.parse(pvd_path)
        first_vtu = tmp_path / tree.findall(".//DataSet")[0].get("file")
        vtu_tree = ET.parse(first_vtu)

        cell_data = vtu_tree.find(".//CellData")
        assert cell_data is not None
        stress_arr = cell_data.find("DataArray[@Name='stress']")
        assert stress_arr is not None
        assert stress_arr.get("NumberOfComponents") == "3"


# ====================================================================
# Abaqus .inp パーサー統合のテスト
# ====================================================================


class TestMeshFromAbaqusInp:
    """mesh_from_abaqus_inp() のテスト."""

    def test_beam_mesh(self, tmp_path):
        """梁要素の .inp ファイルからメッシュ情報を取得."""
        from xkep_cae.output import mesh_from_abaqus_inp

        inp_content = """\
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 2.0, 0.0
*ELEMENT, TYPE=B21, ELSET=BEAM
1, 1, 2
2, 2, 3
*NSET, NSET=FIX
1
*NSET, NSET=LOAD
3
"""
        inp_path = tmp_path / "beam.inp"
        inp_path.write_text(inp_content)

        result = mesh_from_abaqus_inp(str(inp_path))

        assert result["node_coords"].shape == (3, 2)
        assert len(result["connectivity"]) == 1
        vtk_type, conn_arr = result["connectivity"][0]
        assert vtk_type == 3  # VTK_LINE
        assert conn_arr.shape == (2, 2)

        # node_sets（0-based インデックス）
        assert "FIX" in result["node_sets"]
        assert result["node_sets"]["FIX"][0] == 0  # node 1 → index 0
        assert "LOAD" in result["node_sets"]
        assert result["node_sets"]["LOAD"][0] == 2  # node 3 → index 2

    def test_quad_mesh(self, tmp_path):
        """四角形要素の .inp ファイルからメッシュ情報を取得."""
        from xkep_cae.output import mesh_from_abaqus_inp

        inp_content = """\
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPS4R, ELSET=SOLID
1, 1, 2, 3, 4
"""
        inp_path = tmp_path / "quad.inp"
        inp_path.write_text(inp_content)

        result = mesh_from_abaqus_inp(str(inp_path))

        assert result["node_coords"].shape == (4, 2)
        vtk_type, conn_arr = result["connectivity"][0]
        assert vtk_type == 9  # VTK_QUAD
        assert conn_arr.shape == (1, 4)

    def test_3d_mesh(self, tmp_path):
        """3D 節点座標の処理."""
        from xkep_cae.output import mesh_from_abaqus_inp

        inp_content = """\
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 1.0
*ELEMENT, TYPE=B31, ELSET=BEAM3D
1, 1, 2
2, 2, 3
"""
        inp_path = tmp_path / "beam3d.inp"
        inp_path.write_text(inp_content)

        result = mesh_from_abaqus_inp(str(inp_path))

        assert result["node_coords"].shape == (3, 3)  # 3D
        assert result["node_coords"][2, 2] == 1.0  # z 座標

    def test_integration_with_output_database(self, tmp_path):
        """mesh_from_abaqus_inp → build_output_database → export_vtk の統合."""
        from xkep_cae.output import mesh_from_abaqus_inp

        inp_content = """\
*NODE
1, 0.0, 0.0
2, 0.5, 0.0
3, 1.0, 0.0
*ELEMENT, TYPE=B21, ELSET=BEAM
1, 1, 2
2, 2, 3
*NSET, NSET=FIX
1
"""
        inp_path = tmp_path / "test.inp"
        inp_path.write_text(inp_content)

        mesh_info = mesh_from_abaqus_inp(str(inp_path))

        # ダミーソルバー結果
        n_nodes = 3
        ndof_per_node = 3
        ndof = n_nodes * ndof_per_node
        time_arr = np.linspace(0, 0.1, 11)
        disp = np.zeros((11, ndof))
        vel = np.zeros((11, ndof))
        acc = np.zeros((11, ndof))

        result = _DummySolverResult(time_arr, disp, vel, acc)

        step = Step(
            name="test",
            total_time=0.1,
            dt=0.01,
            field_output=FieldOutputRequest(num=2, variables=["U"]),
        )

        db = build_output_database(
            steps=[step],
            solver_results=[result],
            ndof_per_node=ndof_per_node,
            node_coords=mesh_info["node_coords"],
            connectivity=mesh_info["connectivity"],
            node_sets=mesh_info["node_sets"],
        )

        assert db.n_nodes == 3
        assert db.ndim == 2

        # VTK出力
        pvd_path = export_vtk(db, tmp_path / "vtk")
        assert Path(pvd_path).exists()
