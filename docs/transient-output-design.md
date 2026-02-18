# 過渡応答出力インターフェース 設計仕様書

[← README](../README.md) | [← roadmap](roadmap.md)

## 概要

動解析（Phase 5）の計算結果を体系的に記録・出力するインターフェースを設計する。
Abaqus / CalculiX の出力体系に準じた Step / Increment / Frame の階層構造を導入し、
CSV・JSON・VTK（ParaView対応）の3形式でエクスポートする。

---

## 1. 基本概念

### 1.1 Step（ステップ）

解析の時間と境界条件を与えて解く**1つの計算単位**。

- 時間範囲 `[0, total_time]` を持つ
- 境界条件・荷重条件を定義する
- ステップはシリアルに繋げることができ、**前ステップの終状態を次ステップの初期状態**とする
- 複数の Output Request を持てる

### 1.2 Increment（インクリメント）

Newton-Raphson 反復の**1収束分**で、時間増分の最小単位。

- 各インクリメント終了時に構造は平衡状態にある
- 時間刻み `dt` でステップの全時間を分割
- 非線形解析ではインクリメントごとに NR 反復を行い収束させる

### 1.3 Frame（フレーム）

**フィールド出力のために指定されるインクリメントのスナップショット**。

- `output,field,num=N` で指定された時刻に合わせて、インクリメント結果を記録
- 時間増分の調整 or 内挿補間によりフレーム時刻を実現
- ステップ全体を N 等分した時刻にフレームを生成

---

## 2. キーワード体系

### 2.1 Initial Conditions

ステップ前に初期条件を定義する。

```
initial_conditions(type="velocity", nset="all_nodes", dof=1, value=10.0)
```

**サポートする type:**

| type | 説明 |
|------|------|
| `velocity` | 初期速度 |
| `displacement` | 初期変位 |

### 2.2 Output, History

時系列プロファイルの出力要求。特定の節点集合に対して高頻度で記録する。

```python
HistoryOutputRequest(
    dt=0.01,                              # 出力時間間隔
    node_sets={"refmove": [0, 1, 2]},     # 節点集合
    variables=["RF", "U", "F", "ALLIE", "ALLKE"],  # 出力変数
)
```

**出力変数:**

| 変数 | 説明 | スコープ |
|------|------|---------|
| `U` | 変位 | 節点 |
| `V` | 速度 | 節点 |
| `A` | 加速度 | 節点 |
| `RF` | 反力（拘束節点） | 節点 |
| `CF` | 外部集中荷重 | 節点 |
| `ALLIE` | 全内部エネルギー | グローバル |
| `ALLKE` | 全運動エネルギー | グローバル |

### 2.3 Output, Field

空間分布データのスナップショット出力。ステップ全体を等分割して記録する。

```python
FieldOutputRequest(
    num=15,                        # ステップ内のフレーム数
    variables=["U", "V", "A"],     # 出力変数
    node_sets=None,                # None = 全節点
)
```

---

## 3. データモデル

### 3.1 パッケージ構成

```
xkep_cae/output/
├── __init__.py          # 公開API
├── step.py              # Step, Increment, Frame, StepResult
├── request.py           # HistoryOutputRequest, FieldOutputRequest
├── initial_conditions.py # InitialConditions
├── database.py          # OutputDatabase（統合記録・ストレージ）
├── export_csv.py        # CSV エクスポート
├── export_json.py       # JSON エクスポート
└── export_vtk.py        # VTK/VTU エクスポート（ParaView 対応）
```

### 3.2 主要クラス

```python
@dataclass
class Step:
    """解析ステップの定義."""
    name: str
    total_time: float
    dt: float
    history_output: HistoryOutputRequest | None = None
    field_output: FieldOutputRequest | None = None

@dataclass
class IncrementResult:
    """1インクリメントの結果."""
    increment_index: int
    time: float
    dt: float
    displacement: np.ndarray    # (ndof,)
    velocity: np.ndarray        # (ndof,)
    acceleration: np.ndarray    # (ndof,)
    converged: bool
    iterations: int

@dataclass
class Frame:
    """フィールド出力フレーム."""
    frame_index: int
    time: float
    displacement: np.ndarray
    velocity: np.ndarray | None
    acceleration: np.ndarray | None

@dataclass
class StepResult:
    """ステップ全体の結果."""
    step: Step
    step_index: int
    increments: list[IncrementResult]
    frames: list[Frame]
    history: dict[str, dict[str, np.ndarray]]  # {nset: {var: (n_times,) or (n_times, ndof)}}
    history_times: np.ndarray

@dataclass
class OutputDatabase:
    """全ステップの出力データベース."""
    step_results: list[StepResult]
    node_coords: np.ndarray           # (n_nodes, ndim) 節点座標
    connectivity: np.ndarray | None   # 要素接続（VTK出力用）
    ndof_per_node: int
    node_sets: dict[str, np.ndarray]  # {名前: 節点インデックス配列}
    fixed_dofs: np.ndarray | None     # 拘束DOF
```

### 3.3 OutputDatabase の構築フロー

```
1. OutputDatabase を初期化（節点座標、接続、DOF情報）
2. 各 Step に対して:
   a. ソルバー実行中にインクリメント結果をコールバックで受信
   b. HistoryOutputRequest に基づき dt 間隔で変数を記録
   c. FieldOutputRequest に基づき指定時刻のフレームを記録
   d. StepResult を生成して OutputDatabase に追加
3. エクスポート
```

---

## 4. エクスポート形式

### 4.1 CSV

- ヒストリ出力: `{step_name}_history_{nset}.csv`
  - 列: time, U1, U2, ..., RF1, RF2, ..., ALLIE, ALLKE
- フレーム一覧: `{step_name}_frames.csv`

### 4.2 JSON

- 全データを1つの JSON ファイルに構造化出力
- メタデータ（ステップ名、時間、変数名）+ データ配列

### 4.3 VTK/VTU (ParaView)

- `.vtu` (VTK XML Unstructured Grid): 各フレームに1ファイル
  - 節点座標 + 変位・速度等のポイントデータ
  - 要素接続（梁要素 = VTK_LINE, Q4 = VTK_QUAD 等）
- `.pvd` (ParaView Data): タイムステップを束ねるインデックスファイル
  - 全フレームの .vtu ファイルへの参照 + タイムスタンプ

**VTK要素型マッピング:**

| xkep-cae 要素 | VTK Cell Type | VTK ID |
|---------------|---------------|--------|
| 2節点梁（EB/Timo/Cosserat） | VTK_LINE | 3 |
| TRI3 | VTK_TRIANGLE | 5 |
| Q4 | VTK_QUAD | 9 |
| TRI6 | VTK_QUADRATIC_TRIANGLE | 22 |

---

## 5. エネルギー計算

### 5.1 ALLKE（全運動エネルギー）

```
ALLKE = 0.5 * v^T * M * v
```

### 5.2 ALLIE（全内部エネルギー）

```
ALLIE = 0.5 * u^T * K * u  （線形の場合）
```

非線形の場合はインクリメンタルに積算。

---

## 6. ステップの連結

```python
steps = [
    Step(name="step-1", total_time=1.0, dt=0.01, ...),
    Step(name="step-2", total_time=2.0, dt=0.005, ...),
]

# step-1 の終状態が step-2 の初期状態になる
db = run_steps(steps, M, K, C, ...)
```

---

## 7. 実装方針

1. 既存の `dynamics.py` ソルバーの結果からポストプロセスで OutputDatabase を構築する方式
2. 既存ソルバーのインターフェースは変更しない（後方互換性維持）
3. `build_output_database()` 関数で既存ソルバー結果を変換
4. 新規 `run_transient_steps()` 関数で Step 列を順次実行してデータベースを構築

---
