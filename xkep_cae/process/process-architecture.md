# xkep-cae プロセスアーキテクチャ設計仕様書

[← README](../../README.md) | [← process](README.md) | [← roadmap](../../docs/roadmap.md)

## Context

xkep-caeは撚線FEMシミュレーターで、以下の構造的問題を抱えている:

1. **新旧ソルバー混在**: `newton_raphson_contact_ncp` (推奨) vs `newton_raphson_with_contact` (旧AL法) が共存し、テストがどちらに依存しているか不明
2. **オプション爆発**: `solver_ncp.py`の関数に72個のキーワード引数。内部は `_dynamics`/`_smooth`/`_use_friction` 等のフラグで条件分岐まみれ
3. **依存関係不明**: solver↔assembler↔contact間の暗黙的依存が多く、1箇所の変更で予測不能な回帰が発生
4. **検証コード散在**: `scripts/verify_*.py`(14個)、`tests/*_physics.py`(5個)、`tests/*_validation*.py`(7個)、`numerical_tests/`(7モジュール)が整理されていない
5. **deprecated管理不在**: 旧ソルバーのテストが`deprecated`マーカー付きで残存するが体系的管理なし

**基軸構成**: NCP + Uzawa + smooth_penalty（7本撚線曲げ収束実績あり）

**目的**: AbstractProcessクラスによる契約化で、依存関係を機械的に検証可能にし、10セッション以内でリファクタリング完了する。

**設計原則**:
- テストコードと設計仕様書は実装コードのすぐそばに配置
- テストクラスとプロセスは完全に1:1対応
- プロセス横断的テストはBatchProcess等を都度作成
- ソルバー内部はStrategy合成で分解し、if分岐/オプション引数の増殖を防止

---

## 1. AbstractProcess基底クラス

### 1.1 クラス定義

```python
# xkep_cae/process/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, ClassVar, Any
import functools
import time

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class ProcessMeta:
    """プロセスのメタ情報（自動収集）"""
    name: str
    module: str  # "pre", "solve", "assemble", "post", "verify" 等
    version: str = "0.1.0"
    deprecated: bool = False
    deprecated_by: str | None = None  # 後継プロセスのクラス名


class ProcessMeta_(type(ABC)):
    """AbstractProcess のメタクラス.

    process() メソッドを自動ラップし、以下を実現する:
    - 実行トレース: どのprocess()が実際に呼ばれたかを記録
    - オーバーヘッド計測: process()単位の実行時間を自動プロファイリング
    - uses検証: process()内で使用されたプロセスの実行時チェック（デバッグモード）
    """

    _call_stack: ClassVar[list[str]] = []  # 実行中のprocess名スタック
    _profile_data: ClassVar[dict[str, list[float]]] = {}  # {name: [elapsed_list]}

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # process() が定義されていればラップ
        if 'process' in namespace and callable(namespace['process']):
            original = namespace['process']

            @functools.wraps(original)
            def traced_process(self, input_data):
                cls_name = type(self).__name__
                ProcessMeta_._call_stack.append(cls_name)
                t0 = time.perf_counter()
                try:
                    result = original(self, input_data)
                finally:
                    elapsed = time.perf_counter() - t0
                    ProcessMeta_._call_stack.pop()
                    # プロファイルデータ蓄積
                    if cls_name not in ProcessMeta_._profile_data:
                        ProcessMeta_._profile_data[cls_name] = []
                    ProcessMeta_._profile_data[cls_name].append(elapsed)
                return result

            cls.process = traced_process

        return cls

    @classmethod
    def get_trace(mcs) -> list[str]:
        """現在の実行スタック（デバッグ用）."""
        return list(mcs._call_stack)

    @classmethod
    def get_profile_report(mcs) -> str:
        """全プロセスのプロファイルレポート."""
        lines = ["Process Profile Report", "=" * 40]
        for name, times in sorted(mcs._profile_data.items()):
            n = len(times)
            total = sum(times)
            avg = total / n if n > 0 else 0
            lines.append(f"  {name}: {n} calls, total={total:.3f}s, avg={avg:.3f}s")
        return "\n".join(lines)


class AbstractProcess(ABC, Generic[TIn, TOut], metaclass=ProcessMeta_):
    """全プロセスの基底クラス.

    契約:
    - uses に宣言したプロセスのみを process() 内で使用可能
    - Input/Output型はジェネリックパラメータで明示
    - __init_subclass__ でクラス定義時に制約違反を検出
    - メタクラスが process() を自動ラップし、実行トレース + プロファイリング
    """

    # --- クラス変数（サブクラスで上書き） ---
    meta: ClassVar[ProcessMeta]
    uses: ClassVar[list[type[AbstractProcess]]] = []

    # --- 自動管理（__init_subclass__で構築） ---
    _registry: ClassVar[dict[str, type[AbstractProcess]]] = {}
    _used_by: ClassVar[list[type[AbstractProcess]]] = []
    _test_class: ClassVar[str | None] = None  # 1:1 紐付けテストクラス
    _verify_scripts: ClassVar[list[str]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # 抽象中間クラス（SolverProcess等）はスキップ
        if getattr(cls, '__abstractmethods__', None):
            return

        # meta 必須チェック
        if not hasattr(cls, 'meta') or not isinstance(cls.meta, ProcessMeta):
            raise TypeError(
                f"{cls.__name__} は ProcessMeta を定義してください"
            )

        # deprecated チェック
        for dep in cls.uses:
            if hasattr(dep, 'meta') and dep.meta.deprecated:
                import warnings
                warnings.warn(
                    f"{cls.__name__} は deprecated な {dep.__name__} を使用。"
                    f" 後継: {dep.meta.deprecated_by}",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # レジストリ登録
        cls._registry[cls.__name__] = cls
        cls._used_by = []

        # uses → used_by 双方向リンク
        for dep in cls.uses:
            if not hasattr(dep, '_used_by'):
                dep._used_by = []
            dep._used_by.append(cls)

    @abstractmethod
    def process(self, input_data: TIn) -> TOut:
        """メイン処理. サブクラスで実装.

        メタクラスが自動ラップするため、呼び出し時に:
        - 実行トレース（call stack）に記録
        - 実行時間を自動計測（profile_data に蓄積）
        """
        ...

    def execute(self, input_data: TIn) -> TOut:
        """process() の公開エントリポイント.

        process() はメタクラスが既にラップ済みのため、
        execute() は追加のバリデーションや前後処理が必要な場合のみ使う。
        """
        return self.process(input_data)

    @classmethod
    def get_dependency_tree(cls) -> dict:
        """再帰的に依存ツリーを返す."""
        return {
            "name": cls.__name__,
            "module": cls.meta.module if hasattr(cls, 'meta') else "?",
            "uses": [dep.get_dependency_tree() for dep in cls.uses],
        }

    @classmethod
    def document_markdown(cls) -> str:
        """Markdownドキュメント自動生成."""
        lines = [
            f"## {cls.__name__}",
            f"- **モジュール**: {cls.meta.module}",
            f"- **バージョン**: {cls.meta.version}",
        ]
        if cls.meta.deprecated:
            lines.append(f"- **DEPRECATED** → {cls.meta.deprecated_by}")
        if cls.uses:
            lines.append(f"- **依存**: {', '.join(d.__name__ for d in cls.uses)}")
        if cls._used_by:
            lines.append(f"- **被依存**: {', '.join(d.__name__ for d in cls._used_by)}")
        if cls._test_class:
            lines.append(f"- **テスト**: `{cls._test_class}`")
        if cls._verify_scripts:
            lines.append("- **検証スクリプト**:")
            for vs in cls._verify_scripts:
                lines.append(f"  - `{vs}`")
        return "\n".join(lines)
```

### 1.2 オーバーヘッド分析と粒度設計

| レベル | 所要時間 | プロセス化 | 理由 |
|--------|----------|------------|------|
| NR反復内（要素力計算） | μs〜ms | **しない** | ホットパス。オーバーヘッドが支配的になる |
| NR反復1回 | ms〜100ms | **しない** | 同上 |
| 荷重増分1ステップ | 100ms〜数s | **境界** | ステップ間の接触更新はプロセス化可能 |
| 全ステップ（ソルバー実行） | 秒〜時間 | **する** | メインのプロセス単位 |
| メッシュ生成 | ms〜秒 | **する** | 独立した前処理 |
| 結果出力・可視化 | ms〜秒 | **する** | 独立した後処理 |

**原則**: `execute()` 呼び出し1回 = 「ソルバー1回実行」「メッシュ1回生成」等の粗粒度。NR内部ループはプロセス化しない。

---

## 2. ソルバー内部のStrategy分解

### 2.1 問題: 現在の `newton_raphson_contact_ncp` の構造

```
newton_raphson_contact_ncp(72 keyword args)
├── if _dynamics:        # 動的解析パス（Generalized-α）
│   └── 50行の初期化 + ステップ内で予測子・補正子
├── if _smooth:          # smooth penalty パス
│   └── Uzawa外部ループ + δ自動推定
├── if _use_friction:    # 摩擦パス
│   └── return mapping + 摩擦接線剛性
├── if _line_contact:    # line-to-line Gauss積分パス
│   └── 接触力評価を積分に切替
├── if use_mortar:       # mortar法パス
│   └── mortar結合系
└── if _k_pen_cont:      # k_penコンティニュエーション
    └── 段階的ペナルティ増加
```

これらは**直交する振る舞い軸**であり、if分岐ではなくStrategy合成で表現すべき。

### 2.2 Strategy Protocol定義

```python
# xkep_cae/contact/strategies.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@runtime_checkable
class ContactForceStrategy(Protocol):
    """接触力の評価方法を規定する.

    実装:
    - NCPContactForce: Alart-Curnier NCP + 鞍点系
    - SmoothPenaltyContactForce: softplus + Uzawa外部ループ
    """
    def evaluate(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力と接触残差を評価.

        Returns:
            (contact_force, ncp_residual)
        """
        ...

    def tangent(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> sp.csr_matrix:
        """接触接線剛性行列."""
        ...


@runtime_checkable
class FrictionStrategy(Protocol):
    """摩擦力の評価方法を規定する.

    実装:
    - NoFriction: 摩擦なし（デフォルト）
    - CoulombReturnMapping: Coulomb摩擦 return mapping
    - SmoothPenaltyFriction: smooth penalty + Uzawa摩擦
    """
    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と摩擦残差を評価.

        Returns:
            (friction_force, friction_residual)
        """
        ...

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列."""
        ...


@runtime_checkable
class TimeIntegrationStrategy(Protocol):
    """時間積分方法を規定する.

    実装:
    - QuasiStatic: 準静的（荷重制御）
    - GeneralizedAlpha: Generalized-α 動的解析
    """
    def predict(
        self,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """予測子."""
        ...

    def correct(
        self,
        u: np.ndarray,
        du: np.ndarray,
        dt: float,
    ) -> None:
        """補正子."""
        ...

    def effective_stiffness(
        self,
        K: sp.csr_matrix,
        dt: float,
    ) -> sp.csr_matrix:
        """有効剛性行列 K_eff = K + α/(β*dt²)*M + γ/(β*dt)*C."""
        ...

    def effective_residual(
        self,
        R: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """有効残差."""
        ...


@runtime_checkable
class ContactGeometryStrategy(Protocol):
    """接触幾何の評価方法を規定する.

    実装:
    - PointToPoint: 最近接点ペア（現行デフォルト）
    - LineToLineGauss: line-to-line Gauss積分
    - MortarSegment: Mortar法セグメント
    """
    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出."""
        ...

    def compute_gap(
        self,
        pair: object,
        node_coords: np.ndarray,
    ) -> float:
        """ギャップの計算."""
        ...


@runtime_checkable
class PenaltyStrategy(Protocol):
    """ペナルティ剛性の決定方法を規定する.

    実装:
    - AutoBeamEI: beam_E * beam_I / L³ ベース（デフォルト）
    - AutoEAL: E * A / L ベース
    - ManualPenalty: 手動指定
    - ContinuationPenalty: 段階的増加
    """
    def compute_k_pen(
        self,
        step: int,
        total_steps: int,
    ) -> float:
        """現在ステップのペナルティ剛性."""
        ...
```

### 2.3 StrategyをAbstractProcessとして管理する

**設計判断**: 技術検証段階で流動的なソルバー内部コンポーネントは、オーバーヘッドが許容範囲内であればAbstractProcessとして管理する。これにより:

- メタクラスの自動プロファイリングでオーバーヘッドを常時計測可能
- 依存関係がprocessレベルで可視化される
- Strategy追加/変更時に依存元への影響が機械的に検出される

```python
# Strategy もプロセスとして管理（NR内部ではなく、NR前の設定段階で呼ばれる）
class ContactForceProcess(SolverProcess[ContactForceInput, ContactForceOutput]):
    """接触力評価プロセス（NR反復内で呼ばれるが、Process単位の計測対象）.

    粒度の例外: NR反復内だが、技術検証段階で頻繁に差し替わるため
    Processとして管理する。オーバーヘッドはメタクラスのプロファイリングで
    常時モニタリングし、許容範囲を超えたら Protocol に降格する。

    降格基準: NR反復1回あたりのオーバーヘッド > 0.1ms
    """
    pass


class SmoothPenaltyContactForceProcess(ContactForceProcess):
    meta = ProcessMeta(name="Smooth Penalty 接触力", module="solve.contact_force")
    uses = []  # 独立

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        ...


class NCPContactForceProcess(ContactForceProcess):
    meta = ProcessMeta(name="NCP Alart-Curnier 接触力", module="solve.contact_force")
    uses = []

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        ...
```

### 2.4 Strategy合成とプリセット

```python
@dataclass(frozen=True)
class SolverStrategies:
    """ソルバー内部の振る舞いを合成するStrategy群.

    各strategyはAbstractProcessのサブクラス。
    メタクラスが自動的に実行トレース・プロファイリングを行う。
    """
    contact_force: ContactForceProcess
    friction: FrictionProcess
    time_integration: TimeIntegrationProcess
    contact_geometry: ContactGeometryProcess
    penalty: PenaltyProcess


# プリセット（基軸構成）
def default_strategies() -> SolverStrategies:
    """NCP + Uzawa + smooth_penalty の基軸構成."""
    return SolverStrategies(
        contact_force=SmoothPenaltyContactForceProcess(
            smoothing_delta=0.0,
            n_uzawa_max=5,
            tol_uzawa=1e-6,
        ),
        friction=SmoothPenaltyFrictionProcess(mu=0.15),
        time_integration=QuasiStaticProcess(),
        contact_geometry=LineToLineGaussProcess(n_gauss=4),
        penalty=AutoBeamEIProcess(scale=1.0),
    )


class NCPContactSolverProcess(SolverProcess[SolverInputData, SolverResultData]):
    meta = ProcessMeta(name="NCP接触ソルバー", module="solve")
    # uses は strategies の構成要素を動的に宣言
    # （__init__ 時に strategies から uses を構築）

    def __init__(self, strategies: SolverStrategies | None = None):
        self.strategies = strategies or default_strategies()
        # 動的 uses 宣言（メタクラスの依存追跡対象）
        type(self).uses = [
            type(self.strategies.contact_force),
            type(self.strategies.friction),
            type(self.strategies.time_integration),
            type(self.strategies.contact_geometry),
            type(self.strategies.penalty),
        ]

    def process(self, input_data: SolverInputData) -> SolverResultData:
        # strategies の各プロセスを呼び出し（メタクラスが自動トレース）
        ...
```

### 2.5 オーバーヘッド降格メカニズム

```python
# メタクラスのプロファイルデータから自動判定
def check_strategy_overhead():
    """NR内部Strategy のオーバーヘッドを検証.

    降格基準: NR反復1回あたりのProcess呼び出しオーバーヘッド > 0.1ms
    超過時はProtocol（型チェックのみ、トレースなし）への降格を推奨。
    """
    report = ProcessMeta_.get_profile_report()
    for name, times in ProcessMeta_._profile_data.items():
        if "Process" in name and len(times) > 100:
            avg = sum(times) / len(times)
            if avg < 1e-4:  # 0.1ms未満: Process維持OK
                pass
            else:
                print(f"WARNING: {name} avg={avg*1000:.2f}ms — Protocol降格を検討")
```

### 2.4 Strategy分解による効果

**Before** (現在):
```python
# newton_raphson_contact_ncp 内部
if _smooth:
    if _use_friction:
        # smooth penalty + friction
        for uzawa_iter in range(n_uzawa_max):
            ...
    else:
        # smooth penalty, no friction
        ...
elif use_mortar:
    ...
else:
    # NCP + saddle point
    ...
```

**After** (Strategy合成):
```python
# newton_raphson_contact_ncp 内部（統一ループ）
for step in adaptive_steps:
    u_pred = strategies.time_integration.predict(u, dt)
    K_eff = strategies.time_integration.effective_stiffness(K_T, dt)

    for nr_iter in range(max_iter):
        f_c, C_n = strategies.contact_force.evaluate(u, lambdas, manager, k_pen)
        f_t, C_t = strategies.friction.evaluate(u, active_pairs, mu)
        K_c = strategies.contact_force.tangent(u, lambdas, manager, k_pen)
        K_t = strategies.friction.tangent(u, active_pairs, mu)
        # ... 統一NRソルブ ...
```

**新しいオプション追加時**: 新Strategy実装を1ファイル追加するだけ。既存コードに触れない。

---

## 3. プロセス分類体系（モジュール区分）

```python
# xkep_cae/process/categories.py

class PreProcess(AbstractProcess[TIn, TOut], ABC):
    """前処理: メッシュ生成、境界条件設定、初期貫入回避等"""
    pass

class SolverProcess(AbstractProcess[TIn, TOut], ABC):
    """求解: NR反復、時間積分、弧長法"""
    pass

class PostProcess(AbstractProcess[TIn, TOut], ABC):
    """後処理: 結果抽出、出力、可視化"""
    pass

class VerifyProcess(AbstractProcess[TIn, TOut], ABC):
    """検証: 物理量チェック、解析解比較、収束検証"""
    pass

class BatchProcess(AbstractProcess[TIn, TOut], ABC):
    """バッチ: 複数プロセスの直列/並列実行"""
    pass
```

### 3.1 具体プロセスマッピング（現在のコード → 新体系）

| 現在のコード | 新プロセスクラス | カテゴリ |
|-------------|-----------------|---------|
| `TwistedWireMesh` | `StrandMeshProcess` | PreProcess |
| `ContactManager` 初期化 + broadphase | `ContactSetupProcess` | PreProcess |
| 初期貫入回避（`z_sep`調整） | `PenetrationAvoidProcess` | PreProcess |
| `newton_raphson_contact_ncp` | `NCPContactSolverProcess` | SolverProcess |
| `newton_raphson` (非接触) | `NewtonRaphsonProcess` | SolverProcess |
| `arc_length` | `ArcLengthProcess` | SolverProcess |
| `newton_raphson_with_contact` | `LegacyALSolverProcess` | SolverProcess (deprecated) |
| VTK/CSV/JSON出力 | `ExportProcess` | PostProcess |
| 3D梁レンダリング | `BeamRenderProcess` | PostProcess |
| 物理テスト（応力、エネルギー等） | `PhysicsVerifyProcess` | VerifyProcess |
| 解析解比較 | `AnalyticalVerifyProcess` | VerifyProcess |
| 収束検証（scripts/verify_*） | `ConvergenceVerifyProcess` | VerifyProcess |
| 撚線曲げ揺動フルパイプライン | `StrandBendingBatchProcess` | BatchProcess |

---

## 4. Input/Outputデータ契約

### 4.1 設計方針

- **dataclass(frozen=True)** で不変性を保証（NamedTupleからの移行は段階的）
- あるProcessのOutputが別ProcessのInputそのものになってよい
- 名称は `{プロセス名}Data` で統一（Input/Outputの区別はジェネリックで表現）
- **ラッパー方式**: 既存の`NCPSolverInput`を内部で使い続ける。`SolverInputData → NCPSolverInput`の変換メソッドを提供

### 4.2 主要データ型

```python
# xkep_cae/process/data.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Callable
import numpy as np
import scipy.sparse as sp
from xkep_cae.contact.pair import ContactManager


@dataclass(frozen=True)
class MeshData:
    """メッシュ生成結果 = ソルバー入力の一部"""
    node_coords: np.ndarray          # (n_nodes, 3)
    connectivity: np.ndarray          # (n_elems, 2)
    radii: np.ndarray | float
    n_strands: int
    layer_ids: np.ndarray | None = None  # 同層除外用


@dataclass(frozen=True)
class BoundaryData:
    """境界条件"""
    fixed_dofs: np.ndarray
    prescribed_dofs: np.ndarray | None = None
    prescribed_values: np.ndarray | None = None
    f_ext_total: np.ndarray | None = None
    f_ext_base: np.ndarray | None = None


@dataclass(frozen=True)
class ContactSetupData:
    """接触設定結果"""
    manager: ContactManager
    k_pen: float
    use_friction: bool
    mu: float | None = None
    contact_mode: str = "smooth_penalty"  # 基軸構成


@dataclass(frozen=True)
class AssembleCallbacks:
    """アセンブリコールバック（ソルバーへの入力）"""
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix]
    assemble_internal_force: Callable[[np.ndarray], np.ndarray]
    ul_assembler: object | None = None


@dataclass(frozen=True)
class SolverInputData:
    """NCPContactSolverProcess への入力（統合）.

    内部で NCPSolverInput に変換して既存ソルバーを呼び出す（ラッパー方式）。
    """
    mesh: MeshData
    boundary: BoundaryData
    contact: ContactSetupData
    callbacks: AssembleCallbacks
    strategies: SolverStrategies | None = None  # None = default_strategies()

    def to_ncp_solver_input(self) -> NCPSolverInput:
        """既存 NCPSolverInput への変換（ラッパー方式の核心）."""
        from xkep_cae.contact.solver_ncp import NCPSolverInput
        return NCPSolverInput(
            f_ext_total=self.boundary.f_ext_total,
            fixed_dofs=self.boundary.fixed_dofs,
            assemble_tangent=self.callbacks.assemble_tangent,
            assemble_internal_force=self.callbacks.assemble_internal_force,
            manager=self.contact.manager,
            node_coords_ref=self.mesh.node_coords,
            connectivity=self.mesh.connectivity,
            radii=self.mesh.radii,
            k_pen=self.contact.k_pen,
            use_friction=self.contact.use_friction,
            mu=self.contact.mu,
            ul_assembler=self.callbacks.ul_assembler,
            # ... strategies から設定を引き出す ...
        )


@dataclass
class SolverResultData:
    """ソルバー結果（NCPSolveResultのラッパー）"""
    u: np.ndarray
    converged: bool
    n_increments: int
    total_newton_iterations: int
    displacement_history: list[np.ndarray]
    contact_force_history: list[float]
    elapsed_seconds: float = 0.0
    diagnostics: object | None = None

    @classmethod
    def from_ncp_result(cls, result, elapsed: float = 0.0) -> SolverResultData:
        """NCPSolveResult からの変換."""
        return cls(
            u=result.u,
            converged=result.converged,
            n_increments=result.n_increments,
            total_newton_iterations=result.total_newton_iterations,
            displacement_history=result.displacement_history,
            contact_force_history=result.contact_force_history,
            elapsed_seconds=elapsed,
            diagnostics=result.diagnostics,
        )


@dataclass(frozen=True)
class VerifyInput:
    """検証プロセスへの入力"""
    solver_result: SolverResultData
    mesh: MeshData
    expected: dict[str, float]  # {"max_displacement": 1.23, ...}
    tolerance: float = 0.05  # 5% 許容


@dataclass
class VerifyResult:
    """検証結果"""
    passed: bool
    checks: dict[str, tuple[float, float, bool]]  # {name: (actual, expected, ok)}
    report_markdown: str = ""
    snapshot_paths: list[str] = field(default_factory=list)
```

---

## 5. テスト配置とプロセス1:1対応

### 5.1 テストコロケーション原則

**テストコードと設計仕様書は実装コードのすぐそばに配置する。**

```
xkep_cae/process/
├── base.py
├── test_base.py                    # base.py の 1:1 テスト
├── base.spec.md                    # base.py の設計仕様
├── categories.py
├── test_categories.py
├── strategies.py
├── test_strategies.py
├── strategies.spec.md              # Strategy Protocol の設計仕様
├── concrete/
│   ├── solve_ncp.py
│   ├── test_solve_ncp.py           # NCPContactSolverProcess の 1:1 テスト
│   ├── solve_ncp.spec.md
│   ├── pre_mesh.py
│   ├── test_pre_mesh.py
│   ├── pre_contact.py
│   ├── test_pre_contact.py
│   └── ...
├── verify/
│   ├── convergence.py
│   ├── test_convergence.py
│   └── ...
└── batch/
    ├── strand_bending.py
    ├── test_strand_bending.py       # BatchProcess = プロセス横断テスト
    └── ...
```

### 5.2 1:1テスト対応ルール

```python
# xkep_cae/process/testing.py

def binds_to(process_class: type[AbstractProcess]):
    """テストクラスをプロセスに1:1紐付けするデコレータ.

    1プロセスにつき1テストクラス。複数プロセスにまたがるテストは
    BatchProcess を作成してそこに紐付ける。

    Usage:
        @binds_to(NCPContactSolverProcess)
        class TestNCPContactSolverProcess:
            ...
    """
    def decorator(test_cls):
        if process_class._test_class is not None:
            raise ValueError(
                f"{process_class.__name__} には既に "
                f"{process_class._test_class} が紐付けられています。"
                "1:1 対応を維持してください。"
            )
        path = f"{test_cls.__module__}::{test_cls.__qualname__}"
        process_class._test_class = path
        return test_cls
    return decorator
```

### 5.3 docs/testsからのリンク

```python
# docs からは md リンクで参照
# xkep_cae/process/process-architecture.md 内:
#   [NCPContactSolverProcess 仕様](../../xkep_cae/process/concrete/solve_ncp.spec.md)

# tests/ からは import で参照（conftest.py で自動検出）
# tests/conftest.py:
def pytest_collect_file(parent, file_path):
    """xkep_cae/process/**/test_*.py を自動収集."""
    if file_path.name.startswith("test_") and "process" in str(file_path):
        return pytest.Module.from_parent(parent, path=file_path)
```

---

## 6. ProcessTree（実行グラフ）

```python
# xkep_cae/process/tree.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class NodeType(Enum):
    SEQUENTIAL = auto()
    PARALLEL = auto()
    CONDITIONAL = auto()


@dataclass
class ProcessNode:
    """実行グラフのノード"""
    process_class: type
    children: list[ProcessNode] = field(default_factory=list)
    node_type: NodeType = NodeType.SEQUENTIAL
    condition: Any | None = None

    def to_mermaid(self, indent: int = 0) -> str:
        """Mermaid フローチャート形式で出力"""
        ...


@dataclass
class ProcessTree:
    """プロセス実行グラフ全体"""
    root: ProcessNode
    name: str = ""

    def validate(self) -> list[str]:
        """依存関係の整合性チェック. 違反のリストを返す."""
        errors = []
        visited: set[str] = set()
        self._validate_node(self.root, visited, errors)
        return errors

    def _validate_node(
        self, node: ProcessNode, visited: set[str], errors: list[str]
    ) -> None:
        cls = node.process_class
        name = cls.__name__

        if name in visited:
            errors.append(f"循環依存検出: {name}")
            return
        visited.add(name)

        child_classes = {c.process_class for c in node.children}
        for dep in cls.uses:
            if dep not in child_classes and dep not in self._all_ancestors(node):
                errors.append(
                    f"{name} は {dep.__name__} を uses 宣言しているが、"
                    "ツリーに含まれていない"
                )

        for child in node.children:
            self._validate_node(child, visited.copy(), errors)

    def to_markdown(self) -> str:
        """実行フロー図をMarkdownで出力"""
        ...

    def to_mermaid(self) -> str:
        """Mermaid フローチャート形式で出力"""
        ...
```

### 6.1 撚線曲げ揺動バッチの実行ツリー例

```
StrandBendingBatchProcess (BatchProcess)
├── [SEQUENTIAL]
│   ├── StrandMeshProcess (PreProcess)
│   │   Input: StrandMeshConfig → Output: MeshData
│   ├── ContactSetupProcess (PreProcess)
│   │   Input: MeshData → Output: ContactSetupData
│   ├── PenetrationAvoidProcess (PreProcess)
│   │   Input: MeshData + ContactSetupData → Output: MeshData (adjusted)
│   ├── NCPContactSolverProcess (SolverProcess)  ★基軸構成
│   │   Input: SolverInputData → Output: SolverResultData
│   │   Strategies: SmoothPenalty + CoulombFriction + QuasiStatic + LineToLine + AutoBeamEI
│   ├── BeamRenderProcess (PostProcess)
│   │   Input: SolverResultData + MeshData → Output: RenderData
│   └── ConvergenceVerifyProcess (VerifyProcess)
│       Input: VerifyInput → Output: VerifyResult
```

---

## 7. 依存関係バリデーション（ハイブリッド検証）

### 7.1 __init_subclass__ による静的チェック（クラス定義時）

- `meta` 未定義 → TypeError
- deprecated プロセスへの依存 → DeprecationWarning
- 自動 used_by 逆リンク構築
- 1:1テスト重複登録 → ValueError

### 7.2 バリデーションスクリプト（CI実行）

```python
# scripts/validate_process_deps.py
"""
プロセスの依存関係を構文解析で検証する（ハイブリッド検証のCI側）。

__init_subclass__ が捕捉できない問題:
- process() 内で uses 未宣言のプロセスを直接インスタンス化
- テスト紐付け漏れ（_test_class が None のプロセス）
- deprecated プロセスの active 被依存

これらをAST解析で検出する。
"""
import ast
import inspect
from xkep_cae.process.base import AbstractProcess


def validate_all() -> list[str]:
    errors = []
    for name, cls in AbstractProcess._registry.items():
        # 1. process() 内の未宣言依存チェック
        source = inspect.getsource(cls.process)
        tree = ast.parse(source)
        used_names = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in AbstractProcess._registry
        }
        declared = {dep.__name__ for dep in cls.uses}
        undeclared = used_names - declared - {name}
        for u in undeclared:
            errors.append(f"{name}.process() が uses 未宣言の {u} を参照")

        # 2. テスト紐付け漏れチェック（1:1対応）
        if not cls.meta.deprecated and cls._test_class is None:
            errors.append(f"{name} にテストが紐付けられていない (@binds_to 未使用)")

        # 3. deprecated プロセスの被依存チェック
        if cls.meta.deprecated and cls._used_by:
            for user in cls._used_by:
                if not user.meta.deprecated:
                    errors.append(
                        f"active な {user.__name__} が deprecated な {name} に依存"
                    )
    return errors
```

---

## 8. Deprecated管理

### 8.1 deprecatedプロセスの宣言

```python
class LegacyALSolverProcess(SolverProcess[SolverInputData, SolverResultData]):
    meta = ProcessMeta(
        name="AL法接触ソルバー（Outer/Inner分離）",
        module="solve",
        deprecated=True,
        deprecated_by="NCPContactSolverProcess",
    )
    uses = []
```

### 8.2 既存テストの分類

| テストファイル | 依存ソルバー | 状態 |
|-------------|------------|------|
| `test_solver_ncp.py` | NCP | active |
| `test_solver_ncp_s3.py` | NCP | active |
| `test_ncp_convergence_19strand.py` | NCP | active |
| `test_friction_validation_ncp.py` | NCP | active |
| `test_friction_validation.py` | 旧AL | deprecated |
| `test_beam_contact_penetration.py` | 旧AL | deprecated |
| `test_mortar_twisted_wire.py` | NCP+Mortar | active |

---

## 9. ファイル構成

```
xkep_cae/process/
├── __init__.py              # 公開API
├── base.py                  # AbstractProcess, ProcessMeta
├── test_base.py             # ★ 1:1 テスト
├── base.spec.md             # ★ 設計仕様
├── categories.py            # PreProcess, SolverProcess, ...
├── test_categories.py       # ★ 1:1 テスト
├── data.py                  # MeshData, SolverInputData, ...
├── tree.py                  # ProcessTree, ProcessNode
├── test_tree.py             # ★ 1:1 テスト
├── concrete/
│   ├── __init__.py
│   ├── pre_mesh.py          # StrandMeshProcess
│   ├── test_pre_mesh.py     # ★ 1:1
│   ├── pre_contact.py       # ContactSetupProcess, PenetrationAvoidProcess
│   ├── test_pre_contact.py  # ★ 1:1
│   ├── solve_ncp.py         # NCPContactSolverProcess (基軸)
│   ├── test_solve_ncp.py    # ★ 1:1
│   ├── solve_ncp.spec.md    # ★ 設計仕様
│   ├── solve_legacy.py      # LegacyALSolverProcess (deprecated)
│   ├── solve_static.py      # NewtonRaphsonProcess, ArcLengthProcess
│   ├── test_solve_static.py # ★ 1:1
│   ├── post_export.py       # ExportProcess
│   ├── test_post_export.py  # ★ 1:1
│   ├── post_render.py       # BeamRenderProcess
│   └── test_post_render.py  # ★ 1:1
├── strategies/
│   ├── __init__.py
│   ├── protocols.py         # ★ ContactForceStrategy, FrictionStrategy, ...
│   ├── test_protocols.py    # ★ Protocol準拠テスト
│   ├── protocols.spec.md    # ★ Strategy仕様
│   ├── contact_ncp.py       # NCPContactForce
│   ├── test_contact_ncp.py  # ★ 1:1
│   ├── contact_smooth.py    # SmoothPenaltyContactForce
│   ├── test_contact_smooth.py
│   ├── friction_none.py     # NoFriction
│   ├── friction_coulomb.py  # CoulombReturnMapping
│   ├── test_friction_coulomb.py
│   ├── friction_smooth.py   # SmoothPenaltyFriction
│   ├── time_quasi_static.py # QuasiStatic
│   ├── time_gen_alpha.py    # GeneralizedAlpha
│   ├── test_time_gen_alpha.py
│   ├── geometry_ptp.py      # PointToPoint
│   ├── geometry_l2l.py      # LineToLineGauss
│   ├── test_geometry_l2l.py
│   ├── penalty_auto.py      # AutoBeamEI, AutoEAL
│   └── penalty_continuation.py  # ContinuationPenalty
├── verify/
│   ├── __init__.py
│   ├── physics.py           # PhysicsVerifyProcess
│   ├── test_physics.py      # ★ 1:1
│   ├── analytical.py        # AnalyticalVerifyProcess
│   ├── test_analytical.py
│   ├── convergence.py       # ConvergenceVerifyProcess
│   └── test_convergence.py
└── batch/
    ├── __init__.py
    ├── strand_bending.py    # StrandBendingBatchProcess
    └── test_strand_bending.py  # ★ プロセス横断テスト
```

---

## 10. リファクタリングロードマップ（10セッション計画）

### Phase 1: 基盤 + Strategy Protocol（セッション1-2）
1. `xkep_cae/process/base.py` + `test_base.py` — AbstractProcess, ProcessMeta
2. `xkep_cae/process/categories.py` + `test_categories.py` — 5カテゴリクラス
3. `xkep_cae/process/strategies/protocols.py` + `test_protocols.py` — 5つのStrategy Protocol
4. `xkep_cae/process/data.py` — データ型定義
5. `xkep_cae/process/tree.py` + `test_tree.py` — ProcessTree
6. `xkep_cae/process/testing.py` — binds_to（1:1制約つき）

### Phase 2: Strategy実装（セッション3-4）
7. `strategies/contact_ncp.py` — NCPContactForce（solver_ncp.py のNCP部分を抽出）
8. `strategies/contact_smooth.py` — SmoothPenaltyContactForce（smooth_penalty部分を抽出）
9. `strategies/friction_*.py` — 摩擦3バリアント
10. `strategies/time_*.py` — 準静的 + Generalized-α
11. `strategies/geometry_*.py` — PtP + L2L + Mortar
12. `strategies/penalty_*.py` — 自動推定 + コンティニュエーション
13. 各strategyの1:1テスト

### Phase 3: 具体プロセス実装（セッション5-6）
14. `concrete/solve_ncp.py` — NCPContactSolverProcess（strategies注入、NCPSolverInputラッパー）
15. `concrete/pre_mesh.py` — StrandMeshProcess
16. `concrete/pre_contact.py` — ContactSetupProcess
17. `concrete/post_*.py` — 後処理プロセス群
18. `concrete/solve_legacy.py` — LegacyALSolverProcess（deprecated宣言）

### Phase 4: 検証プロセス移行（セッション7-8）
19. `verify/convergence.py` — scripts/verify_*.pyの契約化
20. `verify/physics.py` — 物理テスト紐付け
21. `verify/analytical.py` — 解析解比較紐付け
22. 既存テストに`@binds_to`デコレータ追加

### Phase 5: バッチプロセス・統合・クリーンアップ（セッション9-10）
23. `batch/strand_bending.py` — 撚線曲げ揺動フルパイプライン（プロセス横断テスト）
24. `scripts/validate_process_deps.py` — ハイブリッドバリデーション
25. 旧ソルバーテストの整理（deprecated移動 or 削除）
26. 重複verify_スクリプトの統合
27. docs/verification/ の階層整理
28. README, roadmap, status 最終更新

---

## 11. 検証方法

1. **コロケーションテスト**: `pytest xkep_cae/process/` — 実装そばのテスト実行
2. **依存関係バリデーション**: `python scripts/validate_process_deps.py` — CI統合
3. **既存テスト不退行**: `pytest tests/ -x` — 既存2271テストが壊れないこと
4. **lint**: `ruff check xkep_cae/process/` && `ruff format --check`
5. **1:1対応チェック**: バリデーションスクリプトで `_test_class is None` を検出

---

## 12. 今回のセッションでの成果物

1. **本設計仕様書** → `xkep_cae/process/process-architecture.md` としてコミット
2. **status-150** → 設計仕様策定完了の記録
3. **roadmap更新** → リファクタリングフェーズ追加
4. **README更新** → 設計仕様書へのリンク追加

---

## 13. 契約抜け腐敗シナリオ分析

### 13.1 検出可能（__init_subclass__ + バリデーションスクリプトで対応済み）

| # | シナリオ | 検出メカニズム | 対策 |
|---|---------|--------------|------|
| C1 | `meta` 未定義のプロセス定義 | `__init_subclass__` → TypeError | クラス定義時に即座にエラー |
| C2 | deprecated プロセスへの依存 | `__init_subclass__` → DeprecationWarning | 警告表示 + CI で -W error 化 |
| C3 | テスト未紐付けのプロセス | バリデーションスクリプト → `_test_class is None` | CI で fail |
| C4 | 1:1 テスト対応の重複登録 | `binds_to` → ValueError | デコレータ実行時にエラー |
| C5 | process() 内の未宣言依存 | AST解析 → uses にないProcess参照を検出 | CI で fail |

### 13.2 検出困難（設計的対策が必要）

| # | シナリオ | 問題 | 対策案 | 状態 |
|---|---------|------|--------|------|
| C6 | **Strategy Protocol の暗黙違反** | Protocol は構造的部分型。メソッドシグネチャは合うが意味論が違う（例: `evaluate()` が接触力ではなくゼロを返す） | Protocol テストに「契約テスト」を追加: 各実装が満たすべき不変条件を `test_protocols.py` で検証 | **Phase 1 で対応** |
| C7 | **メタクラスのラップ漏れ** | `process()` をクラス定義時ではなく後から monkey-patch した場合、メタクラスのラップが適用されない | `execute()` 内で `process.__wrapped__` の存在チェック。ない場合は警告 | **Phase 1 で対応** |
| C8 | **動的 uses の整合性崩壊** | `NCPContactSolverProcess.__init__` で strategies から uses を構築するが、`__init__` が呼ばれる前に `uses` を参照するコードがある場合に不整合 | クラス変数 `uses` は空のまま。インスタンス変数 `_runtime_uses` で管理。バリデーションスクリプトは `_runtime_uses` も検査 | **Phase 2 で対応** |
| C9 | **frozen dataclass の numpy 配列変更** | `MeshData(frozen=True)` でも `mesh.node_coords[0] = 999` は防げない（numpy は in-place 変更可能） | `execute()` 入口で入力のハッシュ（checksumチェック）を計算し、出口で変更されていないか検証（デバッグモード限定） | **Phase 3 で対応（オプション）** |
| C10 | **Strategy Process のオーバーヘッド蓄積** | NR反復が数百回 × Strategy 5個 = 数千回の Process 呼び出し。メタクラスのプロファイリング自体がボトルネックになる可能性 | `check_strategy_overhead()` を Phase 2 終了時に実行。0.1ms/call 超過なら Protocol に降格 | **Phase 2 で計測→判断** |
| C11 | **uses チェーンの推移的依存漏れ** | A uses B, B uses C だが、A の process() が直接 C を使う場合。AST解析は直接参照しか検出しない | AST解析を拡張: `self.b_instance.c_instance.process()` のようなチェーンも検出するか、uses に推移的依存も含める規約にする | **未確定: Phase 4 で再評価** |
| C12 | **バッチプロセスの順序依存** | `StrandBendingBatchProcess` 内のプロセス実行順序が暗黙的。ProcessTree で定義していても、実装が順序を無視する可能性 | ProcessTree.validate() に実行順序チェック追加: output型 → 次のinput型の型互換チェック | **Phase 4 で対応** |

### 13.3 対応不能（アーキテクチャの限界として受容）

| # | シナリオ | 問題 | 受容理由 |
|---|---------|------|---------|
| C13 | **ランタイム型安全性** | Python のジェネリクスは実行時に消去される。`AbstractProcess[MeshData, SolverResultData]` の型パラメータは実行時に検証できない | Python の言語仕様。mypy/pyright による静的解析で補完。実行時型チェックは `isinstance` ベースで最小限に |
| C14 | **マルチプロセス環境でのレジストリ不整合** | `_registry` はクラス変数（グローバル状態）。fork 後に子プロセスがプロセスを登録しても親に伝搬しない | 現時点ではシングルプロセス前提。GPU並列化（S7）時に `multiprocessing.Manager` への移行を検討 |
| C15 | **Jupyter/REPL 環境でのクラス再定義** | 同名クラスを再定義すると `_registry` の既存エントリが上書きされ、`_used_by` リンクが切れる | 開発環境の制約として文書化。`_registry` に重複警告を追加するが、完全な防止は不可能 |
| C16 | **Strategy 間の暗黙的相互作用** | 例: `SmoothPenaltyContactForce` と `CoulombReturnMapping` の組み合わせは摩擦接線剛性の符号問題で発散する（status-147）。Strategy の直交性が保証されない | 「互換性マトリクス」を `SolverStrategies` に定義し、`__post_init__` で非互換組み合わせを検出。ただし全組み合わせの検証は実質不可能なので、検証済み組み合わせのホワイトリスト方式を採用 |

### 13.4 Strategy互換性マトリクス

```python
# xkep_cae/process/strategies/compatibility.py

# 検証済み組み合わせ（ホワイトリスト）
VERIFIED_COMBINATIONS = [
    # 基軸構成（7本撚線曲げ収束実績）
    {
        "contact_force": "SmoothPenaltyContactForceProcess",
        "friction": "SmoothPenaltyFrictionProcess",
        "time_integration": "QuasiStaticProcess",
        "contact_geometry": "LineToLineGaussProcess",
        "penalty": "AutoBeamEIProcess",
    },
    # NCP法線 + 摩擦なし（基本テスト用）
    {
        "contact_force": "NCPContactForceProcess",
        "friction": "NoFrictionProcess",
        "time_integration": "QuasiStaticProcess",
        "contact_geometry": "PointToPointProcess",
        "penalty": "AutoBeamEIProcess",
    },
    # 動的解析（Generalized-α）
    {
        "contact_force": "SmoothPenaltyContactForceProcess",
        "friction": "SmoothPenaltyFrictionProcess",
        "time_integration": "GeneralizedAlphaProcess",
        "contact_geometry": "LineToLineGaussProcess",
        "penalty": "AutoBeamEIProcess",
    },
]

# 既知の非互換組み合わせ（ブラックリスト）
INCOMPATIBLE_COMBINATIONS = [
    # status-147: NCP鞍点系 + Coulomb摩擦 → 摩擦接線剛性の符号問題で発散
    {
        "contact_force": "NCPContactForceProcess",
        "friction": "CoulombReturnMappingProcess",
        "reason": "摩擦接線剛性の符号問題で鞍点系が不定値化（status-147）",
    },
]


def validate_strategy_combination(strategies: SolverStrategies) -> list[str]:
    """Strategy組み合わせの互換性チェック.

    Returns:
        警告メッセージのリスト。空ならOK。
    """
    warnings = []
    combo = {
        "contact_force": type(strategies.contact_force).__name__,
        "friction": type(strategies.friction).__name__,
        "time_integration": type(strategies.time_integration).__name__,
        "contact_geometry": type(strategies.contact_geometry).__name__,
        "penalty": type(strategies.penalty).__name__,
    }

    # ブラックリストチェック
    for incompat in INCOMPATIBLE_COMBINATIONS:
        match = all(
            combo.get(k) == v
            for k, v in incompat.items()
            if k != "reason"
        )
        if match:
            warnings.append(f"非互換: {incompat['reason']}")

    # ホワイトリストチェック
    is_verified = any(
        all(combo.get(k) == v for k, v in verified.items())
        for verified in VERIFIED_COMBINATIONS
    )
    if not is_verified:
        warnings.append(
            f"未検証の組み合わせ: {combo}。"
            "VERIFIED_COMBINATIONS に追加する前に収束テストを実行してください。"
        )

    return warnings
```

### 13.5 未確定事項（次セッション以降で決定）

| # | 事項 | 判断時期 | 判断基準 |
|---|-----|---------|---------|
| U1 | Strategy を Process として維持するか Protocol に降格するか | Phase 2 完了時 | オーバーヘッド計測結果 (0.1ms/call 基準) |
| U2 | 推移的依存チェックの実装方式 | Phase 4 | 実際の依存チェーン深度（2段以上あるか） |
| U3 | numpy 配列 immutability の検証強度 | Phase 3 | デバッグ時のバグ発生頻度 |
| U4 | ProcessTree の実行順序型チェック | Phase 4 | BatchProcess の複雑度 |
| U5 | マルチプロセス対応のレジストリ設計 | S7（GPU並列化） | 並列化アーキテクチャ決定後 |
