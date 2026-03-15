# Process Architecture Phase 8 設計書

[← README](../../../README.md) | [← process-architecture](process-architecture.md) | [← roadmap](../../../docs/roadmap.md)

**日付**: 2026-03-13
**前提**: Phase 7 完了（status-162）— 契約違反0件、23プロセス登録済み

## 概要

Phase 8 では、Phase 7 までに構築した Process + Strategy 基盤の上に、
**実行管理**（ProcessRunner）と**型安全性**（Strategy slot 型宣言）を追加する。

Phase 7 完了時点の主な課題:
1. プロセス実行の呼び出しが `process.process(input)` 直呼びで、依存チェックなし
2. `_runtime_uses` がインスタンス変数で、静的バリデーション不可
3. SolverStrategies が frozen dataclass だが、Strategy 組み合わせの検証が `__post_init__` にない
4. concrete プロセスの実行順序管理が BatchProcess 任せで、パイプライン記述が冗長

---

## A: ProcessRunner / ExecutionContext

### 目的

プロセス実行時に依存チェック・プロファイリング・ログ出力を一元管理する。
現在はメタクラスがトレースを行っているが、**実行コンテキスト**（ログ先、プロファイル有無、
ドライラン）を切り替える手段がない。

### 設計

```python
# xkep_cae/process/runner.py

@dataclass
class ExecutionContext:
    """プロセス実行のコンテキスト."""
    dry_run: bool = False                    # True なら process() を呼ばない
    profile: bool = True                     # プロファイリング有無
    log_file: Path | None = None             # ログ出力先
    validate_deps: bool = __debug__          # 依存チェック有無
    checksum_inputs: bool = __debug__        # 入力 checksum 検証（C9）


class ProcessRunner:
    """プロセスの実行管理.

    責務:
    1. 実行前: uses 依存チェック（validate_deps 時）
    2. 実行前: 入力 checksum 計算（checksum_inputs 時）
    3. 実行: process() 呼び出し（dry_run 時はスキップ）
    4. 実行後: checksum 検証 + プロファイル記録 + ログ出力
    """

    def __init__(self, context: ExecutionContext | None = None) -> None:
        self.context = context or ExecutionContext()
        self._execution_log: list[dict] = []

    def run(self, process: AbstractProcess[TIn, TOut], input_data: TIn) -> TOut:
        """プロセスを実行する.

        execute() ではなく run() を経由することで:
        - 依存チェックが統一的に行われる
        - プロファイルデータが ExecutionContext に集約される
        - dry_run でパイプライン検証ができる
        """
        ...

    def run_pipeline(
        self,
        steps: list[tuple[AbstractProcess, Any]],
    ) -> list[Any]:
        """複数プロセスを順次実行."""
        ...
```

### メタクラスとの役割分担

| 責務 | メタクラス (ProcessMeta_) | ProcessRunner |
|------|--------------------------|---------------|
| process() ラップ | ✅ 自動（クラス定義時） | — |
| コールスタック追跡 | ✅ _call_stack | — |
| プロファイルデータ蓄積 | ✅ _profile_data | 読み取り + レポート |
| 依存チェック | — | ✅ run() 時に _runtime_uses 検証 |
| 入力 checksum | base.py execute() | ✅ run() に移動 |
| ログ出力 | — | ✅ log_file に出力 |
| dry_run | — | ✅ process() スキップ |

### 影響範囲

- `base.py`: execute() の checksum ロジックを runner.py に移動
- `batch/`: StrandBendingBatchProcess が ProcessRunner を内部使用
- テスト: test_runner.py 新規作成

---

## B: Strategy Slot 型宣言

### 目的

現在の `_runtime_uses` はインスタンス変数で、静的解析ツール（mypy/pyright）で型チェックできない。
**クラス変数としての Strategy slot 宣言**により、型安全性を向上させる。

### 現状の問題

```python
# 現在: NCPContactSolverProcess.__init__()
self._runtime_uses = [
    type(self.strategies.penalty),
    type(self.strategies.friction),
    ...
]
# → mypy は _runtime_uses の内容を型チェックできない
# → validate_process_contracts.py で AST 検査が必要
```

### 設計

```python
# xkep_cae/process/slots.py

class StrategySlot(Generic[T]):
    """Strategy slot の型付きディスクリプタ.

    クラス変数として宣言し、インスタンスで具象 Strategy を設定する。
    """

    def __init__(self, protocol: type[T], required: bool = True) -> None:
        self.protocol = protocol
        self.required = required
        self._attr_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = f"_slot_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> T:
        if obj is None:
            return self  # type: ignore
        return getattr(obj, self._attr_name)

    def __set__(self, obj: Any, value: T) -> None:
        if not isinstance(value, self.protocol):
            raise TypeError(
                f"{self._attr_name}: {type(value).__name__} は "
                f"{self.protocol.__name__} を満たしていない"
            )
        setattr(obj, self._attr_name, value)


# 使用例
class NCPContactSolverProcess(SolverProcess[SolverInputData, SolverResultData]):
    # Strategy slot 宣言（クラス変数 — 静的解析可能）
    contact_force = StrategySlot(ContactForceStrategy, required=False)
    friction = StrategySlot(FrictionStrategy)
    time_integration = StrategySlot(TimeIntegrationStrategy)
    penalty = StrategySlot(PenaltyStrategy)
    contact_geometry = StrategySlot(ContactGeometryStrategy, required=False)

    def __init__(self, strategies: SolverStrategies | None = None) -> None:
        s = strategies or default_strategies()
        # StrategySlot.__set__ が Protocol 準拠を検証
        self.penalty = s.penalty
        self.friction = s.friction
        self.time_integration = s.time_integration
        if s.contact_force is not None:
            self.contact_force = s.contact_force
        if s.contact_geometry is not None:
            self.contact_geometry = s.contact_geometry
```

### _runtime_uses からの移行

1. StrategySlot ディスクリプタが `__set_name__` で自動登録
2. `effective_uses()` は StrategySlot から動的に構築（ディスクリプタ走査）
3. `_runtime_uses` は deprecated 化し、StrategySlot 優先で依存解決
4. validate_process_contracts.py の C8 チェックを StrategySlot 対応に更新

---

## C: CompatibilityProcess カテゴリ

### 目的

deprecated プロセス（ManualPenaltyProcess 等）を明示的なカテゴリで隔離し、
レガシーコードとの境界を型レベルで表現する。

### 設計

```python
# xkep_cae/process/categories.py に追加

class CompatibilityProcess(AbstractProcess[TIn, TOut], ABC):
    """後方互換プロセス.

    新規コードからの uses 宣言を禁止。
    既存テスト維持のみに使用する。
    """
    pass
```

### 影響範囲

- ManualPenaltyProcess を `SolverProcess` → `CompatibilityProcess` に変更
- validate_process_contracts.py に C13 チェック追加:
  「active プロセスが CompatibilityProcess を uses している場合はエラー」

---

## D: Preset First-class 化

### 目的

現在の `default_strategies()` 関数を Preset クラスに昇格し、
検証済み組み合わせ（VERIFIED_COMBINATIONS）との整合を型レベルで保証する。

### 設計

```python
# xkep_cae/process/presets.py

@dataclass(frozen=True)
class SolverPreset:
    """検証済みの Strategy 組み合わせ.

    name: プリセット名（一意）
    strategies: 構成済み SolverStrategies
    verified_by: 検証 status 番号
    description: 用途説明
    """
    name: str
    strategies: SolverStrategies
    verified_by: str  # e.g. "status-147"
    description: str = ""


# 組み込みプリセット
PRESET_SMOOTH_PENALTY = SolverPreset(
    name="smooth_penalty_quasi_static",
    strategies=SolverStrategies(
        contact_force=SmoothPenaltyContactForceProcess(),
        friction=SmoothPenaltyFrictionProcess(mu=0.15),
        time_integration=QuasiStaticProcess(),
        contact_geometry=LineToLineGaussProcess(n_gauss=4),
        penalty=AutoBeamEIProcess(scale=1.0),
    ),
    verified_by="status-147",
    description="7本撚線曲げ揺動収束実績あり。基軸構成。",
)

PRESET_NCP_FRICTIONLESS = SolverPreset(
    name="ncp_frictionless",
    strategies=SolverStrategies(
        contact_force=NCPContactForceProcess(ndof=0),  # ndof はプロセス実行時に設定
        friction=NoFrictionProcess(),
        time_integration=QuasiStaticProcess(),
        contact_geometry=PointToPointProcess(),
        penalty=AutoBeamEIProcess(scale=1.0),
    ),
    verified_by="status-112",
    description="NCP法線 + 摩擦なし。基本テスト用。",
)

# プリセットレジストリ
PRESETS: dict[str, SolverPreset] = {
    p.name: p
    for p in [PRESET_SMOOTH_PENALTY, PRESET_NCP_FRICTIONLESS]
}
```

### default_strategies() との関係

```python
# data.py の default_strategies() は PRESET_SMOOTH_PENALTY.strategies を返すだけに
def default_strategies() -> SolverStrategies:
    from xkep_cae.process.presets import PRESET_SMOOTH_PENALTY
    return PRESET_SMOOTH_PENALTY.strategies
```

---

## 実装順序（推奨）

| ステップ | 内容 | 依存 | 推定テスト数 |
|---------|------|------|------------|
| 8-A | ProcessRunner + ExecutionContext | なし | +15 |
| 8-B | StrategySlot ディスクリプタ | なし | +10 |
| 8-C | CompatibilityProcess カテゴリ | 8-B | +5 |
| 8-D | SolverPreset + レジストリ | 8-B | +10 |
| 8-E | NCPContactSolverProcess を 8-A/B/D で更新 | 8-A, 8-B, 8-D | +5 |
| 8-F | validate_process_contracts.py 更新 | 8-B, 8-C | 0 (既存更新) |

**8-A と 8-B は独立 → 並行実施可能**

---

## リスク・留意点

1. **StrategySlot は Python ディスクリプタ** — メタクラスとの相互作用に注意。
   ProcessMeta_ の `__new__` と StrategySlot の `__set_name__` の実行順序を検証する。
2. **SolverPreset の ndof 問題** — NCPContactForceProcess は ndof が必須引数だが、
   Preset 定義時には未知。`ndof=0` としてプレースホルダー化するか、
   Preset を「ファクトリ」として `create(ndof=...)` メソッドを持たせるか要検討。
3. **後方互換** — `_runtime_uses` を使う既存コード（validate_process_contracts.py の C8）
   が StrategySlot 移行後も動くことを保証する移行期間が必要。

---

## 次セッションへの引き継ぎ

- Phase 8 は設計段階。実装は次セッション以降。
- 8-A (ProcessRunner) と 8-B (StrategySlot) を先行実装し、
  既存テスト（275 process テスト + 2477 既存テスト）が壊れないことを確認後、
  8-C/D/E/F を進める。
- フォーカスガード: S3スケーリング作業・ソルバー性能改善は凍結継続。
