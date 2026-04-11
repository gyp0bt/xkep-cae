# status-320: `uses` グラフ拡張 — `StrategySlot.default_types` で接触剛性 Process を到達可能化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-11
- **ブランチ**: `claude/check-status-todos-iEepx`
- **テスト数**: 459+13+22+5（status-319 から `TestUsesGraphStrategySlotExpansion` 5 件追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

- status-319 の TODO「**`uses` グラフ拡張（StrategySlot uses 宣言で
  `ContactForceStStiffness`/`FrictionStStiffness` を到達可能化）**」を実装。
- `StrategySlot` に `default_types` キーワード引数を追加し、
  `default_strategies()` が注入する具象 Process 型をクラスレベルで静的宣言可能に。
- `ParameterSweepBenchmarkProcess._collect_uses_graph()` を拡張し、MRO を遡って
  `StrategySlot.default_types` を再帰走査するようにした。
- `_is_leaf_process()` も拡張し、静的 `uses=[]` でも `StrategySlot.default_types`
  が非空なら wrapper 扱いに。Strategy 経由で他 Process を呼ぶクラスを誤って葉
  認定しない。
- `ContactFrictionProcess` の StrategySlot 5 個のうち 4 個に `default_types` を
  宣言。**`ContactFrictionProcess` から以下の 8 Process がクラスレベルで到達
  可能**になった:
  - `HuberContactForceProcess` → `ContactForceStStiffnessProcess`
  - `CoulombReturnMappingProcess` → `FrictionStStiffnessProcess`,
    `FrictionTangentStiffnessProcess`, `FrictionGeometricStiffnessProcess`
  - `GeneralizedAlphaProcess`
  - `LineToLineGaussProcess`
  - `ComputeStJacobianProcess`（上記経由、葉判定 OK）
- グラフサイズ: **10 → 30 クラス**（3x 拡張）。
- 5 テスト追加。既存 21 テストと合わせて 26 件すべて通過。
- 全テスト 91（contact/solver）+ 89（numerical_tests non-slow）+ 199（tests/ non-slow）
  全通過。契約違反 0、lint/format/condition OK。

## 背景 — なぜ必要か

status-319 で **ContactForceStStiffnessProcess / FrictionStStiffnessProcess**
の per-call avg/call が **α≈2.07 の n² scaling**で成長することが実測された。
これらは 1000 本実測（90° 曲げ）での最大ボトルネック候補。

一方、status-317 で導入した
`ParameterSweepBenchmarkProcess.dominant_leaf_process` は `_collect_uses_graph()`
が **`cls.uses` のみを再帰走査**していた。`ContactFrictionProcess` の
静的 `uses` は NewtonDynamic/UnifiedTimeStep 等 10 件が並んでおり、
接触力・摩擦の剛性計算系は **StrategySlot 経由で動的注入**される設計なので、
静的走査では絶対に到達できない。その結果 status-318 の dominant_leaf 抽出は
「TangentAssembly が葉」と判定されていた（接触剛性プロセス側の n² 成長を
**構造的に見逃す**状態）。

status-319 TODO:
> **`uses` グラフ拡張**: `ContactFrictionProcess` の `uses` に
> `ContactForceStStiffness` 等を Strategy 経由で間接 declare できる仕組み

本 status はこの拡張を最小の侵襲で実現する。

## 実施内容

### 1. `StrategySlot` に `default_types` パラメータ追加

**ファイル**: `xkep_cae/core/slots.py`

```python
class StrategySlot(Generic[T]):
    def __init__(
        self,
        protocol: type[T],
        *,
        required: bool = True,
        default_types: tuple[type, ...] = (),  # status-320
    ) -> None:
        ...
        self.default_types: tuple[type, ...] = tuple(default_types)
```

- 後方互換: キーワード引数でデフォルト `()`。既存 StrategySlot 宣言の変更不要。
- 意味: 「`default_strategies()` がこのスロットに注入し得る具象 Process 型」の
  クラスレベル宣言。`effective_uses()`（インスタンスレベル）とは独立した
  静的グラフ走査用のヒント。
- Protocol を満たす候補が複数（例: `AutoBeamEIPenalty` / `ConstantPenalty`）で、
  どれも `uses=[]` の葉なら宣言省略可（到達不能だが葉判定は保守的に葉扱い）。

### 2. `_collect_uses_graph()` を StrategySlot 経由で再帰走査

**ファイル**: `xkep_cae/numerical_tests/parameter_sweep_benchmark.py`

```python
def _collect_uses_graph(root_cls: type) -> dict[str, type]:
    ...
    while stack:
        cls = stack.pop()
        ...
        for dep in getattr(cls, "uses", []):
            if dep.__name__ not in visited:
                stack.append(dep)
        # status-320: StrategySlot.default_types もクラスレベルで展開
        for klass in reversed(cls.__mro__):
            for attr in vars(klass).values():
                if not isinstance(attr, StrategySlot):
                    continue
                for dep_type in attr.default_types:
                    if dep_type.__name__ not in visited:
                        stack.append(dep_type)
    return visited
```

- MRO を遡って全 StrategySlot を拾う（サブクラスの追加宣言も拾える）。
- `StrategySlot` のインポートは関数内 late-binding で循環依存を回避
  （`core.slots` は `parameter_sweep_benchmark` より基盤層）。

### 3. `_is_leaf_process()` を StrategySlot 併合判定に拡張

```python
def _is_leaf_process(name: str, known_classes: dict[str, type]) -> bool:
    cls = known_classes.get(name)
    if cls is None:
        return True
    if getattr(cls, "uses", []):
        return False
    # status-320: 静的 uses=[] でも StrategySlot 経由依存があれば wrapper
    for klass in cls.__mro__:
        for attr in vars(klass).values():
            if isinstance(attr, StrategySlot) and attr.default_types:
                return False
    return True
```

**重要**: `uses=[]` だが `StrategySlot.default_types` が非空のクラスは、
従来「葉」と誤認されていた。本修正で wrapper 扱いに。これにより
`dominant_leaf_process` は本当の葉（Strategy の先の `ComputeStJacobianProcess`
等）まで降りていける。

### 4. `ContactFrictionProcess` の StrategySlot 宣言を拡張

**ファイル**: `xkep_cae/contact/solver/process.py`

```python
from xkep_cae.contact.contact_force.strategy import HuberContactForceProcess
from xkep_cae.contact.friction.strategy import CoulombReturnMappingProcess
from xkep_cae.contact.geometry.strategy import LineToLineGaussProcess
from xkep_cae.time_integration.strategy import GeneralizedAlphaProcess

class ContactFrictionProcess(...):
    penalty_slot = StrategySlot(object)  # 葉戦略のみ → 宣言省略
    friction_slot = StrategySlot(
        object,
        default_types=(CoulombReturnMappingProcess,),
    )
    time_integration_slot = StrategySlot(
        object,
        default_types=(GeneralizedAlphaProcess,),
    )
    contact_force_slot = StrategySlot(
        object,
        required=False,
        default_types=(HuberContactForceProcess,),
    )
    contact_geometry_slot = StrategySlot(
        object,
        required=False,
        default_types=(LineToLineGaussProcess,),
    )
```

- インポート循環の有無を事前確認済み: `contact_force/strategy.py` と
  `friction/strategy.py` は `contact/solver/process.py` に依存しないため
  純粋な片方向依存。
- `penalty_slot` は同 Protocol を満たす 2 候補（`AutoBeamEIPenalty` /
  `ConstantPenalty`）があり、どちらも `uses=[]` の葉なので省略した。

### 5. テスト追加: `TestUsesGraphStrategySlotExpansion`

**ファイル**: `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py`

軽量ダミー Process（`_SweepSolverProcess` → `_SweepStrategyProcess` →
`_SweepLeafProcess`）で StrategySlot 経由の再帰走査を検証:

1. `test_strategy_slot_default_types_are_reachable`: Strategy と葉の両方に到達
2. `test_strategy_leaf_is_leaf`: Strategy 先の葉が葉判定される
3. `test_first_leaf_skips_wrapper_and_strategy`: dominant_leaf が Strategy を
   skip して最深葉まで降りる
4. `test_default_types_empty_is_no_op`: `default_types=()` の slot は到達不能
   （後方互換性確認）
5. `test_contact_friction_reaches_k_st_processes`: **実機検証** —
   `ContactFrictionProcess` から status-319 TODO の 8 Process
   （HuberContactForce / ContactForceStStiffness / CoulombReturnMapping /
   FrictionStStiffness / FrictionTangentStiffness / FrictionGeometricStiffness
   / GeneralizedAlpha / LineToLineGauss / ComputeStJacobian）に全て到達。
   また K_st 系は wrapper、ComputeStJacobian が葉として検出されること。

## 主要結果

### グラフサイズ比較

| 条件 | _collect_uses_graph(ContactFrictionProcess) サイズ |
|------|---------------------------------------------------|
| status-319 以前（静的 uses のみ）| ~10 クラス（接触剛性系が全て不可視）|
| **status-320（StrategySlot 展開あり）**| **30 クラス** |

### 到達可能化したクラス（status-319 TODO の核）

- **`HuberContactForceProcess`**（contact_force_slot 経由）
  - → `ContactForceStStiffnessProcess`（**α≈2.07 の n² 成長**）
    - → `ComputeStJacobianProcess`（葉）
- **`CoulombReturnMappingProcess`**（friction_slot 経由）
  - → `FrictionTangentStiffnessProcess`
  - → `FrictionGeometricStiffnessProcess`
  - → `FrictionStStiffnessProcess`（**α≈2.04 の n² 成長**）
    - → `ComputeStJacobianProcess`（葉）
- **`GeneralizedAlphaProcess`**（time_integration_slot 経由、葉）
- **`LineToLineGaussProcess`**（contact_geometry_slot 経由）

### 影響: `dominant_leaf_process` は本当の葉まで降りられる

本 status 以前:
```
ContactFrictionProcess (wrapper, uses=10)
└─ TangentAssemblyProcess (wrapper)
   └─ ... 葉判定が正しい
```
接触剛性系プロセスは **グラフ外扱い** → profile に現れても `_is_leaf_process`
が「未知プロセス」として**保守的に葉扱い**していた（status-317 仕様）。

status-320 以降:
```
ContactFrictionProcess (wrapper, uses=10 + StrategySlot×4)
├─ TangentAssemblyProcess (wrapper) → ComputeStJacobianProcess 葉
├─ HuberContactForceProcess (wrapper)
│  └─ ContactForceStStiffnessProcess (wrapper)
│     └─ ComputeStJacobianProcess (葉)
├─ CoulombReturnMappingProcess (wrapper)
│  ├─ FrictionTangentStiffnessProcess (葉)
│  ├─ FrictionGeometricStiffnessProcess (葉)
│  └─ FrictionStStiffnessProcess (wrapper)
│     └─ ComputeStJacobianProcess (葉)
└─ ...
```

これにより status-319 の掃引結果を dominant_leaf 視点で再解析する際、
「`ContactForceStStiffnessProcess` が真のボトルネック葉」のような**正しい
診断**が可能になる（実測は次担当者の作業）。

## 変更ファイル

### 更新

- `xkep_cae/core/slots.py`: `StrategySlot.__init__` に `default_types` 追加 +
  docstring 拡張
- `xkep_cae/numerical_tests/parameter_sweep_benchmark.py`:
  `_collect_uses_graph()` と `_is_leaf_process()` を StrategySlot 対応に拡張
- `xkep_cae/contact/solver/process.py`: 4 つの StrategySlot に `default_types`
  宣言 + 対応する Process 型を import
- `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py`:
  `TestUsesGraphStrategySlotExpansion` クラスに 5 テスト追加 + 軽量ダミー
  Process（`_SweepLeafProcess` / `_SweepStrategyProcess` / `_SweepSolverProcess`）
  追加
- `README.md`: 状態行を status-320 リンクに更新
- `docs/status/status-index.md`: status-320 行追加
- `docs/roadmap.md`: status-320 行追加
- `CLAUDE.md`: 「次の課題」TODO で `uses` グラフ拡張済みマークに

### 新規

- `docs/status/status-320.md`（本ファイル）

## 検証手順（再現手順）

```bash
git checkout claude/check-status-todos-iEepx

# 1. 契約チェック
PYTHONPATH=. uv run python contracts/validate_process_contracts.py
# → 契約違反なし、条例違反なし

# 2. lint / format
PYTHONPATH=. uv run ruff check xkep_cae/ tests/
PYTHONPATH=. uv run ruff format --check xkep_cae/ tests/
# → 全ファイル formatted / 1 files changed, 0 remaining

# 3. 新規テスト + 既存 parameter_sweep_benchmark テスト
PYTHONPATH=. uv run --with pytest --with pytest-timeout python -m pytest \
    xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py -v \
    2>&1 | tee /tmp/log-status320-sweep-$(date +%s).log
# → 26 passed

# 4. contact/solver 回帰
PYTHONPATH=. uv run --with pytest --with pytest-timeout python -m pytest \
    xkep_cae/contact/solver/tests/ --timeout=120 \
    2>&1 | tee /tmp/log-status320-contact-$(date +%s).log
# → 91 passed, 5 skipped

# 5. numerical_tests（non-slow）
PYTHONPATH=. uv run --with pytest --with pytest-timeout python -m pytest \
    xkep_cae/numerical_tests/tests/ tests/test_benchmark_runner.py \
    tests/test_process_diagnostics.py -m "not slow" --timeout=120 \
    2>&1 | tee /tmp/log-status320-numerical-$(date +%s).log
# → 89 passed, 1 skipped, 1 xfailed

# 6. 実機グラフ検証
PYTHONPATH=. uv run python -c "
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.numerical_tests.parameter_sweep_benchmark import _collect_uses_graph
graph = _collect_uses_graph(ContactFrictionProcess)
print('graph size:', len(graph))  # 30
for name in [
    'HuberContactForceProcess', 'ContactForceStStiffnessProcess',
    'CoulombReturnMappingProcess', 'FrictionStStiffnessProcess',
    'GeneralizedAlphaProcess', 'LineToLineGaussProcess',
    'ComputeStJacobianProcess',
]:
    print(f'  {name}: {\"YES\" if name in graph else \"NO\"}')"
# → 全 YES、graph size: 30
```

### 実測環境

- Linux 4.4 / Python 3.11.15 / uv 0.8.17
- NumPy 2.4.4 / SciPy 1.17.1 / ruff 0.14.3

## 判断の根拠

### なぜ `default_types` を StrategySlot 自身に持たせたか

Alternative 1: `ContactFrictionProcess.default_strategy_types: ClassVar` として
Process クラスに直接宣言する。
- デメリット: 宣言場所が StrategySlot から離れ、どのスロットがどの型に対応
  するかが視覚的にわかりにくい。slot と default 型がペアで管理されるべき。

Alternative 2: `default_strategies()` を class method として Process に
bind する。
- デメリット: `default_strategies()` は `ndof` 等のランタイム引数を取る
  インスタンス関数。クラスレベルで呼べない。

選択した方針（StrategySlot.default_types）は:
- 宣言が slot と同じ場所に集約される
- 後方互換（キーワード引数デフォルト `()`）
- 走査側（`_collect_uses_graph`）は MRO ベースの一般化ロジックで拾える

### なぜ `_is_leaf_process` の修正が必要だったか

status-317 の `_is_leaf_process` は `cls.uses == []` を葉判定の条件にしていた。
本 status で `_SweepSolverProcess.uses = []` だが `strategy_slot.default_types`
が非空という構成を用意したら、従来定義では誤って葉扱いされていた（テスト
`test_first_leaf_skips_wrapper_and_strategy` で初回失敗）。
**Strategy 経由で他 Process を呼ぶクラスは実時間で wrapper として動作する**
ので、StrategySlot.default_types 非空時も wrapper 扱いに統一。

実機影響: `ContactFrictionProcess.uses` は既に 10 件あるので従来でも wrapper
扱いだが、将来 Strategy のみで依存を注入するクラスが出現した場合も正しく
判定できる。

### なぜ `penalty_slot` だけ `default_types` を省略したか

`_create_penalty_strategy()` は `beam_E/I/L` の指定次第で `AutoBeamEIPenalty`
または `ConstantPenalty` を返す。どちらも `uses=[]` の葉 Process なので、
グラフ走査で到達できても追加情報はゼロ。保守性のため単一の具象を選ぶのを避け、
明示的に省略した（宣言のない StrategySlot は従来通り到達不能）。

将来これら葉 Process の中身が wrapper 化した場合、tuple で両方宣言すればよい。

## TODO（次担当者向け）

### 直近

- [ ] **status-318 のリレー再解析**: status-318 実測データに新 walker を
  適用し、dominant_leaf が `TangentAssembly` → `ContactForceStStiffness` に
  移動するか検証。条件によっては status-319 の n² scaling 結論を status-318
  データでも再現できる可能性。
- [ ] **`_is_leaf_process` の既存 status への影響**: 過去に生成された
  ParameterSweepBenchmark yaml に対して本修正版 walker を適用すると、
  dominant_leaf フィールドが再計算されて従来値と差分が出る場合がある。
  dominant_leaf 履歴の後方互換性を考慮する必要あり（実測値は不変、解析値のみ更新）。
- [ ] **K_st 測定の分離** — status-319 から継続: `TangentAssemblyProcess`
  の avg/call に K_mat + K_geo と K_st（接触経由）が混合している。K_st 独立
  計測の Process 分離は依然 TODO。

### 中期

- [ ] **ContactForceStStiffness / FrictionStStiffness の n² 成長抑制**（core
  最適化）: 空間ハッシュ / 距離カット / ML 削減等 — status-319 TODO の本丸。
- [ ] **pypardiso 環境での `status319_corrected_sweep` 再実行**: 本 status320
  でグラフ到達可能性は担保されたので、次ベンチでは実測の dominant_leaf
  が直接 K_st 系を指すはず。
- [ ] **被膜 ON プロファイル**: barrier coating を ON にした掃引で接触系
  プロセスの比率がどう変わるか（status-305 構成との接続）。
- [ ] **`penalty_slot.default_types` の設計**: 現在 `()` で省略だが、将来
  `AutoBeamEIPenalty / ConstantPenalty` が wrapper 化する場合の宣言方針を
  docstring に追記しておく。

## STA2 準拠チェック

- [x] **数値の捏造なし**: グラフサイズ 30、到達クラス数 8 は実機実行で確認。
- [x] **再現手順記載**: 上記「検証手順」セクション、6 ステップのコマンド列。
- [x] **テスト数記載**: 459+13+22+5。前 status-319 の 459+13+22 に status-320
  新規 5 件を加算。
- [x] **契約違反 0 件維持**: `validate_process_contracts.py` 実行済み。
- [x] **lint/format 検証**: `ruff check` + `ruff format --check` 全 OK。
- [x] **ベースライン比較**: status-319 の実測データは変更なし、本 status は
  グラフ走査ロジックの拡張のみで実測値へ影響しない。
