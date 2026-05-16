# status-324: K_st distance culling 実装 — Huber 遷移幅ベースの gap pre-filter

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-12
- **ブランチ**: `claude/check-status-todos-lFX9c`
- **テスト数**: 459+13+22+5+8（distance culling 8 テスト追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-323 TODO「distance culling 実装」を完了。Contact K_st と Friction K_st の両方に Huber 遷移幅ベースの gap pre-filter を実装。

- **ContactForceStStiffnessProcess**: `delta_h / k_pen + 1e-8` を gap 閾値として自動計算し、step 1 の has_state_pairs 抽出時に遠方ペアを除外
- **FrictionStStiffnessProcess**: `gap_cull_threshold` パラメータをパイプライン貫通（FrictionStStiffnessInput → _assemble_friction_st_stiffness → TangentAssemblyProcess）
- **HuberContactForceProcess.compute_gap_cull_threshold()**: 公開メソッド追加で摩擦側からも同一閾値を取得可能に
- 8 テスト追加（contact 5 + friction 3）、全既存テスト回帰なし

## 背景

status-319 の scaling 分析で **ContactForceStStiffness α≈2.07（n²）** と **FrictionStStiffness α≈2.04（n²）** が判明。status-323 の調査メモで距離カットの設計方針が策定されていた:

- **実装箇所**: `_process_batch` で `state.gap > threshold` をフィルタ
- **閾値**: Huber derivative = 0 となる gap = `delta_h / k_pen` を閾値に使用
- **リスク軽減**: 力評価は全ペアで維持（K_st のみに適用）

## 設計

### Gap culling threshold の自動計算

Huber 関数 `h(x, δ)` の導関数は `x < -δ` で厳密にゼロ。K_st の寄与は `h_deriv(k_pen * (-gap), delta_h)` に比例するため:

```
h_deriv = 0  when  k_pen * (-gap) < -delta_h
              ⟺   gap > delta_h / k_pen
```

この gap を超えるペアの K_st 寄与は厳密にゼロ。浮動小数点境界のマージンとして `+1e-8` を追加:

```python
_gap_cull = (delta_h / k_pen if delta_h > 0 else 0.0) + 1e-8
```

### Contact K_st（パイプライン変更不要）

`ContactForceStStiffnessProcess._process_batch()` は入力に `delta_h` と `k_pen` を既に持つため、内部で自動計算:

```python
# Distance culling threshold (status-324)
_gap_cull = float("inf")
if inp.k_pen > 0:
    _gap_cull = (inp.delta_h / inp.k_pen if inp.delta_h > 0 else 0.0) + 1e-8

# Step 1: state + gap < _gap_cull pre-filter
has_state_pairs = [
    p for p in inp.pairs
    if hasattr(p, "state") and p.state.gap < _gap_cull
]
```

### Friction K_st（パイプライン貫通）

摩擦側は `delta_h` / `k_pen` を持たないため、パイプラインで閾値を伝搬:

1. `FrictionStStiffnessInput.gap_cull_threshold: float = float("inf")` 追加
2. `_assemble_friction_st_stiffness(gap_cull_threshold=...)` パラメータ追加
3. `CoulombReturnMappingProcess.tangent()` の `**kwargs` 経由で伝搬
4. `TangentAssemblyProcess.process()` で `compute_gap_cull_threshold()` を呼び出し、`_fric_kw["gap_cull_threshold"]` に設定

## 実施内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/contact_force/strategy.py` | _process_batch に gap pre-filter 追加。`compute_gap_cull_threshold()` 公開メソッド追加 |
| `xkep_cae/contact/friction/strategy.py` | FrictionStStiffnessInput に `gap_cull_threshold` フィールド追加。kwargs 伝搬 |
| `xkep_cae/contact/friction/_assembly.py` | `_assemble_friction_st_stiffness` に `gap_cull_threshold` パラメータ追加、gap フィルタ |
| `xkep_cae/contact/solver/_newton_steps.py` | TangentAssemblyProcess → friction tangent に gap_cull_threshold を渡す |
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | TestDistanceCulling: 5 テスト追加 |
| `xkep_cae/contact/friction/tests/test_assembly_process.py` | TestFrictionStStiffnessDistanceCulling: 3 テスト追加 |

## テスト

### 新規テスト（8 件）

| テスト | 内容 |
|--------|------|
| `TestDistanceCulling::test_far_pair_culled` | gap > threshold → K_st = 0 |
| `TestDistanceCulling::test_near_pair_kept` | gap < threshold → K_st > 0 |
| `TestDistanceCulling::test_threshold_auto_computed` | delta_h=0 → gap > 1e-8 で culled |
| `TestDistanceCulling::test_mixed_pairs_only_near_contributes` | 近接+遠方混合 → 近接のみの結果と一致 |
| `TestDistanceCulling::test_huber_gap_cull_threshold_method` | compute_gap_cull_threshold() の値検証 |
| `TestFrictionStStiffnessDistanceCulling::test_far_pair_culled` | friction gap >= threshold → K_st = 0 |
| `TestFrictionStStiffnessDistanceCulling::test_near_pair_kept` | friction gap < threshold → K_st > 0 |
| `TestFrictionStStiffnessDistanceCulling::test_default_inf_no_culling` | デフォルト inf → 全ペア処理 |

### 回帰確認

```
contact_force/tests: 54 passed
friction/tests:      50 passed
solver/tests:        91 passed, 5 skipped
contact/ (残り):     189 passed
契約違反: 0 件
ruff check: OK
ruff format: OK
```

## 性能への影響

### 直接的効果

- **has_state_pairs リスト削減**: broadphase が生成する全候補ペアのうち、gap >= threshold のペアを step 1 で除外。後続の p_n 抽出（np.fromiter）と StJacobian バッチ計算の入力サイズを削減。
- **摩擦 K_st ループ短縮**: gap >= threshold のペアを dict lookup 前にスキップ。

### 制限事項

- gap culling は Python list comprehension レベルの pre-filter であり、StJacobian/K_st のバッチ計算自体のコストは削減しない（そちらは既に p_n > 1e-30 フィルタ済み）
- **n² scaling の根本は物理的接触ペア数の二次増大**。distance culling は broadphase 候補のうち非接触ペアの overhead 削減に効果的だが、物理的に接触中のペア数は削減しない
- 本格的な n² 抑制には空間ブロック分離やペアクラスタリングが必要（後続 status）

## TODO（次担当者向け）

### 直近

- [ ] **n=37 以上の掃引で culling 効果を定量計測** — status-319 と同一条件（gap=0.07固定、κ=0.005）で culled ペア数と per-call 時間を比較
- [ ] **symbolic factorization reuse 実装**（1 status 相当）: pypardiso `analyze()` キャッシュを LinearSolveProcess に統合
- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-320 TODO 継続
- [ ] **ファイバー梁 Phase F1 着手** — status-313 継続

### 中期

- [ ] **リスタート解析方式への移行**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理
- [ ] **ProcessMetaclass._profile_data と ProcessExecutionLog の統合** — status-322 TODO 継続
- [ ] **空間ブロック分離 or ペアクラスタリング**: 物理的接触ペア数の n² 成長を抑制する構造的対策

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果は pytest -v 出力で確認
- [x] **再現手順記載**: 上記テスト結果セクション
- [x] **テスト数記載**: 459+13+22+5+8
- [x] **契約違反 0 件維持**: validate_process_contracts.py 実行済み
- [x] **lint/format 検証**: ruff check + ruff format --check OK
