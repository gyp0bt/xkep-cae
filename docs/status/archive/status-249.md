# status-249: 同素線除外修正 + 7本撚線メッシュ作成 + フォーカスガード更新

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-27
**ブランチ**: `claude/twisted-wire-mesh-creation-tINC2`
**テスト数**: 200+10s（変更なし）
**契約違反**: 1件（C3 既知、ComputeStJacobianProcess テスト未紐付け）

---

## 実施事項

### 1. 同素線除外バグ修正（exclude_same_layer → exclude_same_strand）

#### 問題
旧 `exclude_same_layer=True` は**同一レイヤーの全接触ペアを除外**していた。
7本撚線（1+6）では外周6本が全てlayer=1のため、**外周素線同士の接触が全て除外**されていた。

| ケース | 実際に接触するか | 旧ロジック | 修正後 |
|--------|----------------|-----------|--------|
| 同層・同素線 | しない | 除外 ✅ | 除外 ✅ |
| 同層・隣接素線 | **する** | 除外 ❌ | 含める ✅ |
| 異層・隣接素線 | する | 含める ✅ | 含める ✅ |

#### 修正内容
- `MeshData.layer_ids` → `MeshData.strand_ids`（要素→素線IDマップ）
- `exclude_same_layer` → `exclude_same_strand`（全箇所リネーム）
- `elem_layer_map` → `elem_strand_map`（全箇所リネーム）
- `StrandMeshProcess`: `info.layer` → `info.strand_id` を使用

#### 影響ファイル（10ファイル）
- `xkep_cae/core/data.py` — MeshData フィールド名
- `xkep_cae/contact/_contact_pair.py` — _ContactConfigInput フィールド名
- `xkep_cae/contact/_manager_process.py` — フィルタロジック
- `xkep_cae/contact/setup/process.py` — ContactSetupConfig
- `xkep_cae/contact/geometry/strategy.py` — 3つのGeometryProcess + ファクトリ
- `xkep_cae/mesh/process.py` — StrandMeshProcess
- `xkep_cae/numerical_tests/three_point_bend_jig.py` — 2箇所
- `xkep_cae/contact/geometry/tests/test_strategy.py`
- `xkep_cae/contact/solver/tests/test_process.py`
- `tests/contact/test_strand_contact_process.py`

### 2. 7本撚線メッシュ作成（貫入ゼロ確認）

仕様:
- E=130MPa, ρ=8.96e-9 t/mm³, 線径10mm（半径5mm）, ピッチ100mm
- 32要素/ピッチ, 3ピッチ, **gap=0（自動: 3.33mm）**
- 節点679, 要素672, **接触ペア3356**, **初期貫入ゼロ**

`_compute_min_safe_gap` 改良: ヘリックス斜め交差効果を弦間最小距離の数値計算で反映。
N本撚線の自動ギャップ: 3本0.35mm, 7本3.33mm, 19本@130mm 22.5mm。
幾何制約チェック: pitch×Δφ/(2π) < 2R なら ValueError。

検証スクリプト: `contracts/check_strand_mesh_7wire.py`
可視化画像: `docs/verification/strand_mesh_7wire.png`

### 3. プロセス脱法摘発

#### 脱法カテゴリA: Process化すべきアセンブラ（4件）

| # | クラス | ファイル | 対応 |
|---|--------|---------|------|
| A1 | `ULCRBeamAssembler` | `elements/_beam_assembler.py` | 次セッション |
| A2 | `Hex8Assembler` | `elements/_hex8_assembler.py` | 次セッション |
| A3 | `MixedAssembler` | `elements/_mixed_assembler.py` | 次セッション |
| A4 | `_RigidEdgeAssembler` | `three_point_bend_jig.py` | 次セッション |

#### 脱法カテゴリB: frozen=True漏れ（9件）

- B1 `SolverStrategies`, B2 `SolverResultData`: 設計的mutable（許容）
- B3 `VerifyResult`: **今回frozen化済み**（snapshot_paths: list → tuple）
- B4 `StrandBatchResult`: 次セッション
- B5-B9: 内部管理用（低優先度）

#### 脱法カテゴリC: 大規模プライベート関数（13件）
- C2 `_batch_update_geometry` (175行), C3 `_build_contact_frame_batch` (108行) — HIGH
- C4-C13: MEDIUM（次セッション以降）

### 4. フォーカスガード更新

三点曲げフォーカスガードを解除し、7本撚線曲げ揺動を新フォーカスに設定。

---

## 再現手順

```bash
git checkout claude/twisted-wire-mesh-creation-tINC2
pip install -e .
python contracts/check_strand_mesh_7wire.py 2>&1 | tee /tmp/log-strand-mesh.log
python -m pytest xkep_cae/contact/geometry/tests/test_strategy.py xkep_cae/contact/solver/tests/test_process.py xkep_cae/mesh/tests/test_process.py tests/contact/test_strand_contact_process.py -x -q
python contracts/validate_process_contracts.py
```

---

## 次の課題

### Phase A: アセンブラProcess化（A1-A3, A4）
- [ ] A1: `ULCRBeamAssembler` → ファクトリProcess
- [ ] A2: `Hex8Assembler` → ファクトリProcess
- [ ] A3: `MixedAssembler` → ファクトリProcess
- [ ] A4: `_RigidEdgeAssembler` → `RigidEdgeAssemblerProcess`

### Phase B: DOF消去MPC実装
- [ ] `xkep_cae/constraints/mpc_elimination.py` — MPCEliminationProcess
- [ ] BoundaryData拡張（mpc_transform）
- [ ] 7本撚線端部参照点 + MPC結合

### Phase C: 7本撚線曲げ揺動Process
- [ ] `xkep_cae/numerical_tests/strand_bending_oscillation.py`
- [ ] StrandBendingOscillationProcess (BatchProcess)
- [ ] 端部参照点 + MPC + 曲げ処方変位 + 揺動サイクル

### Phase D: 幾何計算Process化（C2-C3）
- [ ] C2: `_batch_update_geometry` → BatchUpdateGeometryProcess
- [ ] C3: `_build_contact_frame_batch` → ContactFrameProcess

### Phase E: 摩擦・接触力Process化（C4-C8）
- [ ] C4-C8: 摩擦アセンブリ関数 → 個別Process

---

## 懸念事項・確認依頼

1. **ユーザー目視確認待ち**: `docs/verification/strand_mesh_7wire.png`
2. **gap=2.5mm**: 弦近似クロス効果が想定以上に大きく、要素密度32/ピッチでも線径の25%のgapが必要。要素密度を増やすか、`_compute_min_safe_gap`を外周間ギャップも考慮するよう改善すべき
3. **DOF消去法の設計**: BoundaryData拡張の詳細は次セッション
