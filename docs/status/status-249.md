# status-249: プロセス脱法摘発 + 7本撚線メッシュ作成 + フォーカスガード更新

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-27
**ブランチ**: `claude/twisted-wire-mesh-creation-tINC2`
**テスト数**: 200+10s（変更なし）
**契約違反**: 1件（C3 既知、ComputeStJacobianProcess テスト未紐付け）

---

## 実施事項

### 1. プロセス脱法摘発

#### 公式契約違反（1件、既知）
- **C3**: `ComputeStJacobianProcess` — テスト未紐付け

#### 脱法修正（今回実施）
- **A4**: `_RigidEdgeAssembler` → `RigidEdgeAssemblerProcess`（PreProcess）
  - `xkep_cae/constraints/rigid_assembler.py` に独立化
  - `three_point_bend_jig.py` から旧クラス削除、Process経由に移行
  - テスト: `xkep_cae/constraints/tests/test_rigid_assembler.py`（7テスト）
- **B3**: `VerifyResult` → `frozen=True`（snapshot_paths: list → tuple）
- **B4**: `StrandBatchResult` → `frozen=True`（process_log: list → tuple, process内を一括生成に書換え）

#### 脱法摘発結果（対応ロードマップ → 下記「次の課題」参照）

| カテゴリ | 件数 | 重度 | 対応 |
|---------|------|------|------|
| A: Process化すべきアセンブラ | 4件 | HIGH | A4は今回修正。A1-A3は次セッション |
| B: frozen=True漏れ | 9件 | CRITICAL/MEDIUM | B3,B4は今回修正。B1,B2は設計的mutable |
| C: 大規模プライベート関数 | 13件 | HIGH/MEDIUM | 次セッション以降 |

#### 重要バグ修正
- **ContactSetupProcess: layer_ids → elem_layer_map 変換漏れ**
  - `mesh.layer_ids` が存在しても `_ContactConfigInput.elem_layer_map` に渡されていなかった
  - `exclude_same_layer=True` が機能せず、同層素線間の接触ペアが生成されていた
  - 7本撚線で 9389ペア → 3796ペア（同層除外適用後）、初期貫入 3577 → 0
- **StrandMeshProcess: layer_ids 未構築**
  - `TwistedWireMeshOutput` に `layer_ids` がなく、`getattr(..., None)` で常にNone
  - `strand_infos.layer` から要素ごとの `layer_ids` を構築するよう修正

### 2. 7本撚線メッシュ作成

仕様:
- E=130MPa, ρ=8.96e-9 t/mm³, 線径10mm（半径5mm）, ピッチ100mm
- 32要素/ピッチ, 3ピッチ, gap=0.5mm
- 節点679, 要素672, 接触ペア3796, **初期貫入ゼロ**

検証スクリプト: `contracts/check_strand_mesh_7wire.py`
可視化画像: `docs/verification/strand_mesh_7wire.png`

### 3. フォーカスガード更新

三点曲げフォーカスガードを解除し、7本撚線曲げ揺動を新フォーカスに設定。

---

## 再現手順

```bash
git checkout claude/twisted-wire-mesh-creation-tINC2
PYTHONPATH=. python contracts/check_strand_mesh_7wire.py 2>&1 | tee /tmp/log-strand-mesh.log
python -m pytest xkep_cae/constraints/tests/ xkep_cae/core/batch/tests/ xkep_cae/mesh/tests/ -v
python contracts/validate_process_contracts.py
```

---

## 次の課題（脱法修正ロードマップ）

### Phase A: アセンブラProcess化（3件）
- [ ] A1: `ULCRBeamAssembler` → ファクトリProcess
- [ ] A2: `Hex8Assembler` → ファクトリProcess
- [ ] A3: `MixedAssembler` → ファクトリProcess

### Phase B: DOF消去MPC実装
- [ ] `xkep_cae/constraints/mpc_elimination.py` — MPCEliminationProcess
- [ ] BoundaryData拡張（mpc_transform）
- [ ] 7本撚線端部参照点 + MPC結合

### Phase C: 7本撚線曲げ揺動Process
- [ ] `xkep_cae/numerical_tests/strand_bending_oscillation.py`
- [ ] StrandBendingOscillationProcess (BatchProcess)
- [ ] 端部参照点 + MPC + 曲げ処方変位 + 揺動サイクル

### Phase D: 幾何計算Process化（3件、HIGH）
- [ ] C2: `_batch_update_geometry` → BatchUpdateGeometryProcess
- [ ] C3: `_build_contact_frame_batch` → ContactFrameProcess
- [ ] C9: `_process_hermite` → HermiteStJacobianProcess

### Phase E: 摩擦・接触力Process化（5件、MEDIUM）
- [ ] C4: `_add_kst_contact` → ContactStiffnessProcess
- [ ] C5-C8: 摩擦アセンブリ4関数 → 個別Process

---

## 懸念事項・確認依頼

1. **ユーザー目視確認待ち**: `docs/verification/strand_mesh_7wire.png` — 7本撚線の端面・側面・上面図
2. **DOF消去法の設計**: BoundaryData拡張の設計について次セッションで詳細検討が必要
3. **アセンブラProcess化の粒度**: statefulオブジェクトをProcessの出力として扱うファクトリPatternが適切か、別の設計が良いか
