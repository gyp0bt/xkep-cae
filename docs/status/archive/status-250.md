# status-250: アセンブラ Process 化（A1-A3）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-27
**ブランチ**: `claude/check-status-todos-IpSV1`
**テスト数**: 200+10s+16（新規16件）
**契約違反**: 1件（C3 既知、ComputeStJacobianProcess テスト未紐付け）

---

## 実施事項

### アセンブラ Process 化（status-249 Phase A の A1-A3）

status-249 で摘発された脱法カテゴリA（Process化すべきアセンブラ）のうち
A1-A3 を完了。

| # | クラス | ファイル | Process 名 | 状態 |
|---|--------|---------|------------|------|
| A1 | `ULCRBeamAssembler` | `elements/_beam_assembler.py` | `ULCRBeamAssemblerProcess` | **完了** |
| A2 | `Hex8Assembler` | `elements/_hex8_assembler.py` | `Hex8AssemblerProcess` | **完了** |
| A3 | `MixedAssembler` | `elements/_mixed_assembler.py` | `MixedAssemblerProcess` | **完了** |
| A4 | `_RigidEdgeAssembler` | `constraints/rigid_assembler.py` | `RigidEdgeAssemblerProcess` | 完了済み（status-249） |

### 変更内容

#### 新規ファイル
- `xkep_cae/elements/tests/__init__.py` — テストパッケージ
- `xkep_cae/elements/tests/test_assembler_process.py` — 全16テスト（@binds_to紐付き）
- `docs/elements.md` — アセンブラ Process ドキュメント

#### 変更ファイル（7ファイル）
- `xkep_cae/elements/_beam_assembler.py` — Process + frozen I/O 追加
- `xkep_cae/elements/_hex8_assembler.py` — Process + frozen I/O 追加
- `xkep_cae/elements/_mixed_assembler.py` — Process + frozen I/O 追加
- `xkep_cae/elements/__init__.py` — re-export 追加（9クラス）
- `xkep_cae/numerical_tests/three_point_bend_jig.py` — 3 Process の uses 更新 + 直接インスタンス化を Process API 経由に変更
- `xkep_cae/numerical_tests/beam_oscillation.py` — uses 更新 + Process API 経由に変更

### パターン

RigidEdgeAssemblerProcess（status-249）と同じパターンを踏襲:
1. 既存クラスをプライベート実装として維持（`_` prefix 不要、元々プライベートモジュール内）
2. frozen dataclass の Input/Output を追加
3. PreProcess[Input, Output] のラッパー Process を追加
4. 使用箇所を Process API 経由に変更
5. uses 宣言を更新

---

## テスト結果

- 新規テスト: 16件（全合格）
- 既存テスト: 508 passed, 19 skipped（全合格）
- 契約違反: 1件（既知 C3）
- lint: 全合格
- 既知の無関係失敗: `test_stress_contour.py`（matplotlib 依存）

---

## 再現手順

```bash
git checkout claude/check-status-todos-IpSV1
pip install -e .
python -m pytest xkep_cae/elements/tests/test_assembler_process.py -x -q
python -m pytest tests/elements/test_beam_assembler.py -x -q
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次の課題（status-249 引き継ぎ）

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

### frozen=True 漏れ（B4）
- [ ] B4: `StrandBatchResult` → frozen化

---

## 懸念事項

1. **test_stress_contour.py**: matplotlib 未インストール環境で失敗。今回の変更とは無関係だが、CI 環境で問題になる可能性あり。
2. **A4 完了済み**: status-249 で RigidEdgeAssemblerProcess は実装済み。Phase A は全完了。
