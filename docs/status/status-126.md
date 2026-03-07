# status-126: NCPソルバー段階的活性化移植 + NCP曲げ揺動テスト + 2D投影置換

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-07
**テスト数**: 2271（fast: 1691 + 1 xfailed / slow: 362 / deprecated: 218）

## 概要

statusファイルのTODOに基づき、以下の3つの主要改善を実施。

1. **NCPソルバーへの段階的活性化メカニズム移植**（solver_hooks.py → solver_ncp.py）
2. **NCP曲げ揺動収束テスト新規作成**（S3真の目標: 非線形梁の90度曲げ+揺動）
3. **既存3Dプロットの2D投影版への完全置換**（mplot3dアスペクト比問題の解消）

## 実施内容

### 1. NCPソルバーへの段階的活性化メカニズム移植

**問題**: `staged_activation_steps`（層別接触ペアの段階的投入）が旧ALソルバー（`solver_hooks.py`）にのみ実装されており、NCPソルバー（`solver_ncp.py`）では使用できなかった。37本Layer1+2テストの一斉活性化（276ペア同時）が不収束の主因。

**対処**:
- `solver_ncp.py` のステップ開始時（broadphase後、Newton反復前）に `staged_activation_steps` ロジックを追加
- `ContactManager.compute_active_layer_for_step()` → `filter_pairs_by_layer()` の呼び出しを挿入

**テスト有効化**:

| テスト | staged_activation_steps | 意図 |
|--------|------------------------|------|
| 37本Layer1+2 | 20 | 4層構造で段階的投入 |
| 61本Layer1 | 25 | 5層構造で段階的投入 |
| 91本Layer1 | 30 | 6層構造で段階的投入 |

### 2. NCP曲げ揺動収束テスト新規作成

**背景**: 既存の多素線収束テストは全て**線形Timoshenko梁の径方向圧縮**。S3の真の目標は**非線形梁（CR: Co-Rotational）の90度曲げ+サイクル変位（揺動）**であり、径方向圧縮は序の口。

**新規テストファイル**: `tests/contact/test_ncp_bending_oscillation.py`（8テスト）

| クラス | テスト | 内容 |
|--------|--------|------|
| `TestNCP7StrandBendingOscillation` | `test_ncp_7strand_bending_45deg` | 7本: CR梁45度曲げ（NCP収束必須） |
| | `test_ncp_7strand_bending_90deg` | 7本: CR梁90度曲げ（NCP収束必須） |
| | `test_ncp_7strand_bending_oscillation_full` | 7本: 90度曲げ+揺動1周期（S3ベンチマーク） |
| `TestNCP19StrandBendingOscillation` | `test_ncp_19strand_bending_45deg` | 19本: 45度曲げ（収束トラッキング） |
| | `test_ncp_19strand_bending_oscillation` | 19本: 45度曲げ+揺動（収束トラッキング） |
| `TestNCP7StrandBendingPhysics` | `test_tip_displacement_direction` | 物理テスト: 先端変位方向の妥当性 |
| | `test_penetration_ratio_within_limit` | 物理テスト: 貫入比がワイヤ直径の5%以内 |

**技術詳細**:
- `run_bending_oscillation()` を `use_ncp=True` で呼び出し
- CR（Co-Rotational）梁アセンブラによる幾何学的非線形解析
- Phase 1: 変位制御（端部回転角の処方）で曲げ
- Phase 2: z方向サイクル変位（揺動）
- 16要素/ピッチ厳守（`n_elems_per_strand=16`）

### 3. 既存3Dプロットの2D投影版への完全置換

**問題**: `plot_twisted_wire_3d_surface` と `plot_beam_3d_stress_contour` が mplot3d の `Poly3DCollection` を使用しており、アスペクト比が1:1:1にならない問題があった。

**対処**: 四元数ベースの3D→2D投影方式（status-123で実装済みの `_project_3d_to_2d`, `_beam_surface_polys_2d` 等）を使用して完全置換。

| 関数 | 旧方式 | 新方式 |
|------|--------|--------|
| `plot_twisted_wire_3d_surface` | mplot3d Poly3DCollection | 四元数2D投影 + PolyCollection |
| `plot_beam_3d_stress_contour` | mplot3d Poly3DCollection | 四元数2D投影 + PolyCollection |

**検証プロット図の再生成**:
- `docs/verification/twisted_wire_3d_surface.png` — 再生成完了
- `docs/verification/beam_3d_stress_contour.png` — 再生成完了

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver_ncp.py` | 段階的活性化メカニズム追加 |
| `tests/contact/test_ncp_convergence_19strand.py` | 37/61/91本テストにstaged_activation追加 |
| `tests/contact/test_ncp_bending_oscillation.py` | **新規**: NCP曲げ揺動収束テスト8件 |
| `tests/generate_verification_plots.py` | 3Dプロット2関数を2D投影版に置換 |
| `docs/verification/twisted_wire_3d_surface.png` | 2D投影版に再生成 |
| `docs/verification/beam_3d_stress_contour.png` | 2D投影版に再生成 |
| `README.md` | テスト数・状態更新 |
| `docs/roadmap.md` | S3 TODO更新 |
| `docs/status/status-index.md` | status-126追加 |

## 設計上の懸念・ユーザーへの確認事項

1. **7本NCP曲げ揺動の収束状況**: CR梁+NCP変位制御での90度曲げは、旧ALソルバーでは収束実績あり。NCP版も`run_bending_oscillation`の機構を利用するため収束が期待されるが、CI環境でのslow test実行で確認が必要
2. **19本曲げ揺動**: 現時点では収束を必須としない（トラッキング目的）。径方向圧縮と比べ非線形度が格段に高い
3. **段階的活性化の効果検証**: 37本Layer1+2の収束改善は、段階的活性化＋NCPの組み合わせで初めてテスト可能に。CI slowテストでの実行結果待ち

## 運用フィードバック

### 効果的な点
- 段階的活性化のメカニズムが既にContactManagerに実装済みだったため、solver_ncp.pyへの移植は5行の追加で完了
- `run_bending_oscillation` が既にNCP対応していたため、テスト作成が効率的

### 非効果的な点
- CJKフォントが環境にないため検証プロットの日本語タイトルが文字化け（機能に影響なし）

## TODO

- [ ] 7本NCP曲げ揺動テストのCI実行確認（slow test）
- [ ] 37本Layer1+2の段階的活性化による収束改善確認
- [ ] 19本曲げ揺動の収束達成（S3拡張目標）
- [ ] 37本以上での曲げ揺動テスト追加
- [ ] 段階的活性化パラメータのチューニング（staged_activation_stepsの最適値）

---

[← README](../../README.md)
