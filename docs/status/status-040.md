# status-040: .inp実行スクリプト + Abaqus三点曲げバリデーション + inp_runnerユーティリティ

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-19

## 概要

status-039 の TODO「examples の .inp ファイルを使った実際の解析実行スクリプトの追加」を実行。
加えて、Abaqus三点曲げ（`assets/test_assets/Abaqus/1-bend3p/`）のバリデーションテストを実装。

テスト数 845 → 865（+20テスト）。

## 実施内容

### 1. `inp_runner.py` 新設 — .inp → 梁解析モデル構築ユーティリティ

| クラス/関数 | 概要 |
|-------------|------|
| `BeamModel` | パース済み .inp から構築された梁解析モデル（nodes, element_groups, material, sections, fixed_dofs 等） |
| `build_beam_model_from_inp(mesh)` | AbaqusMesh → BeamModel 変換。要素タイプ（B21/B31）・断面タイプ（RECT/CIRC/PIPE）を自動判定 |
| `solve_beam_static(model, f_ext)` | 線形静解析の実行（assemble → apply_dirichlet → solve_displacement） |
| `node_dof(model, node_label, local_dof)` | ノードラベルとローカルDOF番号からグローバルDOFインデックスを返す |

#### 内部ヘルパー

- `_detect_3d(mesh)`: B31→3D, B21→2D を自動判定
- `_build_material(mesh)`: *MATERIAL + *ELASTIC → BeamElastic1D
- `_build_section(bsec, is_3d)`: RECT/CIRC/PIPE → BeamSection/BeamSection2D
- `_build_element(section, is_3d, formulation, kappa, direction)`: Timoshenko/EB 要素生成
- `_get_connectivity_for_elset(mesh, elset, label_to_index)`: ELSET → 0基底接続配列
- `_build_fixed_dofs(mesh, label_to_index, ndof_per_node)`: *BOUNDARY → fixed_dofs

### 2. `run_examples.py` 新設 — サンプル .inp 解析実行スクリプト

`examples/` 配下の4つの .inp ファイルを読み込んで線形静解析を実行し、
解析解との比較結果を表示する。

| 解析ケース | 解析解比較 | 結果 |
|-----------|-----------|------|
| cantilever_beam_3d | Timoshenko片持ち梁（せん断含む） | 0.0000% |
| three_point_bending | Timoshenko三点曲げ（せん断含む） | 0.0000% |
| portal_frame | — (定性確認のみ) | 正常完了 |
| l_frame_3d | — (定性確認のみ) | 正常完了 |

### 3. `test_inp_runner.py` 新設 — 10テスト

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestBuildBeamModel | 6 | モデル構造(3D/2D), 複数断面, パイプ断面, DOFマッピング, ノードセット変換 |
| TestSolveBeamStatic | 4 | 片持ち梁解析解(< 0.1%), 三点曲げ解析解(< 0.1%), 門型フレーム/L型フレーム定性 |

### 4. `test_abaqus_validation_bend3p.py` 新設 — 10テスト（Abaqusバリデーション）

#### モデル概要

- Abaqusモデル: `assets/test_assets/Abaqus/1-bend3p/` のAbaqus結果との比較
- yz対称1/2モデル、線長100mm（半モデル: x=0〜50mm）
- 円形断面 直径1mm (r=0.5mm)、E=100 GPa, ν=0.3
- xkep-cae側: 同じワイヤメッシュ（100要素）、接触の代わりに点支持（x=25mm）

#### テスト一覧

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestAbaqusBend3pElastic | 5 | 線形性検証, **Abaqus剛性比較(1.09%)**, 反力方向, 変位プロファイル, 対称BC |
| TestAbaqusBend3pCurvature | 1 | 中心付近の曲率が非ゼロ |
| TestModelConstruction | 4 | ノード数, 支持位置, 断面特性, メッシュ収束性(< 2%) |

#### 剛性比較結果

```
xkep-cae 線形剛性: 0.9416 N/mm
Abaqus 線形剛性:   0.9520 N/mm
相対差異: 1.09%
```

許容誤差10%に対して**1.09%**で良好な一致。
差異の主因: Abaqusは接触分布荷重 + 動的効果 + NLGEOM、xkep-caeは点支持 + 静的線形。

### 5. `apply_dirichlet` スパース行列の潜在バグ発見

バリデーション実装中に `apply_dirichlet()` のスパース行列（CSR）処理で非ゼロ規定変位の場合に
数値的問題が生じることを発見。

**症状**: 非ゼロ規定変位（変位制御）をスパース行列で適用すると、拘束されるべきDOFの値が
異常値になる（例: ux=0が期待値なのに ux=12021）。

**原因**: CSR行列の行/列消去処理で、複数DOFを順次処理する際にスパース構造の変更が
後続の列抽出に正しく反映されない。

**対処**: `test_abaqus_validation_bend3p.py` の `solve_displacement_control()` では
密行列で BC を適用する方式を採用。これにより正しい結果が得られることを確認済み。

**影響範囲**: ゼロ値の境界条件（f -= K[:,d]*0 が無影響）では問題が顕在化しないため、
既存テストへの影響なし。非ゼロ規定変位を使う場合のみ注意が必要。

### 6. `xkep_cae/io/__init__.py` 更新

新規シンボルのエクスポート追加:
- `BeamModel`, `build_beam_model_from_inp`, `node_dof`, `solve_beam_static`

### 7. 既存テストへの影響

- **既存テスト 845 件**: 全パス（破壊なし）
- **新規テスト 20 件**: 全パス
- **合計 865 件**: 841 passed, 24 skipped

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/io/inp_runner.py` | **新規** — .inp → BeamModel 変換ユーティリティ |
| `xkep_cae/io/__init__.py` | エクスポート追加（BeamModel, build_beam_model_from_inp, node_dof, solve_beam_static） |
| `examples/run_examples.py` | **新規** — サンプル .inp 解析実行スクリプト |
| `tests/test_inp_runner.py` | **新規** — 10テスト（モデル構築 + 解析解比較） |
| `tests/test_abaqus_validation_bend3p.py` | **新規** — 10テスト（Abaqus三点曲げバリデーション） |
| `README.md` | テスト数・バリデーション反映 |
| `docs/roadmap.md` | テスト数更新 |
| `docs/status/status-index.md` | status-040 行追加 |

## テスト数

845 → 865（+20テスト）

## 確認事項・懸念

1. **`apply_dirichlet` の非ゼロ規定変位バグ**: スパース CSR 行列での非ゼロ規定変位処理に問題がある。密行列アプローチで回避中。ソース側の修正は今後のTODOとする
2. **Abaqus比較の許容誤差**: 現在10%設定。接触分布・動的効果・NLGEOMの差異を考慮した値。モデル簡略化をさらに精密にすれば5%以下にできる可能性がある
3. **弾塑性バリデーション未着手**: `go_idx2_RF.csv`（弾塑性解析結果）との比較は将来の拡張候補

## TODO

- [ ] `apply_dirichlet` のスパース行列非ゼロ規定変位バグの修正
- [ ] Abaqus弾塑性三点曲げ（idx2）のバリデーション
- [ ] 要素単体剛性のAbaqus比較（接触フェーズ完了後に入念に実施）
- [ ] Phase C3: 摩擦 return mapping + μランプ
- [ ] Phase C4: merit line search + 探索/求解分離の運用強化

---
