# status-213: 接触なし梁揺動解析 + 3D応力コンターレンダリング

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-19
**ブランチ**: `claude/verify-beam-oscillation-analysis-TTOHo`

---

## 概要

三点曲げの前準備として、接触なしで単純支持梁を初速度制御で動的揺動させる解析を実施。
非線形域での挙動を3D応力コンターで可視化し、物理的妥当性を検証した。

## 変更内容

### 1. BeamOscillationProcess（新規）

| ファイル | 操作 |
|---------|------|
| `xkep_cae/numerical_tests/beam_oscillation.py` | **新規** — 接触なし梁揺動解析 Process |
| `xkep_cae/numerical_tests/docs/beam_oscillation.md` | **新規** — ドキュメント |
| `xkep_cae/numerical_tests/tests/test_beam_oscillation.py` | **新規** — @binds_to テスト |

**仕様:**
- 単純支持梁（ピン+ローラー）に中央初速度 v₀ を付与
- メッシュサイズ ≈ 半径（n_elems=100 for L=100mm, d=2mm）
- 動的ソルバー（GeneralizedAlpha, rho_inf=0.9）
- 初期時間増分: T₁/100（小さめ）
- 適応時間増分制御

### 2. StressContour3DProcess（新規）

| ファイル | 操作 |
|---------|------|
| `xkep_cae/output/stress_contour.py` | **新規** — 3D応力コンター PostProcess |
| `xkep_cae/output/docs/stress_contour.md` | **新規** — ドキュメント |
| `xkep_cae/output/tests/test_stress_contour.py` | **新規** — @binds_to テスト |
| `xkep_cae/output/__init__.py` | **更新** — re-export 追加 |

**機能:**
- 変形梁のチューブ状3Dレンダリング
- 要素ごと最大曲げ応力のカラーマッピング（jet カラーマップ）
- XY側面ビュー + 斜視ビュー
- 時刻歴プロット（変位 + 応力）

### 3. テスト

| ファイル | テスト数 |
|---------|---------|
| `tests/test_beam_oscillation.py` | 11 passed + 2 xfail |

**テスト構成:**
- `TestBeamOscillationAPI`: 設定生成、応力計算の単体テスト（3件）
- `TestBeamOscillationConvergence`: 小振幅/大振幅の収束テスト（2件, slow）
- `TestBeamOscillationPhysics`: 振幅比、応力分布、変位有界性テスト（6件, slow）
- `TestStressContour3DRendering`: レンダリング出力テスト（1件, slow）

### 4. 検証画像

出力先: `tmp/oscillation/`（一覧: [list.md](../../tmp/oscillation/list.md)）

| ファイル | 内容 |
|---------|------|
| `tmp/oscillation/beam_osc_stress3d_000.png` 〜 `_007.png` | 時系列3D応力コンター（8フレーム） |
| `tmp/oscillation/beam_osc_time_history.png` | 変位・応力時刻歴 |
| `tmp/oscillation/list.md` | 画像一覧+解析条件+考察 |

## 解析結果サマリ

| 項目 | 値 |
|------|-----|
| 収束 | Yes |
| インクリメント数 | 303 |
| 計算時間 | ~67s |
| 最大変位 | 1.128 mm |
| 最大曲げ応力 | 304.7 MPa |
| 固有振動数（解析解） | 396.4 Hz |
| 固有周期 | 2.52 ms |

## 発見された問題

### UL + GeneralizedAlpha 結合問題（重要）

**現象**: 梁が「揺動」せず単調に変位する。
- 変位時刻歴が 0 → -1.1mm と単調減少
- 振動（方向反転）が観測されない

**原因推定**: UL定式化の update_reference 後、state.u が増分ではなく累積値のまま時間積分に渡される。
- assembler.assemble_internal_force(state.u) は現在参照配置からの増分を期待
- 動的ソルバーでは state.u が累積変位のまま渡されるため、内力が過大評価される
- 結果として復元力が不正確になり、振動が発生しない

**影響**: status-211 TODO「dt_sub 問題」と同根。接触なし揺動でも再現。

### エネルギー計算の不整合

UL定式化では `0.5 * u_total^T * f_int(u_total)` がひずみエネルギーの正しい推定にならない。
正確なエネルギー評価にはULステップごとの増分仕事の累積が必要。

## テスト結果

```
11 passed, 2 xfailed (tests/test_beam_oscillation.py)
5 passed (@binds_to + API テスト)
ruff check: All checks passed
ruff format: 119 files already formatted
契約違反: 1件（既存の1件、新規0件）
```

## TODO

- [ ] **UL + 動的時間積分の結合修正**: state.u の増分/累積管理を明確化し、振動を正しく再現
- [ ] **エネルギー計算の修正**: ULステップごとの増分仕事累積方式に変更
- [ ] **数値粘性の定量評価**: 振動修正後に rho_inf 依存性を検証
- [ ] 応力コンターの降伏応力ライン追加（材料非線形の閾値可視化）

## 設計上の懸念

- UL+動的問題は ContactFrictionProcess の内部設計に起因。修正には solver/process.py の
  ステップ管理（predict → NR → correct → update_reference）の再設計が必要。
- 現状の動的解析は事実上「慣性力付き準静的荷重漸増」として動作しており、
  自由振動の再現には根本的な修正が必要。
- 三点曲げ揺動の前提条件として、この問題の解決が必須。

---
