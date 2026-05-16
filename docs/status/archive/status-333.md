# status-333: M-κ追跡 + 接触ペアスナップショット — CR梁接触動解析のM-κヒステリシス直接取得基盤

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+9（新規9テスト追加）

## 概要

CR梁接触動解析（ContactFrictionProcess）にM-κ（曲げモーメント-曲率）追跡と接触ペアスナップショット記録機能を追加。status-332の反省「近似モデル同士の比較は循環論法」に基づき、**実摩擦接触から直接M-κヒステリシスと素線間接触力・滑り量を観測する基盤**を構築。

### 核心

1. **ContactFrictionProcess にM-κ追跡を追加**
   - `track_mk=True` で各収束インクリメントに (κ, M) を記録
   - `mk_moment_dofs`: f_intから曲げモーメントを取得するDOFインデックス群（合算）
   - `mk_curvature_func`: load_frac → κ の関数
   - M = Σ f_int[mk_moment_dofs]（全素線の反力モーメント合算）

2. **ContactFrictionProcess に接触ペアスナップショット記録を追加**
   - `track_contact_pairs=True` で各収束インクリメントの活性接触ペアをスナップショット
   - `ContactPairSnapshotEntry`: (elem_a, elem_b, p_n, gap, slip_s, slip_t, stick, dissipation)
   - `contact_pair_history`: tuple of (load_frac, tuple[ContactPairSnapshotEntry, ...])

3. **StrandBendingOscillationProcess に配線追加**
   - `track_contact_mk` / `track_contact_pairs` フラグを Config に追加
   - free-end モード: bending/oscillation 両フェーズでM-κ + ペア追跡を配線
   - 曲率関数: κ = θ(frac) / L（combined mode / 2-phase 両対応）

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/core/data.py` | `ContactPairSnapshotEntry` dataclass 追加、`ContactFrictionInputData` にM-κ/ペア追跡パラメータ追加、`SolverResultData` に `contact_pair_history` フィールド追加 |
| `xkep_cae/core/__init__.py` | `ContactPairSnapshotEntry` エクスポート追加 |
| `xkep_cae/contact/solver/process.py` | ソルバーループにM-κ記録 + 接触ペアスナップショット記録を追加（成功/失敗両return文） |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig` にフラグ追加、free-end モードの曲げ/揺動フェーズにM-κ配線 |
| `tests/numerical_tests/test_mk_tracking.py` | 新規テストファイル（9テスト） |

## テスト結果

| クラス | テスト名 | 結果 | 内容 |
|--------|----------|------|------|
| TestContactPairSnapshotEntryAPI | test_create_snapshot_entry | PASS | スナップショットエントリ作成 |
| TestContactPairSnapshotEntryAPI | test_snapshot_entry_frozen | PASS | frozen不変性 |
| TestMkTrackingConfig | test_default_tracking_disabled | PASS | デフォルト無効 |
| TestMkTrackingConfig | test_tracking_flags_settable | PASS | フラグ設定可能 |
| TestMkTrackingConvergence | test_mk_tracking_records_history | PASS | 2本撚線M-κ記録（κ単調増加、M非ゼロ） |
| TestMkTrackingConvergence | test_contact_pair_tracking_records_history | PASS | 2本撚線ペアスナップショット記録 |
| TestMkTrackingConvergence | test_mk_and_pairs_combined | PASS | M-κ+ペア同時追跡（23エントリ=23インクリメント） |
| TestMkTrackingDisabled | test_default_no_mk_history | PASS | 既存動作への回帰なし |
| TestMkTrackingDisabled | test_input_default_no_tracking | PASS | 入力デフォルト確認 |

## 設計判断

### M-κのモーメント取得方法

- f_int[prescribed_θ_y_dofs] の合算を使用
- 動的解析では慣性項 (M*a) の寄与があるが、準静的荷重（slow ramping）ではf_intが支配的
- update_reference() の前にf_intを計算しているため、収束状態の正しいf_intが得られる

### 接触ペアスナップショットの軽量化

- `_ContactPairOutput` / `_ContactStateOutput` の全フィールドではなく、M-κ検証に必要な最小限のフィールドのみ保持
- `p_n > 0` の活性ペアのみ記録（非活性ペアはノイズ）

## 次のステップ

- [ ] **7本撚線でM-κヒステリシスループを直接取得**（曲げ+揺動でティアドロップ形状を観測）
- [ ] **接触力・滑り量からκ_cr分布を実測**（ファイバー梁キャリブレーションデータ）
- [ ] **ピッチ依存性検証**（p=50/100/200 での散逸差を直接計測）
- [ ] Papailiouモデルのキャリブレーション → 予測モデルとして完成
- [ ] strand_cross_section_model.py / cable_dissipation.py の純粋関数をProcess化（C16違反解消）

## 契約違反

**12件**（status-332の4件 + strand_cross_section_model.py 追加分8件。本status での新規追加はなし）
