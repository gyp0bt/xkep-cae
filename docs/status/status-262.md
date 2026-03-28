# status-262: delta_h最適値探索 + three_point_bend_jig huber_delta_h貫通 + 3Dパイプレンダリング

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/check-status-todos-hf9lH`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4（新規4件）→ **合計574 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. δ=1000 完走テスト実行確認（status-261 TODO #1）

`test_strand_bending_full_completion_delta1000` を CI 環境で実行確認。

- **結果**: PASSED（337.7秒）
- smoothing_delta=1000 で frac≥0.95 の要件をクリア

### 2. huber_delta_h 直接指定スイープ（status-261 TODO #2）

`contracts/bench_huber_delta_h.py` を作成し、7本撚線曲げ揺動で delta_h を 0.005〜0.050 の範囲でスイープ。

#### スイープ結果

| delta_h | frac | incr | cutback | NR_avg | time |
|---------|------|------|---------|--------|------|
| 0.000 (自動δ=2000) | 0.591 | 195 | 3 | 9.2 | 173s |
| 0.005 | 0.373 | 22 | 1 | 8.0 | 18s |
| 0.010 | 0.355 | 17 | 2 | 7.1 | 16s |
| 0.015 | 0.355 | 17 | 2 | 7.1 | 16s |
| 0.020 | 0.998 | 163 | 5 | 8.0 | 187s |
| **0.025** | **1.000** | **129** | 6 | **7.3** | **132s** ← 最速 |
| 0.030 | 0.372 | 22 | 3 | 6.6 | 22s |
| **0.040** | **1.000** | **126** | 7 | 7.6 | 169s |
| 0.050 | 0.969 | 204 | 2 | 7.8 | 220s |

#### 分析

- **最適値**: delta_h=0.025 が最速完走（132s）。δ=1000 間接指定（176s）より 25% 高速
- **有効範囲**: delta_h ∈ [0.020, 0.025] ∪ {0.040} で完走可能（0.050はfrac=0.97）
- **非単調性**: delta_h=0.030 で急落（frac=0.37）、0.040 で再復活する非単調パターン
  - これは Huber 遷移幅と活性集合変化の離散的相互作用が原因
  - 特定の delta_h で活性集合の切り替わりが NR ステップと同期し発散を誘発
- **推奨値**: delta_h=0.025（梁-梁接触の標準推奨値）
- **注意**: 0.030 が非完走なので、delta_h は問題に敏感。デフォルト化は時期尚早

### 3. three_point_bend_jig への huber_delta_h 貫通（status-261 TODO #3）

両コンフィグに `huber_delta_h` パラメータを追加し、`_ContactConfigInput` へパススルー。

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `ThreePointBendContactJigConfig.huber_delta_h` + `DynamicThreePointBendContactJigConfig.huber_delta_h` 追加、`_ContactConfigInput` パススルー |

### 4. 3Dパイプレンダリング（貫入検査）

`contracts/visualize_strand_bending_3d.py` を作成。7本撚線の最終変形状態を3Dパイプとして描画。

- **3Dパイプ図**: `docs/verification/strand_bending_3d_pipe.png`（側面 + 端面 × 4フレーム）
- **断面図**: `docs/verification/strand_bending_cross_section.png`（Z中央のワイヤ断面）
- **貫入検査結果**: 全ストランド間ギャップ >= 0（**貫入なし**）

### 5. テスト追加

| テストファイル | テスト名 | 内容 |
|--------------|---------|------|
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | `test_contact_jig_config_default_zero` | huber_delta_h デフォルト値確認 |
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | `test_dynamic_config_default_zero` | 動的版 huber_delta_h デフォルト確認 |
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | `test_contact_jig_config_manual` | 手動指定可能確認 |
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | `test_dynamic_config_manual` | 動的版手動指定確認 |

---

## テスト結果

- 新規テスト: 4件（全通過）
- 既存テスト: 574 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-hf9lH
pip install -e .
# 新規テスト（高速）
python -m pytest xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py -v --timeout=30 -k "not slow"
# δ=1000 完走テスト（~338s）
python -m pytest tests/numerical_tests/test_strand_bending_convergence.py::TestStrandBendingConvergence::test_strand_bending_full_completion_delta1000 -v -s --timeout=600 2>&1 | tee /tmp/log-delta1000.log
# delta_h スイープ（~20分）
python contracts/bench_huber_delta_h.py 2>&1 | tee /tmp/log-delta-h-sweep.log
# 3D パイプレンダリング（~3分）
python contracts/visualize_strand_bending_3d.py 2>&1 | tee /tmp/log-strand-3d.log
# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **delta_h デフォルト値の検討**: delta_h=0.025 が梁-梁で最速だが、0.030 が非完走の非単調性あり。安全なデフォルト値の設定にはさらなる検証が必要
2. **three_point_bend_jig での delta_h 効果検証**: 剛体-梁接触で delta_h=0.025 が有効か実験が必要（k_pen スケールが異なる）
3. **T1 Hermite atol 厳格化**: Hermite 非局所 ∂g/∂u 対応（4ノードペア外のDOF結合）が必要（status-258 から継続）
4. **NR力収束改善**: 中盤後〜終盤で 25 反復が力収束に不足、disp 収束で抜ける状態

### 設計メモ

- delta_h の有効範囲は非単調: [0.020, 0.025] と {0.040} で完走するが 0.030 は失敗
- この非単調性は Huber 平滑化幅と活性集合変化の離散的な相互作用に起因
- smoothing_delta=0 + huber_delta_h>0 の場合、プロセス内部で自動推定 δ=1000/r が引き続き設定されるが、`_resolve_delta_h` で huber_delta_h が優先される（正常動作）
- 3Dパイプレンダリング: smoothing_delta=1000 で frac=1.0 完走、貫入なし確認

---

## 懸念・設計メモ

1. **delta_h 非単調性**: 0.025→0.030 で急落するのは活性集合の切り替わりタイミングに依存。問題のメッシュ・材料・荷重条件により最適値が変わる可能性。デフォルト値設定は慎重に
2. **delta_h=0（自動推定 δ=2000）の挙動**: smoothing_delta=0.0 を渡しても内部で 1000/r=2000 に自動推定される。huber_delta_h=0.0 の場合この自動推定が使われ delta_h=k_pen/2000≈0.015 相当。frac=0.59 は delta_h=0.015 直接指定（frac=0.35）より良い結果だが、これは同時実行の自動推定パスが異なるルートを通る可能性あり（要調査）
