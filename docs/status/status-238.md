# status-238: n_periods=30 収束検証 + 接線剛性FD整合性診断

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-25
**テスト**: 190+10s+8 | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-237 の TODO 2項を実施:
1. **n_periods=30 収束検証** — 解析的剛体表面+粗メッシュの効果測定
2. **接線剛性FD整合性診断** — NR力収束が0件である根本原因の特定

---

## 1. n_periods=30 収束検証結果

### ベースライン比較

| 指標 | status-234 (旧メッシュ) | status-237 (剛体表面+粗メッシュ) | 改善率 |
|------|------------------------|--------------------------------|--------|
| frac | 1.0 | 1.0 | ✓ 完走 |
| fc | 208.6N | 216.96N | ≈同等 |
| **incr** | 1592 | **707** | **55%削減** |
| **cutback** | 2477 | **400** | **84%削減** |
| **cutback_rate** | 60.9% | **36.1%** | **大幅改善** |
| 力収束 | 0件 | 0件 | 未改善 |
| push_reached | 30.0mm | 30.0mm | ✓ |

**結論**: 解析的剛体表面+梁粗化（status-237）により、カットバック数84%削減。
ただし力収束は依然として0件（全ステップがdisp収束で通過）。

### n_periods=3 クイックテスト

| 指標 | 値 |
|------|-----|
| frac | 0.8806（max_increments=500 到達） |
| fc | 146.82N |
| cutback_rate | 42.3% |
| 力収束 | 0件 |

frac≈0.68 付近で壁（力残差 30反復で 1.0→0.45）。壁通過後は frac=0.88 まで進行。

---

## 2. 接線剛性FD整合性診断

### 手法

最小構成（2梁直交交差、4ノード）で接触接線剛性 K_c の中心差分有限差分検証:
```
K_c_FD[:,j] = -(f_c(u+ε*e_j) - f_c(u-ε*e_j)) / (2ε)
```

3構成で検証:
1. K_st なし（consistent_st_tangent=False、デフォルト）
2. K_st あり + 線形補間
3. K_st あり + Hermite補間

### 結果

| テスト | K_st | Hermite | 最大相対誤差 | 判定 |
|--------|------|---------|------------|------|
| 1 | **なし** | なし | **100%** | **不整合** |
| 2 | あり | なし | **0.00%** | **整合** |
| 3 | あり | **あり** | **33.3%** | **不整合** |

### 根本原因

**テスト 1（100%不整合）**: `consistent_st_tangent=False`（デフォルト）では、
接触点パラメータ s,t の変位依存性 `ds/du`, `dt/du` が接線剛性に含まれない。
変位摂動で s,t が変化すると力が大きく変化するが、K_c はこれを捕捉できず、
NR の線形化が不正確。結果として力収束に到達できない。

**テスト 2（0.00%）**: K_st を有効化すると線形補間では完全整合。
理論上はこれで NR の2次収束が回復するはず。

**テスト 3（33.3%）**: Hermite + K_st では z方向DOFに33%の不整合が残る。
原因: `∂p_n/∂s`（gapのs依存性を通じた力変化）項が K_st に含まれていない。
Hermite の場合、曲線の曲率により ∂gap/∂s ≠ 0 となりこの項が有効。

### K_st 有効化の実験

`consistent_st_tangent=True` を三点曲げジグに追加してテスト実施。
**結果: NR が発散し、1ステップも完了できず。**

原因: K_st が K_T に大きな非対称・非正定値成分を追加し、
status-227 で報告された K_T 非正定値性問題が悪化。
線形ソルバーが有効な Newton 方向を計算できなくなる。

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `contracts/check_rigid_surface_effect.py` | **新規**: n_periods=30 収束検証スクリプト |
| `contracts/check_rigid_surface_quick.py` | **新規**: n_periods=3 クイック検証スクリプト |
| `contracts/check_tangent_consistency.py` | **新規**: 接線剛性FD整合性検証スクリプト |
| `docs/status/status-238.md` | 本ステータス |
| `docs/status/status-index.md` | 更新 |
| `docs/roadmap.md` | 更新 |
| `README.md` | 更新 |

---

## TODO

- [ ] **K_st の安全な有効化**: Levenberg-Marquardt正則化（K_T + λI）で正定値性を保証しつつ K_st を適用する。λ の自動調整（Trust region method）が必要
- [ ] **Hermite K_st の ∂p_n/∂s 項追加**: gap の s,t 依存性を K_st に追加して Hermite 33% 不整合を解消
- [ ] **摩擦アセンブリの Hermite 完全対応**: use_hermite=False デフォルトの解消
- [ ] **K_st スケーリング制御**: K_st を段階的に導入する continuation 法の検討

---

## 設計上の懸念

1. **K_st と正定値性のトレードオフ**: K_st は接線整合性を改善するが、K_T の正定値性を破壊する。
   NR の線形ソルブが失敗するリスクがある。修正 Newton 法や Trust region 法との組み合わせが必要。

2. **disp 収束への依存**: 現在のソルバーは力収束なしで disp 収束のみで動作。
   n_periods=30 で frac=1.0 に到達しているが、解の精度が力収束時と比べて低い可能性。
   ただし、解析解との比較（216.96N vs EB梁理論）で大きな乖離はない。

3. **Hermite vs 線形の整合性**: 幾何計算（_closest_point_hermite_refine）は Hermite、
   力評価は Hermite shape functions、接線剛性は K_mat（直接部分）+ K_st（s,t部分）。
   K_st の Hermite 版に ∂p_n/∂s が不足しており、完全な整合を達成するには追加実装が必要。

---

## 運用メモ

- n_periods=30 テストは wall_time 数分（status-237 の改善効果で大幅短縮）
- FD 整合性検証は数秒で完了、回帰テストとして `contracts/check_tangent_consistency.py` を使用可
- K_st 有効化は現時点で非推奨（NR 発散の原因となる）
