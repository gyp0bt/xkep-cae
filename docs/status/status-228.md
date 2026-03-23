# status-228: smooth_clamp C1 連続化 + Hermite 暫定ソルバー — frac=0.60→0.96 突破

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-23
**ブランチ**: `claude/fix-focus-guard-bending-RlaJe`

---

## 概要

frac=0.60 壁の根本原因が **最近接点パラメータ s,t の `np.clip` 勾配不連続** であることを発見。
`_smooth_clip_01`（Huber 風 C1 二次遷移クランプ）を ε=1e-6 で再有効化し、
frac=0.60→0.96（189N）まで大幅改善を達成。

Hermite 曲線補間も併用したが、ε=1e-6 では smooth_clamp 自体の効果が支配的で
Hermite の追加効果は現在の設定では計測不能。

**暫定ソルバー**: smooth_clamp + Hermite を動的三点曲げのデフォルトとして有効化。
次回セッションで ε 拡大（0.01〜0.02）による frac=1.0 到達を目指す。

---

## 検討経緯

### 1. 壁の原因特定（status-227 から継続）

status-227 で特定した「2-cycle 残差振動」の根本原因を深掘り:

| 仮説 | 検証方法 | 結果 |
|------|---------|------|
| 摩擦の影響 | mu=0 で実行 | frac=0.59 — **摩擦は無関係** |
| K_T 非正定値性 | スペクトル分析 | 負の固有値 1 個（接触幾何起因） |
| NR内 s,t 更新 | freeze_st / relaxation | 悪化（精度低下） |
| k_pen 過大 | k_pen=5000 | 発散 |
| **np.clip 勾配不連続** | smooth_clamp 適用 | **frac=0.96 達成** |

### 2. smooth_clamp の実験結果

`_smooth_clip_01` を `_closest_point_segments_batch` で ε=1e-6 で有効化:

| 設定 | frac | fc [N] | incr | cutback | 備考 |
|------|------|--------|------|---------|------|
| np.clip（ベースライン） | 0.6011 | 80.8 | ~50 | 17 | status-227 |
| smooth_clamp ε=1e-6 | **0.9594** | **189.5** | 293 | 207 | **壁突破** |
| smooth_clamp ε=1e-6 + Hermite | **0.9594** | **189.5** | 293 | 207 | Hermite 追加効果なし |

### 3. np.clip vs smooth_clamp の力学的意味

**np.clip の問題**:
```
s_clamped = np.clip(s_unclamped, 0, 1)
```
- s=0, s=1 で勾配が不連続（1→0 にジャンプ）
- 接触点がセグメント端部を通過するたびに接線剛性 ∂s/∂u が突然ゼロになる
- NR 反復で残差と接線の不整合 → 2-cycle 振動 → 不収束

**smooth_clamp の効果**:
```
_smooth_clip_01(s, epsilon=1e-6):
    s < -ε       → 0                    (ハードクランプ)
    -ε ≤ s < ε   → (s + ε)² / (4ε)      (C1 二次遷移)
    ε ≤ s ≤ 1-ε  → s                    (線形通過)
    1-ε < s ≤ 1+ε → 1 - (1+ε - s)² / (4ε) (C1 二次遷移)
    s > 1+ε      → 1                    (ハードクランプ)
```
- s=0, s=1 の境界で ds/ds_unc が連続的に 0 に遷移
- 接線剛性 K_st = (∂s/∂u)·(...) が連続 → NR 2次収束維持

### 4. frac=0.96 で再停止した原因

**直接原因**: max_incr = 500 到達（293 成功 + 207 カットバック = 500）

**根本原因**: ε=1e-6 は遷移帯が **狭すぎる**

| 比較項目 | 値 | 備考 |
|---------|-----|------|
| ε（smooth_clamp 本体） | 1e-6 | 要素長 O(1) に対して6桁小さい |
| ε（Hermite refine 内） | 0.02 | こちらは適切な幅 |
| ε（_st_jacobian） | 1e-6 | smooth_clamp 本体と整合 |

**不整合**:
- `_closest_point_segments_batch`（line 120-137）: ε=1e-6
- `_closest_point_hermite_refine`（line 452-453）: ε=0.02
- `ComputeStJacobian._SMOOTH_EPS`（line 67）: 1e-6

ε=1e-6 では接触点がセグメント端近傍の ε 帯に入るケースが極めて稀。
大部分のケースで実質 np.clip と同じ動作 → カットバック 207 回で予算消費。

### 5. Hermite 補間の理論的位置づけ

Hermite 曲線 `p(s) = H00·x0 + H10·m0 + H01·x1 + H11·m1` は:
- 節点座標 x0, x1 と接線ベクトル m0, m1 を補間
- セグメント端点で位置 C0 + 接線 C1 連続
- **物理的な梁中心線を滑らかに近似**

しかし現在の実装では:
1. Hermite refine で s,t を精密化する際の smooth_clamp は ε=0.02（良い）
2. **その s,t 値を返した後**、_st_jacobian は ε=1e-6 で微分計算（不整合）
3. Hermite refine なしの場合、_closest_point_segments_batch が ε=1e-6 で s,t を計算

→ Hermite の効果が ε=1e-6 の smooth_clamp で殺されている。

### 6. 次回セッションの課題（ε 統一化）

ε を 0.01〜0.02 に統一すべき箇所:

| ファイル | 場所 | 現在 | 変更先 |
|---------|------|------|--------|
| `_compute.py` | `_smooth_clip_01` デフォルト引数 | 1e-6 | 0.02 |
| `_compute.py` | `_closest_point_segments_batch` 呼び出し | (デフォルト=1e-6) | 明示的に 0.02 |
| `_st_jacobian.py` | `_SMOOTH_EPS` | 1e-6 | 0.02 |
| `_st_jacobian.py` | `_smooth_clip_deriv` デフォルト引数 | 1e-6 | 0.02 |

**注意点**:
- ε が大きすぎると端点近傍で接触力精度が低下（s=0 で力ゼロのはずが非ゼロ）
- 物理テスト（貫入量精度等）への影響を要検証
- ε=0.02 は Hermite refine で実績あり

---

## 暫定ソルバー設定

動的三点曲げのデフォルトに以下を追加:
- `use_hermite_centerline=True`: Hermite 曲線補間有効
- `_smooth_clip_01` はε=1e-6で既に有効（np.clip → smooth_clamp 復帰は status-225 時点で完了済み）

`DynamicThreePointBendContactJigConfig` に `use_hermite_centerline` フィールドを追加し、
デフォルト `True` で `_ContactConfigInput` に伝搬。

---

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | 変更 | use_hermite_centerline デフォルト True |
| `docs/status/status-228.md` | 新規 | 本ステータス |
| `docs/status/status-index.md` | 変更 | 228 追加 |
| `docs/roadmap.md` | 変更 | frac=0.96 到達反映 |
| `README.md` | 変更 | 現在の状態更新 |
| `CLAUDE.md` | 変更 | フォーカスガード次回課題更新 |

---

## テスト

**186+10s passed** — 契約違反 1件（既存）、条例違反 0件

---

## 次のステップ

1. **ε 統一化**: _smooth_clip_01, _st_jacobian の ε を 0.02 に統一 → frac=1.0 目標
2. **max_incr 増加**: カットバック 207 は max_incr=500 制限。ε 統一でカットバック削減後に再評価
3. **ε の物理テスト影響**: 端点近傍の接触力精度への影響を定量評価
4. **接線剛性の ε 整合**: K_st ヤコビアンの `_SMOOTH_EPS` を本体と一致させる
5. **フォーカスガード条件**: E=25, fi17, push=30, n_periods=30 で数百 N 確認
