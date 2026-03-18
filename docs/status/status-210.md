# status-210: smooth_penalty ソルバー復元 + HEX8 連続体要素ジグ基盤

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/check-status-todos-9DQ07`

---

## 概要

リファクタリング（status-175〜207）で旧 `solver_smooth_penalty.py`（691行モノリシック、status-147 で収束確認済み）から
Process アーキテクチャへの移行時にロストした成功ソルバー構成を復元。
HEX8 連続体要素ジグの基盤を新規実装（収束は未解決 → TODO）。

## 1. smooth_penalty ソルバー復元

### 変更一覧（strategy + process + meta）

| Process | 変更 | meta ver |
|---------|------|---------|
| `SmoothPenaltyContactForceProcess` | AL 力評価 `p_n = max(0,λ) + k_pen*softplus(g)` + ゼロ接線 + INACTIVE スキップ廃止 | 1.0.0 → **2.0.0** |
| `DetectCandidatesProcess` | smooth_penalty: 新規ペアを ACTIVE で作成、候補外ペアの INACTIVE 化をスキップ | 2.0.0 → **2.1.0** |
| `UpdateGeometryProcess` | smooth_penalty: active set 更新をスキップ（常に ACTIVE） | — |
| `ContactFrictionProcess` | beam_E/I/L + smoothing_delta + n_uzawa_max + tol_uzawa を default_strategies() に伝搬 | — |

### 修正した問題

| # | 問題 | 根本原因 | 修正 |
|---|------|---------|------|
| 1 | 接触チャタリング（active=2↔0 振動） | smooth_penalty evaluate() が INACTIVE ペアをスキップ | INACTIVE スキップ完全廃止。softplus は C∞ |
| 2 | NR 不定値化 | softplus 接線 dp/dg < 0（負定値）を K_T に加算 | ゼロ接線（NCP と同様）。Uzawa が接触剛性を担う |
| 3 | Uzawa 外部ループ無効 | evaluate() で λ を力に含めていない | `p_n = max(0,λ) + k_pen*softplus(g)`（拡大ラグランジアン） |
| 4 | auto k_pen 未使用 | default_strategies() に beam_E/I/L 未伝搬 | process.py → default_strategies() 経由で伝搬 |
| 5 | n_uzawa_max 固定 | config → strategy への接続切れ | n_uzawa_max, tol_uzawa, smoothing_delta を伝搬 |

### 変更ファイル

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/contact_force/strategy.py` | AL力 + ゼロ接線 + INACTIVE廃止 |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | ゼロ接線テスト更新 |
| `xkep_cae/contact/_manager_process.py` | smooth_penalty active set 廃止 |
| `xkep_cae/contact/solver/process.py` | beam param + uzawa param 伝搬 |
| `xkep_cae/core/data.py` | default_strategies() に n_uzawa_max, tol_uzawa 追加 |

## 2. HEX8 連続体要素ジグ基盤

### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/elements/_hex8.py` | 8節点線形六面体要素（3DOF/node + 6DOF 埋め込み） |
| `xkep_cae/elements/_hex8_assembler.py` | HEX8 メッシュアセンブラ |
| `xkep_cae/elements/_mixed_assembler.py` | 梁+HEX8 統合アセンブラ |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `ThreePointBendContactJigProcess` 追加 |
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | C3 紐付けテスト追加（xfail） |

### HEX8 要素の検証

```
Ke shape: (24, 24)
symmetric: True
positive semidefinite (min eig): ~0 (浮動小数点誤差)
rigid body force norm: ~1e-11 (ゼロ)
rotation DOF stiffness: 0 (6DOF埋め込み正常)
```

## テスト結果

```
13 passed (既存三点曲げ直接変位制御)
24 passed (contact_force strategy)
契約違反: 0件
条例違反: 0件
ruff check: All checks passed
ruff format: already formatted
```

## TODO

- [ ] **HEX8 接触ジグ NR 収束問題**: smooth_penalty の R=f_int+f_c 符号規約で、ゼロ接線時に
      NR 補正がワイヤをジグ方向に動かし貫入増大→発散。
      根本対策: 接触接線の正値近似、または接触力の符号規約修正が必要。
- [ ] smooth_penalty の INACTIVE スキップ廃止が既存梁–梁接触テストに影響ないか広範テスト
- [ ] 動的三点曲げ（質量行列 + 時間積分）の実装
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase2 xfail 解消
- [ ] numerical_tests の slow テスト復旧

## 設計上の懸念

- **R=f_int+f_c の符号規約**: 接触力 f_c = p_n * g_shape は bodies を分離する方向。
  これを f_int に加算すると、NR 補正が bodies を接近させる方向に働く。
  梁–梁接触では両方の body が動くため相殺するが、梁–剛体接触では片側のみ動き発散する。
  旧モノリシックソルバーではこの問題が表面化しなかった理由の調査が必要。
- **auto k_pen と接触ジグのミスマッチ**: auto k_pen（要素局所剛性ベース、~1593 N/mm）は
  梁–梁接触向け設計。梁–剛体ジグでは梁グローバル剛性（~7.54 N/mm）と 200倍ミスマッチ。

---
