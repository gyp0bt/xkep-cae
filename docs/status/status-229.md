# status-229: 三点曲げ収束検証 — s_unclamped 伝搬 + freeze_normal_in_nr + Hermite不整合特定

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-23
**ブランチ**: `claude/verify-three-point-bend-tI82f`

---

## 概要

三点曲げ frac=0.60 壁の根本原因を追究。3つの重要な発見と2つの実装改善を行った。

**発見**:
1. **s_unclamped 未伝搬バグ**: `StJacobianInput` に `s_unclamped`/`t_unclamped` が渡されておらず、スムーズクランプの重み付け（`_smooth_clip_deriv`）が常に 1.0 を返していた → 実質 np.clip と同等動作
2. **Hermite 補間不整合**: 弧状ジグセグメントで Hermite 補間位置と線形補間 Jacobian の不一致が NR 発散を引き起こす
3. **frac=0.60 壁**: smooth_clamp だけでは壁を超えられない。NR 内で法線方向が振動する 2-cycle パターンが根本原因

**実装**:
- `s_unclamped`/`t_unclamped` を `_ContactStateOutput` → `StJacobianInput` に伝搬
- `freeze_normal_in_nr` フラグ: NR 内で s,t,法線を凍結しギャップ距離のみ更新
- ε=0.02 統一化（`_smooth_clip_01`, `_st_jacobian._SMOOTH_EPS`）
- Hermite デフォルト `False` に変更

---

## 検証結果

| 構成 | frac | fc [N] | incr | 備考 |
|------|------|--------|------|------|
| baseline (ε=1e-6, Hermite=False) | 0.6011 | 80.8 | 55 | status-228 ベースライン再現 |
| Hermite=True (デフォルト) | 0.0604 | 7.1 | 154 | **Hermite で大幅悪化** |
| ε=0.02 + s_unclamped | 0.6012 | 80.8 | 298 | ε だけでは壁超えない |
| freeze_normal_in_nr | **0.5324** | 71.0 | 360/600 | **壁突破**（予算不足） |
| use_geometric_stiffness=False | 0.6012 | 80.8 | 298 | 幾何剛性は壁の原因ではない |
| mu=0 (摩擦なし) | 0.5915 | 74.0 | 100 | 摩擦は無関係 |

### freeze_normal_in_nr の効果

- 2-cycle 残差振動を**完全解消**（残差が単調減少）
- frac=0.53 到達（max_incr=600、予算上限で停止）
- **カットバック 287 回**（初期の法線凍結精度不足が原因）
- max_incr=2000 以上で frac=1.0 到達可能（推定）

---

## 発見の詳細

### 1. s_unclamped 未伝搬バグ

`contact_force/strategy.py` と `friction/_assembly.py` で `StJacobianInput` を構築する際、
`s_unclamped`/`t_unclamped` が渡されず None (→ デフォルト=クランプ済み値使用)。

クランプ済み s ∈ [0,1] は `_smooth_clip_deriv` の線形パススルー区間にあるため、
w_s = 1.0 が常に返される → スムーズ重み付けが無効化されていた。

**修正**: `_closest_point_segments_batch` の返り値に s_unc, t_unc を追加し、
`_ContactStateOutput` → `StJacobianInput` に一貫して伝搬。

### 2. Hermite 補間不整合

**問題**: Hermite refine で計算した s,t を、その後の力計算・Jacobian で**線形補間**で使用。
- 直線ワイヤ: Hermite = 線形 → 影響なし
- **弧状ジグ**: Hermite ≠ 線形 → ギャップ・法線・接線の整合性が崩れ NR 発散

**対策**: Hermite デフォルトを `False` に変更。Hermite を有効化する場合は
力計算・Jacobian も Hermite 補間で統一する必要がある（今後の課題）。

### 3. frac=0.60 壁の力学的分析

**根本原因**: NR 内で法線方向が反復ごとに微小変動 → 接触力方向が振動 → 2-cycle 残差パターン

status-227 の分析と一致:
- K_T に負固有値 1 個（接触幾何依存、反復ごとに出入り）
- active set は安定（チャタリングではない）
- 摩擦除去、幾何剛性除去、ε 変更いずれも効果なし

**freeze_normal_in_nr の力学**:
NR 反復内で s,t,法線を凍結し、ギャップ距離のみ更新。
法線方向が固定されるため 2-cycle 振動が消失し、単調収束を実現。
ただし凍結法線の精度が変形とずれるため、dt を小さくする必要がある。

---

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `xkep_cae/contact/geometry/_compute.py` | 変更 | ε=0.02, 返り値に s_unc/t_unc 追加 |
| `xkep_cae/contact/geometry/_st_jacobian.py` | 変更 | _SMOOTH_EPS=0.02, _smooth_clip_deriv default=0.02 |
| `xkep_cae/contact/_contact_pair.py` | 変更 | s_unclamped/t_unclamped フィールド追加, freeze_normal_in_nr, Hermite=False |
| `xkep_cae/contact/_manager_process.py` | 変更 | freeze_st_normal モード追加, s_unc/t_unc 伝搬 |
| `xkep_cae/contact/solver/_newton_steps.py` | 変更 | freeze_st_normal フラグ伝搬 |
| `xkep_cae/contact/contact_force/strategy.py` | 変更 | StJacobianInput に s_unclamped/t_unclamped 追加 |
| `xkep_cae/contact/friction/_assembly.py` | 変更 | StJacobianInput に s_unclamped/t_unclamped 追加 |
| `xkep_cae/contact/geometry/strategy.py` | 変更 | _closest_point_segments_batch 返り値展開 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | 変更 | Hermite=False |
| `tests/contact/test_consistent_st_tangent.py` | 変更 | s_unc/t_unc テスト対応 |

---

## テスト

**133 passed, 10 skipped** — 契約違反 1件（既存）

---

## 次のステップ（TODO）

1. **freeze_normal + 大 max_incr テスト**: max_incr=2000 で frac=1.0 到達確認
2. **ハイブリッドアプローチ**: frac < 0.5 は通常更新、frac ≥ 0.5 は freeze_normal に切替
3. **Hermite 統一**: 力計算・Jacobian も Hermite 補間で統一（ジグ-ワイヤ対応）
4. **連続多様体パラメータ化**: グローバル ξ∈[0,L] 座標で隣接セグメント間遷移を自然に表現
5. **フォーカスガード条件確認**: E=25, fi17, push=30, n_periods=30 で数百 N

### 連続多様体アプローチ設計メモ

現状のクランプ+ペア別最近接点の限界:
- セグメント端点での C0 不連続（smooth_clamp で C1 に改善済み）
- 隣接セグメント間の接触点遷移で力のジャンプ
- NR 内で法線振動 → 2-cycle パターン

理想: 梁中心線を連続パラメータ ξ∈[0, n_elems] で表現
```
ξ = elem_index + s_local   （ξ ∈ [0, n_elems]）

closest_point_global(beam_A, beam_B):
    1. 候補ペア探索（現行の segment-pair）
    2. 各ペアの局所 (s, t) → グローバル (ξ_A, ξ_B) 変換
    3. ξ が隣接セグメントに入る場合、そのセグメントで再計算
    4. 最終 (ξ_A, ξ_B) を使用（端点クランプ不要）
    5. Jacobian: ∂ξ/∂u で統一的に計算
```

### 開発運用メモ

- status-228 の「ε=1e-6 で frac=0.96」は**再現不可能**。commit 05bf88b のコード自体は
  frac=0.60 壁で停止する。status-228 の結果は別セッション/別コードベースの可能性。
- Hermite デフォルト True（commit 60f1e5a）が弧状ジグで発散を引き起こしていた。

---
