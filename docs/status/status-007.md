# status-007: Cowper κ(ν) 実装 / Q4 Abaqus比較テスト / Abaqus差異ドキュメント

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-006](./status-006.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/check-status-and-todos-BNs9l`

---

## 実施内容

### 1. Cowper (1966) のν依存せん断補正係数 κ(ν) の実装

Abaqus B21/B22 が使用する Cowper のν依存せん断補正係数を
TimoshenkoBeam2D に実装。

#### 定式化

| 断面形状 | Cowper κ | ν=0 の値 | ν=0.3 の値 |
|---------|---------|---------|-----------|
| 矩形 | 10(1+ν)/(12+11ν) | 5/6 ≈ 0.833 | 0.850 |
| 円形 | 6(1+ν)/(7+6ν) | 6/7 ≈ 0.857 | 0.886 |
| 一般 | 5/6（フォールバック）| 0.833 | 0.833 |

#### 変更点

**`xkep_cae/sections/beam.py`**:
- `BeamSection2D` に `shape` フィールド追加（"rectangle", "circle", "general"）
- `cowper_kappa(nu)` メソッド追加（断面形状に応じたCowper κ計算）
- `rectangle()`, `circle()` ファクトリメソッドが自動的に `shape` を設定

**`xkep_cae/elements/beam_timo2d.py`**:
- `TimoshenkoBeam2D` の `kappa` パラメータが `float | str` を受け付けるように拡張
- `kappa="cowper"` 指定で、材料のνから自動計算
- 後方互換性: デフォルト `kappa=5/6` は変更なし

#### 使用例

```python
sec = BeamSection2D.rectangle(b=10.0, h=10.0)

# 固定κ（従来互換）
beam_fixed = TimoshenkoBeam2D(section=sec, kappa=5.0/6.0)

# Cowper κ（Abaqus準拠）
beam_cowper = TimoshenkoBeam2D(section=sec, kappa="cowper")
```

### 2. Q4要素のAbaqus比較テスト（CPE4I相当の精度検証）

`tests/test_abaqus_comparison.py` を新規作成。

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| `TestCPE4IEquivalentCantilever` | 7 | 片持ち梁 1要素厚さ精度, メッシュ収束, plain Q4比較 |
| `TestCPE4IEquivalentCooksMembrane` | 3 | Cook's membrane（歪み要素の標準ベンチマーク） |
| `TestIncompressibleBending` | 6 | ν=0.49/0.499/0.4999 での体積ロッキング耐性 |

Cook's membrane は平面要素の標準ベンチマーク問題で、
歪み四角形要素の性能を評価する。EAS-4 が plain Q4 に対して
収束速度・精度の両面で優位であることを定量的に検証。

### 3. Abaqus差異ドキュメント作成

`docs/abaqus-differences.md` を新規作成。以下を網羅的に文書化:

- Q4要素の対応表（CPE4 ↔ Quad4, CPE4I ↔ EAS-4, CPE4H ↔ B-bar）
- EAS-4 と CPE4I の理論的差異
- 梁要素の差異（κ, SCF, 剛性行列定式化）
- Abaqus ベンチマーク比較時のチェックリスト
- `*TRANSVERSE SHEAR STIFFNESS` での SCF 無効化手順

---

## テスト結果

```
115 passed, 2 deselected (external), 44 warnings
ruff check: All checks passed!
ruff format: All files formatted
```

| テストファイル | テスト数 | 結果 |
|---------------|---------|------|
| `test_abaqus_inp.py` | 21 | PASSED |
| `test_beam_eb2d.py` | 21 | PASSED |
| `test_beam_timo2d.py` | **25** | **PASSED (+11 新規)** |
| `test_benchmark_shear.py` | 4 | PASSED |
| `test_benchmark_tensile.py` | 4 | PASSED |
| `test_elements_manufactured.py` | 3 | PASSED |
| `test_protocol_assembly.py` | 7 | PASSED |
| `test_quad4_eas.py` | 14 | PASSED |
| **`test_abaqus_comparison.py`** | **16** | **PASSED (新規)** |
| `test_benchmark_cutter_q4tri3.py` | 1 | DESELECTED (external) |
| `test_benchmark_cutter_tri6.py` | 1 | DESELECTED (external) |

### 新規テスト内訳

**`test_beam_timo2d.py` に追加 (+11)**:

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| `TestCowperKappa` | 8 | Cowper κ の値検証（矩形・円形・一般、ν=0/0.3） |
| `TestCowperKappaIntegration` | 3 | kappa="cowper"の統合テスト（解析解比較・固定κとの差・エラー） |

**`test_abaqus_comparison.py` (新規 16)**:

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| `TestCPE4IEquivalentCantilever` | 7 | 片持ち梁: 1要素厚さ精度（ν=0.3/0.4/0.4999）、メッシュ収束 |
| `TestCPE4IEquivalentCooksMembrane` | 3 | Cook's membrane: 歪み要素性能、収束速度比較 |
| `TestIncompressibleBending` | 6 | 非圧縮曲げ: EAS-4 vs plain Q4（ν=0.49/0.499/0.4999）|

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/sections/beam.py` | 拡張 — shape, cowper_kappa() 追加 |
| `xkep_cae/elements/beam_timo2d.py` | 拡張 — kappa="cowper" サポート |
| `tests/test_beam_timo2d.py` | 拡張 — Cowperκテスト追加（+11） |
| `tests/test_abaqus_comparison.py` | **新規** — CPE4I相当精度検証（16テスト） |
| `docs/abaqus-differences.md` | **新規** — Abaqus差異ドキュメント |
| `docs/status/status-007.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — Cowper κ 完了マーク |
| `README.md` | 更新 — ドキュメントリンク追加 |

---

## Abaqusとの既知の差異（現時点）

| カテゴリ | xkep-cae | Abaqus | 差異度 |
|---------|----------|--------|-------|
| Q4要素 | EAS-4（デフォルト） | CPE4I（非適合モード） | 小（同等精度） |
| κ（せん断補正）| 5/6 or Cowper | Cowper（デフォルト） | `kappa="cowper"` で一致 |
| SCF | なし | 0.25（デフォルト） | **大**（Abaqus側で無効化要） |
| 梁剛性行列 | Przemieniecki厳密解 | Hughes低減積分ペナルティ | 小（十分なメッシュで一致） |
| CPE4R相当 | 未実装 | 低減積分+hourglass | — |
| 3D梁 | 未実装 | B31/B32 | Phase 2.3で対応予定 |

---

## TODO（次回以降の作業）

- [ ] Phase 2.3: Timoshenko梁（3D空間）の実装
- [ ] Phase 2.4: 断面モデルの拡張（一般断面）
- [ ] Phase 3: 幾何学的非線形（Newton-Raphson, 共回転定式化）
- [ ] SCF（スレンダネス補償係数）のオプション実装検討
- [ ] Abaqus .inp パーサーへの `*TRANSVERSE SHEAR STIFFNESS` サポート追加

---

## 設計上のメモ

1. **BeamSection2D の shape フィールド**: `field(default="general", repr=False)` で
   後方互換性を保つ。既存コードで `BeamSection2D(A=..., I=...)` として構築した場合、
   `shape="general"` になり `cowper_kappa()` は 5/6 を返す。

2. **kappa の型設計**: `float | str` のユニオン型。将来的に他の補正スキーム
   （例: "Mindlin" 等）を追加する拡張点としても機能する。

3. **Cook's membrane テスト**: 文献上の参照値（δ≈23.9）との直接比較ではなく、
   メッシュ収束の単調性と EAS-4 vs plain Q4 の相対比較で検証。
   これは参照値が平面応力条件のものが多いため。

---

## 参考文献

- Cowper, G.R. (1966) "The shear coefficient in Timoshenko's beam theory", J. Applied Mechanics, 33, 335-340.
- Cook, R.D. et al. (1974) "Concepts and Applications of Finite Element Analysis", Wiley.
- Pian, T.H.H. & Sumihara, K. (1984) "Rational approach for assumed stress finite elements", IJNME, 20, 1685-1695.
