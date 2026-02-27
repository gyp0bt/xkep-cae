# status-077: Phase C6-L1 — Line-to-line Gauss 積分（接触アルゴリズム根本整理開始）

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1760（fast: +28）
- **ブランチ**: claude/improve-contact-algorithm-l0EQl

## 概要

Phase C6: 接触アルゴリズム根本整理の第1段階として、Line-to-line Gauss 積分（C6-L1）を実装。従来の Point-to-Point (PtP) 接触を拡張し、セグメントに沿った Gauss 積分で接触力・接線剛性を評価する機能を追加した。

設計仕様: [contact-algorithm-overhaul-c6.md](../contact/contact-algorithm-overhaul-c6.md) §3

## 背景・動機

現行の PtP 接触では、セグメントペアごとに1点の最近接点 (s*, t*) のみで接触力を評価する。準平行な梁（撚線の同層梁など）では接触帯が長くなり、1点評価では不正確になる。Line-to-line Gauss 積分は接触力の空間分布を捉え、精度を向上させる。

参考文献:
- Meier, Popp, Wall (2016): "A unified approach for beam-to-beam contact"
- Meier, Popp, Wall (2016): "A finite element approach for the line-to-line contact interaction of thin beams"

## 実施内容

### 1. line_contact.py 新規作成（コアモジュール）

**新規モジュール**: `xkep_cae/contact/line_contact.py`

| 関数 | 説明 |
|------|------|
| `gauss_legendre_01(n)` | [0,1] 区間の Gauss-Legendre 積分点・重み |
| `project_point_to_segment(p, x0, x1)` | 点→セグメント最近接パラメータ t ∈ [0,1] |
| `auto_select_n_gauss(xA0, xA1, xB0, xB1)` | セグメント間角度ベースの Gauss 点数自動選択 |
| `_build_shape_vector_at_gp(s, t, n)` | Gauss 点での 12DOF 形状ベクトル構築 |
| `_geometric_stiffness_at_gp(s, t, n, p_n, dist)` | Gauss 点での幾何剛性行列 |
| `compute_line_contact_force_local(pair, ...)` | Line contact 力の Gauss 積分（12DOF 局所ベクトル） |
| `compute_line_contact_stiffness_local(pair, ...)` | Line contact 剛性の Gauss 積分（12×12 局所行列） |
| `compute_line_contact_gap_at_gp(pair, ...)` | 各 Gauss 点のギャップ値（診断用） |

**Gauss 点数自動選択ロジック**:

| 角度 θ | n_gauss | 理由 |
|--------|---------|------|
| θ > 30° | 2 | PtP と同等精度、追加コスト最小 |
| 10° < θ < 30° | 3 | 中間領域、標準 |
| θ < 10° | 5 | 長い接触帯、高精度必須 |

### 2. ContactConfig 拡張

`xkep_cae/contact/pair.py` の `ContactConfig` に3フィールド追加:

```python
line_contact: bool = False    # Line-to-line Gauss 積分の有効化
n_gauss: int = 3              # Gauss 積分点数（2-5）
n_gauss_auto: bool = False    # セグメント角度に基づく自動選択
```

デフォルト `line_contact=False` で後方互換を保持。

### 3. assembly.py 統合

`compute_contact_force` / `compute_contact_stiffness` に `node_coords` パラメータを追加。`line_contact=True` かつ `node_coords` が指定された場合、Gauss 積分パスにディスパッチ。摩擦力は PtP 代表点で評価（line contact でも同様）。

### 4. solver_hooks.py 統合

`newton_raphson_with_contact` / `newton_raphson_block_contact` の全6箇所で、`line_contact=True` の場合に変形後座標を計算して assembly に渡すよう修正。

### 5. テスト（+28テスト、全 fast）

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestGaussLegendre01 | 4 | 重み合計、点範囲、点数、多項式精度 |
| TestProjectPointToSegment | 4 | 中点、クランプ始点/終点、縮退 |
| TestAutoSelectNGauss | 4 | 直交、中間角、準平行、縮退 |
| TestLineContactForce | 5 | 作用反作用、方向、離間、正力、AL乗数 |
| TestLineContactStiffness | 3 | 対称性、半正値性、離間ゼロ |
| TestLineVsPtPComparison | 1 | 平行梁で PtP と一致確認 |
| TestGaussPointConvergence | 1 | n_gauss 増加で収束 |
| TestLineContactGapDiag | 2 | 均一/変動ギャップ |
| TestAssemblyLineContactIntegration | 4 | PtP後方互換、line contact mode、剛性、自動選択 |

## ファイル変更

### 新規
- `xkep_cae/contact/line_contact.py` — Line-to-line Gauss 積分コアモジュール
- `tests/contact/test_line_contact.py` — 28テスト
- `docs/status/status-077.md` — 本ステータス

### 変更
- `xkep_cae/contact/pair.py` — ContactConfig に line_contact/n_gauss/n_gauss_auto 追加
- `xkep_cae/contact/assembly.py` — line contact ディスパッチ追加
- `xkep_cae/contact/solver_hooks.py` — 変形座標パススルー（6箇所）
- `xkep_cae/contact/__init__.py` — line_contact エクスポート追加
- `README.md` — テスト数更新
- `docs/roadmap.md` — C6-L1 チェック + テスト数更新
- `docs/status/status-index.md` — status-077 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 消化済み（status-076 → 077）
- [x] C6-L1: Segment-to-segment Gauss 積分（Line-to-line 接触）

### 未解決（引き継ぎ）
- [ ] **C6-L2: 一貫接線の完全化（∂s/∂u, ∂t/∂u Jacobian）** — 現在の line contact は ∂t/∂u を無視（t_closest をパラメータ固定扱い）。完全一貫接線には ∂(s,t)/∂u Jacobian が必要。
- [ ] **C6-L3: Semi-smooth Newton + NCP 関数** — Outer loop を廃止し、NCP 関数で接触条件を直接組み込む。
- [ ] **C6-L4: ブロック前処理強化（接触 Schur 補集合）**
- [ ] **C6-L5: Mortar 離散化**
- [ ] 摩擦力の line contact 拡張 — 現在は PtP 代表点で評価。Gauss 点ごとの摩擦評価が将来課題。
- [ ] 接触プリスクリーニング GNN Step 2-5
- [ ] k_pen推定ML v2 Step 2-7

### 設計メモ
- Line contact は準平行梁（撚線の同層梁）で精度向上が期待される。大角度交差梁では PtP と同等かそれ以下の精度（Gauss 点が接触帯外に配置される）。
- Gauss 積分はパラメトリック空間 [0,1] 上で実行。重み合計 1.0 のため、lambda_n/k_pen の単位は PtP と同一。
- `auto_select_n_gauss` は cos(θ) ベースで θ > 30° → 2, 10° < θ < 30° → 3, θ < 10° → 5 の3段階。

---
