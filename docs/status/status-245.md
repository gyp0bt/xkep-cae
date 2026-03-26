# status-245: Hermite デフォルト ON + n_periods=30 収束検証

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 200+10s | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-244 の TODO 2件を実施:
1. **use_hermite_centerline=True デフォルト化**: frozen-m 解消（status-243）済みのため、Hermite 中心線補間をデフォルト ON に変更
2. **n_periods=30 収束検証**: freeze=F, K_st=ON, dm 補正有りで検証 → frac=0.7360（改善だが未完走）

---

## 変更内容

### 1. Hermite デフォルト ON

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/_contact_pair.py` | `use_hermite_centerline` デフォルト False→True |
| `xkep_cae/contact/geometry/_compute.py` | `_compute_node_counts` の IndexError 修正（connectivity max > n_nodes 対応）|
| `tests/contact/test_consistent_st_tangent.py` | `use_hermite_centerline=False` 明示（線形セグメント K_st 検証用）|
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `test_evaluate_inactive_pair_skipped` → SDI 排除後の挙動に修正 |

### 2. n_periods=30 収束検証結果

| 設定 | frac | incr | 備考 |
|------|------|------|------|
| freeze=F, K_st=ON, Hermite=ON | **0.7360** | 346 | 発散（残差振動 active=2↔3）|
| freeze=T, K_st=OFF, Hermite=ON | 0.0003 | 5 | 壊滅的（初期カットバック地獄）|
| freeze=T, K_st=OFF, Hermite=OFF | 1.0 | 1592 | status-234 ベースライン |

**freeze=F + K_st=ON + Hermite=ON は status-232（frac=0.08）から 0.7360 に大幅改善**。
frozen-m 解消と K_st の ∂p_n/∂s 追加が効いている。
ただし frac=0.74 付近で active 接触点数が 2↔3 で振動し、接線不整合で発散。

### 発散箇所の分析

- frac=0.73 付近で active ペア数が 2↔3 で NR 反復ごとに切り替わる
- 接触チャタリング: gap が 0 近傍の境界ペアが ACTIVE/INACTIVE を反復内で繰り返す
- Hermite の接線感度が高く、node tangent の微小変化が gap 判定を変える
- K_st=ON でも Hermite ∂g/∂u の非局所成分（4ノードペア外）が未対応

---

## 既存テストの修正

### test_evaluate_inactive_pair_skipped → test_evaluate_inactive_pair_with_penetration

SDI 排除（status-233）により、全候補ペアが Huber 評価されるため、
INACTIVE でも gap<0 なら接触力が発生する。テスト期待値を修正。

### _compute_node_counts IndexError

三点曲げジグテストで connectivity の最大ノード番号が n_nodes を超える場合に
IndexError が発生。`max(n_nodes, max(connectivity)+1)` で配列サイズを確保。

---

## TODO

- [ ] **n_periods=30 Hermite ON 完走**: 接触チャタリング対策が必要（active ペア振動の抑制）
- [ ] **Hermite 非局所 ∂g/∂u**: 4ノードペア外の DOF 結合対応（status-243 で指摘済み）
- [ ] **Node tangent 局所化**: 大変形時の接線急変対策（roadmap に記載済み）

---

## 確認事項（次セッションへ）

- Hermite デフォルト ON は n_periods=1〜3 では問題なし（テスト全パス）
- n_periods=30 では Hermite ON は未完走だが、0.08→0.7360 に大幅改善
- freeze=T + Hermite=ON の組合せは壊滅的（0.0003）→ freeze=F が必須
- K_st=ON は Hermite と組み合わせることで効果を発揮

---
