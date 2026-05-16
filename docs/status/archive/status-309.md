# status-309: K_c/K_stアセンブリベクトル化 + broadphase大規模ベンチマーク

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-08
- **ブランチ**: `claude/check-status-todos-7ykWl`
- **テスト数**: 442+20+14+6 passed（バッチStJacobian 6件追加）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-308のTODO「KD-tree broadphaseの大規模ベンチマーク」と「K_c/K_stアセンブリベクトル化」を実施。broadphaseの1000本撚線ベンチマークを追加し、K_stアセンブリのペアごとPython forループを完全ベクトル化。

---

## 実施内容

### 1. KD-tree broadphase大規模ベンチマーク

1000本撚線（32,000セグメント）までのスケーリング性能を計測するTestBroadphaseBenchmarkを追加。ヘリカル配置でリアルな接触ペア密度を再現。

| 規模 | セグメント数 | ペア数 | 時間 |
|------|-------------|--------|------|
| 7本 | 336 | 2,835 | 0.002s |
| 19本 | 912 | 12,124 | 0.010s |
| 61本 | 2,928 | 60,013 | 0.059s |
| 127本 | 6,096 | 172,479 | 0.204s |
| 271本 | 8,672 | 538,106 | 0.617s |
| 547本 | 17,504 | 1,747,142 | 2.027s |
| 1000本 | 32,000 | 4,956,910 | 5.640s |

**考察**: 1000本で5.6秒。ペア数500万は`r=2*max_half_diag`のグローバル検索半径に起因する偽陽性が多い。narrowphaseで大半が除外されるため、broadphase自体の高速化よりnarrowphaseの高速化が効果的。将来的に分割KD-tree（セグメント長でグループ分け）による偽陽性削減が有効。

### 2. バッチ版StJacobian実装（線形+Hermite）

`_batch_st_jacobian_linear` / `_batch_st_jacobian_hermite` を `_st_jacobian.py` に追加。

| 項目 | 旧実装（スカラー） | 新実装（バッチ） |
|------|-------------------|-----------------|
| 処理単位 | 1ペアずつ | 全ペア一括 |
| 高速パス | なし | w_s≈1, w_t≈1, 非特異 → NumPy一括 |
| 低速パス | 全ペア | エッジケースのみスカラーフォールバック |
| 出力 | (12,), (12,) | (N,12), (N,12) |

テスト6件追加: スカラー版との一致検証（線形/Hermite、単一/複数ペア、境界フォールバック）

### 3. K_stアセンブリベクトル化

`ContactForceStStiffnessProcess.process()` を `_process_batch()` に置換。

**旧実装の問題点**:
- ペアごとのPython forループ（`for pair in inp.pairs:`）
- 各ペアで `_add_kst_contact_to_coo()` を呼び出し
- 内部に4重ネストforループ（4ノード×3次元×4ノード×3次元）

**新実装**:
1. ペアデータをNumPy配列に一括抽出
2. バッチStJacobianで全ペアの ds_du/dt_du を一括計算
3. 形状関数係数・微分をバッチ計算（`_batch_hermite_corrected_coeffs`活用）
4. dn/ds, dn/dt をバッチ計算（`np.einsum("nij,nj->ni", P_perp, dpA)`）
5. df_ds, df_dt をバッチ計算
6. K_st_local = `np.einsum("ni,nj->nij", df_ds, ds_du)` で12×12行列を一括生成
7. COO配列をバッチ構築（K_mat+K_geoと同じパターン）

---

## 変更ファイル

- `xkep_cae/contact/tests/test_broadphase.py`: TestBroadphaseBenchmark追加（7テスト）
- `xkep_cae/contact/geometry/_st_jacobian.py`: `_batch_st_jacobian_linear/hermite`, `_smooth_clip_deriv_batch` 追加
- `xkep_cae/contact/geometry/tests/test_st_jacobian.py`: バッチStJacobianテスト6件追加
- `xkep_cae/contact/contact_force/strategy.py`: `_process_batch()` 実装、未使用import削除

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-7ykWl

# バッチStJacobianテスト
python -m pytest xkep_cae/contact/geometry/tests/test_st_jacobian.py -v

# broadphaseベンチマーク
python -m pytest xkep_cae/contact/tests/test_broadphase.py::TestBroadphaseBenchmark -v -s -m slow

# 接触力テスト
python -m pytest xkep_cae/contact/contact_force/tests/ -v

# 全体テスト
python -m pytest xkep_cae/ -v -k "not slow"

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python contracts/validate_process_contracts.py
```

---

## TODO

- [ ] K_stベクトル化の実効性能測定（slowテスト実行で旧版比較）
- [ ] Hermite dpA/dpBのバッチ計算化（現在は`_hermite_deriv_scalar`のforループ残存）
- [ ] 摩擦K_stアセンブリベクトル化（`_assemble_friction_st_stiffness`にも同パターン適用）
- [ ] K_c adj_node_mapのdict lookup ベクトル化
- [ ] スパース求解高速化（高速化フェーズ第3弾）
- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行

---

## 次の担当者向け

### 重要ポイント

1. **K_stアセンブリがベクトル化済み**: `ContactForceStStiffnessProcess.process()` が `_process_batch()` を呼ぶ。全ペアをNumPy一括処理。旧`_add_kst_contact_to_coo`はモジュール内に残存するがデッドコード
2. **バッチStJacobianの高速パス/低速パス**: 大半のペアは高速パス（w_s≈1, w_t≈1, 非特異）に入る。クランプ境界や特異ケースのみスカラー版にフォールバック
3. **Hermite dpA/dpBのforループ残存**: `_hermite_deriv_scalar` のスカラー呼び出しがn_actペアのforループ内に残っている。次のベクトル化候補
4. **broadphaseのペア数爆発**: 1000本で500万ペア。`2*max_half_diag`のグローバル検索半径が原因。分割KD-treeで改善可能

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: broadphaseベンチマークをteeでログ保存
- [x] **再現手順記載**: コマンド列を明記
- [x] **回帰なし**: 既存テスト全合格、契約違反0件
