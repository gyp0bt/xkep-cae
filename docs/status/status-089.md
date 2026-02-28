# Status 089: Broadphase強化 + 中点距離プリスクリーニング + S3ベンチマーク基盤 + ProcessPoolExecutor切替

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日付**: 2026-02-28
- **ブランチ**: claude/execute-status-todos-53Tda
- **テスト数**: 1848（fast: 1525 / slow: 323）— +26

---

## 概要

status-087/088 の TODO を消化。Broadphase グリッドビニングの高速化、
Broadphase→Narrowphase 間の中点距離プリスクリーニング、
Phase S3 スケーラビリティベンチマーク基盤（7/19/37/61/91本）を実装。
要素並列化を ThreadPoolExecutor → ProcessPoolExecutor に切替（GIL回避）。

## 実施内容

### 1. Broadphase グリッドビニング強化

**変更ファイル**: `xkep_cae/contact/broadphase.py`

- **セルサイズ推定のベクトル化**: Python ループ → `np.max(hi_all - lo_all, axis=1)` 一括計算
- **セルインデックス一括計算**: per-segment `np.floor` を排除。全セグメントの `ilo_all`/`ihi_all` を1回の `np.floor(lo_all * inv_cell)` で算出
- **バッチ AABB 重複判定**: per-pair `_aabb_overlap()` 呼び出し → 全ペア一括 numpy 演算に置換
  - `np.all(lo_all[pi] <= hi_all[pj], axis=1) & np.all(lo_all[pj] <= hi_all[pi], axis=1)`
- **未使用の `segment_cells` リスト**を削除
- 既存 21 broadphase テスト全通過

### 2. 中点距離プリスクリーニング

**変更ファイル**: `xkep_cae/contact/pair.py`

- `ContactConfig.midpoint_prescreening: bool = True`（デフォルト有効）
- `ContactConfig.prescreening_margin: float = 0.0`（0=自動推定: `mean(r_a + r_b) * 0.5`）
- `detect_candidates()` にベクトル化された中点間距離フィルタを追加
  - 中点間距離 - 半長の和 > 半径和 + マージン のペアを高速除去
  - broadphase AABB（軸整列）より 3D 距離が正確なため、タイトな候補絞り込みが可能
  - 同層除外フィルタとの併用で候補ペアをさらに削減

### 3. Phase S3 スケーラビリティベンチマークテスト

**新規ファイル**: `tests/test_s3_benchmark.py`（26テスト, slowマーカー）

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| TestMeshScaling | 10 | 7/19/37/61/91本メッシュ生成 + 素線配置 |
| TestBroadphaseScaling | 6 | 候補ペア数スケーリング + サブ二次検証 |
| TestMidpointPrescreening | 4 | プリスクリーニング削減効果 + 真の接触保持 |
| TestCombinedFilters | 2 | 同層除外 + プリスクリーニング併用効果 |
| TestParallelSpeedup | 2 | 並列正しさ検証 + スピードアップ計測 |
| TestBroadphasePerformance | 2 | broadphase/detect_candidates 性能計測 |

#### ベンチマーク結果（参考値、CI環境）

**Broadphase スケーリング**:
```
 n_strands  n_elems  n_cands    time_ms
         7       56      511       0.97
        19      152     3206       4.83
        37      296     9214      14.34
        61      488    19887      33.97
```

**フィルタ付き detect_candidates**:
```
 n_strands  n_elems  n_pairs  n_cands    time_ms
         7       56      132      132       2.52
        19      152     1760     1760      16.54
        37      296     5984     5984      53.73
```

**ProcessPoolExecutor 並列アセンブリ（実要素: TimoshenkoBeam3D）**:
```
 n_elems   seq(s)  par4(s)  speedup
    4096   0.3629   0.3293    1.10x
    8192   0.7343   0.5408    1.36x
```
ProcessPoolExecutor はプロセス間通信オーバーヘッドがあるため、~4096要素以上でスピードアップが得られる。
閾値 `_PARALLEL_MIN_ELEMENTS` を 64 → 4096 に変更。

### 4. ProcessPoolExecutor 切替（GIL回避）

**変更ファイル**: `xkep_cae/assembly.py`, `tests/test_s2_parallel.py`, `tests/test_s3_benchmark.py`

- **Thread→Process 切替**: `ThreadPoolExecutor` → `ProcessPoolExecutor` に変更
  - GIL制約により ThreadPoolExecutor ではスピードアップ不可（0.25x-0.43x、逆に遅い）
  - ProcessPoolExecutor はプロセス並列で GIL を完全回避
- **並列化閾値引き上げ**: `_PARALLEL_MIN_ELEMENTS` を 64 → 4096 に変更
  - ProcessPoolExecutor の IPC オーバーヘッドにより ~4000 要素未満では逆に遅くなる
- **ベンチマークテスト刷新**: ダミー要素 → 実要素（TimoshenkoBeam3D）に更新
  - `test_parallel_correctness_real_element`: 逐次/並列の結果一致検証
  - `test_speedup_measurement_real_element[4096]`: 1.10x スピードアップ
  - `test_speedup_measurement_real_element[8192]`: 1.36x スピードアップ
- 既存 13 テスト（test_s2_parallel）全通過

## TODO

- [ ] S3 接触NR収束ベンチマーク（19/37/61/91本）— 実際の contact solve 込みの性能評価
- [ ] ILU drop_tol / Schur 対角近似精度の段階的チューニング

## 確認事項・懸念

- 中点距離プリスクリーニングはデフォルト有効 (`midpoint_prescreening=True`) だが、
  極端に密な撚線配置では真の接触ペアを落とす可能性がある。
  `prescreening_margin` の自動推定値 `mean(r_a + r_b) * 0.5` は保守的な値だが、
  超密パッケージでは手動マージン指定を推奨。
- ベンチマーク結果は CI 環境依存（CPU性能、メモリ帯域に影響される）。
  スピードアップの絶対値よりもスケーリング傾向を重視すべき。

## 開発運用メモ

- **効果的**: status TODO ベースの引き継ぎ。各TODO が明確な実装ターゲット。
- **効果的**: ベクトル化最適化はテスト駆動で安全に進められる（既存テストが回帰検知）。
- **懸念**: broadphase のグリッドビニング内部ループ（3重 Python ループ）は
  numpy ベクトル化が困難（セグメントごとにセル数が可変）。
  Cython/Numba への移行が次の高速化の鍵。

---
