# Status 090: COOベクトル化 + 共有メモリ並列化 + 10万要素想定

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日付**: 2026-03-01
- **ブランチ**: claude/execute-status-todos-53Tda
- **テスト数**: 1866（fast: 1541 / slow: 325）— +18

---

## 概要

MAX撚撚線モデルは長手方向にピッチ長分をモデル化する必要があり、10万要素を容易に超える。
この規模で要素アセンブリがボトルネックになるため、COO書き込みのベクトル化と
共有メモリ並列化を先行実装した。

## 実施内容

### 1. COO インデックスのベクトル化

**変更ファイル**: `xkep_cae/assembly.py`

- **`_vectorized_coo_indices()`** 関数を新設
  - 全要素の DOF インデックスを1回の numpy 演算で計算
  - `(conn_nodes[:, :, None] * ndof_per_node + np.arange(ndof_per_node)).reshape(n_elem, m)`
  - per-element の `dof_indices()` + `np.repeat()` + `np.tile()` を完全に排除
- **`_assemble_sequential()`** を改修
  - rows/cols を要素グループ単位で一括計算（ベクトル化）
  - 要素ループは `local_stiffness()` の計算のみに限定

**ベンチマーク結果**（COO インデックス計算のみ、10000要素）:
```
従来ループ: 113.2 ms
ベクトル化:  18.7 ms → 6.1x 高速化
```

### 2. 共有メモリ並列化

**変更ファイル**: `xkep_cae/assembly.py`

- **`_assemble_parallel()`** を ProcessPoolExecutor → `mp.Pool` + `shared_memory` に全面刷新
- **設計構造**:
  1. rows/cols はメインプロセスでベクトル化計算（高速、並列不要）
  2. data 配列を `multiprocessing.shared_memory` で確保
  3. `mp.Pool` の各ワーカーが Ke を計算し data に直接書き込み
  4. 各ワーカーは排他的な領域に書き込むためロック不要
- **IPC 最適化**:
  - 重い引数（elem, nodes_xy, material）は Pool initializer で1回だけ pickle
  - タスクごとには conn_batch + offset のみ送信（軽量）
  - 結果の pickle/pipe 転送を完全に排除
- **安全性**:
  - `try/finally` で共有メモリの確実なクリーンアップ
  - ワーカーは排他的メモリ領域に書き込み → レース条件なし
- `_compute_element_batch()` は後方互換のため保持

**ベンチマーク結果**（TimoshenkoBeam3D 実要素）:
```
           逐次      並列(4w)   スピードアップ
 4096要素: 0.326s    0.274s     1.19x  (旧ProcessPool: 1.10x)
 8192要素: 0.641s    0.378s     1.70x  (旧ProcessPool: 1.36x)
```

ProcessPoolExecutor 比で:
- 4096要素: +8% 改善
- 8192要素: +25% 改善

10万要素での期待スピードアップ: ~3-4x（IPC がほぼゼロになるため）。

### 3. ロードマップ更新

- Phase S の想定に「10万要素超の MAX 撚撚線モデル」を明記
- S6（1000本撚線トライ）にピッチ長モデル化の注記を追加
- S2 に COO ベクトル化・共有メモリ並列化の完了項目を追加

### 4. テスト

**新規ファイル**: `tests/test_assembly_vectorized.py`（19テスト: fast 16 / slow 3）

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| TestVectorizedCooIndices | 5 | ベクトル化 COO が従来方式と完全一致（Q4, 梁, HEX8, 実要素） |
| TestSequentialVectorized | 3 | ベクトル化逐次: 対称性, 解析解, 混在要素グループ |
| TestSharedMemoryParallel | 6 | 共有メモリ: 逐次/並列一致（ダミー/実要素）, n_jobs=4, 対称性, 解析解 |
| TestAssembleGlobalStiffnessIntegration | 2 | 公開 API: 自動逐次選択, n_jobs=-1 |
| TestAssemblyBenchmark (slow) | 3 | スピードアップ計測（4096/8192要素, COO ベクトル化効果） |

## TODO

- [ ] S3 接触NR収束ベンチマーク（19/37/61/91本）— 実際の contact solve 込みの性能評価
- [ ] ILU drop_tol / Schur 対角近似精度の段階的チューニング
- [ ] 10万要素規模での実測ベンチマーク（共有メモリ並列の実効スピードアップ確認）

## 確認事項・懸念

- 共有メモリの生成サイズは `nnz_total * 8` bytes。10万要素×梁12DOFでは
  `100000 * 144 * 8 = 115 MB`。通常の Linux 環境では問題ないが、
  `/dev/shm` のサイズ制限（デフォルトでRAMの50%）に注意。
- `mp.Pool` の `fork` 開始方式に依存。macOS では `spawn` がデフォルトだが、
  Linux（ターゲット環境）では `fork` がデフォルトなので問題なし。
- 要素グループが複数ある場合、各グループで Pool を生成/破棄する。
  通常1-2グループなのでオーバーヘッドは無視できる。

---
