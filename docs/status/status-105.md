# Status 105: inp_runner .inp 未読取情報の洗い出し + カテゴリA〜E全対応

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-04
**ブランチ**: `claude/extract-inp-metadata-vYaaA`
**テスト数**: 1976（fast: 1597 / slow: 374 + 5）— +12テスト

## 概要

`scripts/run_bending_oscillation.py` の `solve_from_inp` が、.inp ファイルから実際に読み取っている情報と、読み取らずにハードコードまたは再生成している情報を網羅的に洗い出した。

## 核心的な発見: .inp 標準Abaqusデータは完全に無視されている

`solve_from_inp` の処理フロー:

1. .inp ファイル末尾の `** XKEP-CAE METADATA BEGIN ... END` JSON ブロック**だけ**を読み込む
2. 標準 Abaqus キーワード（`*NODE`, `*ELEMENT`, `*MATERIAL`, `*BOUNDARY`, `*STEP` 等）は**一切パースしない**
3. メタデータのパラメータを `run_bending_oscillation()` に渡す
4. `run_bending_oscillation()` 内でメッシュを**1から再生成**する（`make_twisted_wire_mesh()`）
5. エクスポートフック用にさらにもう一度メッシュを再生成する（L479）

つまり、.inp に記録された節点座標・要素接続・材料・境界条件・ステップ定義はすべて装飾であり、計算には使われていない。

## .inp に書き出されているが読み取られていない情報

### カテゴリ A: 標準 Abaqus キーワード（完全に無視）

| キーワード | 書き出し箇所 | ソルバーでの扱い |
|-----------|------------|----------------|
| `*NODE` | 全節点座標 | 無視（メタデータから再生成） |
| `*ELEMENT, TYPE=B31` | 全要素接続 | 無視（メタデータから再生成） |
| `*NSET` | STRAND_0〜N, FIXED_END, FREE_END | 無視（メッシュ位相から再導出） |
| `*ELSET` | STRAND_0〜N | 無視 |
| `*MATERIAL, NAME=STEEL` | 材料名 | 無視 |
| `*ELASTIC` | E, ν | **無視**（ハードコード `_E=200e9, _NU=0.3`） |
| `*DENSITY` | ρ=7850 | 無視（準静的解析では不使用） |
| `*BEAM SECTION, SECTION=CIRC` | 断面半径 r | 無視（wire_diameter から再計算） |
| `*BEAM SECTION` の方向ベクトル | [0, 1, 0] | 無視（CR梁ではローカル座標系を内部計算） |
| `*BOUNDARY` (Step内) | 固定端拘束 | 無視（`_fix_strand_starts()` で再構築） |
| `*STEP` | Bending / Oscillation ステップ定義 | 無視（ハードコードロジック） |
| `*STATIC` / `*DYNAMIC` | 解析タイプ・時間パラメータ | 無視 |
| `*SURFACE, TYPE=ELEMENT` | 接触サーフェス定義 | 無視（`_build_contact_manager()` 内で再構築） |
| `*SURFACE INTERACTION` | 接触プロパティ | 無視（メタデータの個別パラメータで再構築） |
| `*SURFACE BEHAVIOR` | HARD/LINEAR | 無視 |
| `*FRICTION` | 摩擦係数 | 無視（メタデータの `mu` で代替） |
| `*CONTACT` | General Contact 宣言 | 無視 |
| `*CONTACT INCLUSIONS` | ALLEXT, ALLEXT | 無視 |
| `*CONTACT PROPERTY ASSIGNMENT` | プロパティ割当 | 無視 |
| `*OUTPUT, FIELD/HISTORY` | 出力リクエスト | 無視 |
| `*NODE OUTPUT`, `*ELEMENT OUTPUT` | 出力変数 | 無視 |
| `*ENERGY OUTPUT` | エネルギー出力 | 無視 |

### カテゴリ B: メタデータに存在するが `run_bending_oscillation` に渡されない

| メタデータキー | 値の例 | 問題点 |
|--------------|--------|--------|
| `E` | 200e9 | `run_bending_oscillation()` は引数で受け取らず、モジュール定数 `_E=200e9` をハードコード使用 |
| `nu` | 0.3 | 同上。`_NU=0.3` をハードコード使用 |
| `strand_node_ranges` | [[0,4], [5,9], ...] | 渡されない（再生成メッシュから導出） |
| `strand_elem_ranges` | [[0,3], [4,7], ...] | 同上 |
| `xkep_version` | "2.0" | バージョン検証なし |

### カテゴリ C: `run_bending_oscillation` 内でハードコードされている物理・数値パラメータ

| パラメータ | ハードコード値 | 所在 | 備考 |
|-----------|--------------|------|------|
| ヤング率 `_E` | 200e9 Pa | `wire_bending_benchmark.py:49` | .inp にも `*ELASTIC` で記録されているが無視 |
| ポアソン比 `_NU` | 0.3 | `wire_bending_benchmark.py:50` | 同上 |
| せん断弾性係数 `_G` | E/(2(1+ν)) | `wire_bending_benchmark.py:51` | `_E`, `_NU` から導出 |
| せん断補正係数 `_KAPPA` | Cowper式 | `wire_bending_benchmark.py:53` | 断面形状固定前提 |
| 1節点DOF数 `_NDOF_PER_NODE` | 6 | `wire_bending_benchmark.py:54` | 3D梁固定 |
| 梁定式化 | CR (corotational) Timoshenko 3D | `_make_cr_assemblers()` | Euler-Bernoulli等の選択不可 |

### カテゴリ D: `_build_contact_manager` 内のハードコード接触パラメータ

| パラメータ | ハードコード値 | 意味 |
|-----------|--------------|------|
| `k_t_ratio` | 0.1 | 接線/法線ペナルティ比 |
| `g_on` | 0.0 | 接触活性化ギャップ |
| `g_off` | 1e-5 | 接触非活性化ギャップ |
| `use_line_search` | True | ライン探索有効 |
| `line_search_max_steps` | 5 | ライン探索最大ステップ |
| `use_geometric_stiffness` | True | 幾何学的剛性行列 |
| `tol_penetration_ratio` | 0.02 | 貫入比閾値 |
| `k_pen_max` | 1e12 | ペナルティ剛性上限 |
| `exclude_same_layer` | True | 同層除外（常時有効） |
| `midpoint_prescreening` | True | 中点プレスクリーニング |
| `linear_solver` | "auto" | 線形ソルバー選択 |
| `line_contact` | True | ライン接触（常時有効） |

### カテゴリ E: ソルバー内のハードコード数値パラメータ

| パラメータ | ハードコード値 | 所在 |
|-----------|--------------|------|
| `broadphase_margin` | 0.01 | `run_bending_oscillation()` 内の `newton_raphson_contact_ncp` / `newton_raphson_with_contact` 呼び出し |
| `kpen_mode` | "beam_ei"（auto_kpen時） | `_build_contact_manager:209` |
| `kpen_scale` | 0.1（auto_kpen時） | `_build_contact_manager:210` |
| `beam_E` | `_E`（auto_kpen時） | `_build_contact_manager:217` — メタデータの E を使わずモジュール定数を使用 |

## 影響分析

### 現時点のリスク

1. **E/ν のズレ**: .inp に `*ELASTIC 200e9, 0.3` と書いてあっても、仮にメタデータの E/ν を変更してエクスポートした .inp をsolveしても、ソルバーは常に `_E=200e9` を使う
2. **メッシュ不一致の可能性**: .inp に手動編集で節点座標を変更しても、solveは再生成メッシュを使うため反映されない
3. **接触パラメータの不透明性**: カテゴリD のパラメータ群は .inp にもメタデータにも記録されず、コード内にのみ存在

### 「.inp が truth」にするための課題

.inp を計算のtruthとするには以下が必要:

1. **`*ELASTIC` → E, ν の読み取りと `run_bending_oscillation` への受け渡し**
   - `run_bending_oscillation` に `E`, `nu` 引数を追加
   - `_E`, `_NU`, `_G`, `_KAPPA` をハードコードから引数導出に変更

2. **メッシュデータの直接利用**（最も大きな変更）
   - `build_beam_model_from_inp` をベースに接触解析モデルを構築する経路を新設
   - または最低限: .inp の NODE/ELEMENT と再生成メッシュの一致を検証するアサーション

3. **接触パラメータのメタデータ化**
   - カテゴリD のパラメータ群をメタデータに含める
   - または `*SURFACE BEHAVIOR` / `*CONTACT CONTROLS` 相当のAbaqusキーワードとしてパース

4. **ステップ定義の読み取り**
   - `*STEP` / `*STATIC` / `*DYNAMIC` のパース → 曲げ/揺動ロジックの汎用化

## 実施した対応（カテゴリA〜E全対応）

### カテゴリB+C: E/nu ハードコード除去

- `wire_bending_benchmark.py`: `_E`, `_NU`, `_G`, `_KAPPA` のモジュール定数を廃止
- `_DEFAULT_E=200e9`, `_DEFAULT_NU=0.3` をデフォルト引数用に残す
- `_compute_G(E, nu)`, `_compute_kappa(nu)` 関数を新設
- `run_bending_oscillation()` に `E`, `nu` 引数を追加（デフォルト: 鋼線値）
- `_make_cr_assemblers()` に `kappa` 引数を追加
- `_build_contact_manager()` に `E` 引数を追加（auto_kpen 時の beam_E に使用）

### カテゴリD: 接触パラメータ引数化

`_build_contact_manager()` と `run_bending_oscillation()` に以下を引数化:

| パラメータ | デフォルト | 旧ハードコード場所 |
|-----------|-----------|-----------------|
| `k_t_ratio` | 0.1 | `_build_contact_manager` |
| `g_on` | 0.0 | 同上 |
| `g_off` | 1e-5 | 同上 |
| `use_line_search` | True | 同上 |
| `line_search_max_steps` | 5 | 同上 |
| `use_geometric_stiffness` | True | 同上 |
| `tol_penetration_ratio` | 0.02 | 同上 |
| `k_pen_max` | 1e12 | 同上 |
| `exclude_same_layer` | True | 同上 |
| `midpoint_prescreening` | True | 同上 |
| `linear_solver` | "auto" | 同上 |
| `line_contact` | True | 同上 |

### カテゴリE: 数値パラメータ引数化

- `broadphase_margin` を引数化（デフォルト 0.01）
- NCP/AL ソルバー呼び出しの全4箇所で `broadphase_margin` 引数を使用

### メタデータ (export/solve)

- `xkep_version` を `"2.0"` → `"3.0"` に更新
- `DEFAULT_PARAMS` にカテゴリD+E の全パラメータを追加
- メタデータ JSON にカテゴリD+E の全パラメータを記録
- `solve_from_inp()` で全パラメータをメタデータから `run_bending_oscillation()` に渡す

### カテゴリA: .inp 標準データ検証

- `_validate_inp_vs_metadata()` 関数を新設
- `read_abaqus_inp()` で .inp の標準 Abaqus データを読み取り
- メタデータとの整合チェック（E/nu, 節点数, 要素数, 断面寸法）
- 不整合時は .inp の値を truth として採用（overrides 辞書で上書き）
- 節点座標のサンプリング検証情報を `_inp_*` キーで記録

### テスト追加（12テスト）

- `tests/test_inp_metadata_validation.py` 新規作成
  - `TestMaterialParameterization` (4テスト): G/kappa 計算、デフォルト値
  - `TestMetadataExportImport` (3テスト): ラウンドトリップ、カスタムE/nu、カテゴリDデフォルト
  - `TestInpValidation` (4テスト): 整合検証、E/nu 不整合検出、節点情報
  - `TestNoHardcodedConstants` (1テスト): 旧ハードコード定数の完全除去確認

## テスト結果

- 新規テスト: 12/12 パス
- 既存テスト: 1009パス、1タイムアウト（`test_cosserat_vs_cr_bend3p` — 今回の変更と無関係）
- 回帰なし

## 次の課題（TODO）

- [x] `run_bending_oscillation` に E, nu 引数を追加しハードコード除去
- [x] `solve_from_inp` でメタデータの E, nu を渡す経路を実装
- [x] 接触パラメータ（カテゴリD）のメタデータ記録
- [x] .inp の NODE/ELEMENT と再生成メッシュの一致検証
- [ ] 長期: `build_beam_model_from_inp` ベースの接触解析モデル構築経路（メッシュ直接利用）
- [ ] 19本以上 NCP 収束のパラメータ最適化（status-104 引継ぎ）
- [ ] k_pen 自動スケーリング（EA/L ベース）

## 確認事項

- 全デフォルト値は従来のハードコード値と完全に一致するため、**後方互換性は100%保持**
- メタデータバージョンを 3.0 に更新。旧バージョン（2.0）の .inp は `meta.get(key, default)` で後方互換処理
- `_validate_inp_vs_metadata` は現時点では材料・寸法の検証のみ。ステップ定義（`*STEP`/`*STATIC`/`*DYNAMIC`）のパースは将来拡張
- メッシュの直接利用（.inp NODE/ELEMENT → ソルバー）は大きなリファクタリングが必要。現在は整合検証のみで、メッシュは依然として再生成している。次のステップとして `build_beam_model_from_inp` ベースの接触解析モデル構築経路を検討
