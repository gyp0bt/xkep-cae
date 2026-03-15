# status-132: NCP高速化6x + 揺動Phase2収束達成 + 運用厳格化

[← README](../../README.md) | [← status-index](status-index.md) | [← status-131](status-131.md)

**日付**: 2026-03-07

## 概要

status-131のTODOを実行。NCPソルバーのプロファイリングに基づく6倍高速化、揺動Phase2の収束達成、CLAUDE.md改訂、検証スクリプト整備を実施。

## 実施内容

### 1. NCPソルバー プロファイリング・高速化

cProfileで7本撚線45度曲げのボトルネックを特定:

| 段階 | 時間 | 高速化比 | 主な変更 |
|------|------|---------|---------|
| 初期（数値微分接線） | 49.6秒 | 1.0x | — |
| 解析的接線剛性に切替 | 12.9秒 | 3.8x | `ULCRBeamAssembler.analytical_tangent=True` |
| + バッチ接触幾何 | **8.3秒** | **6.0x** | `closest_point_segments_batch`, `build_contact_frame_batch` |

#### 高速化1: 解析的接線剛性

`ULCRBeamAssembler.assemble_tangent` が `analytical_tangent=False`（数値微分）をハードコードしていた。
数値微分は1要素あたり26回の内力評価が必要で、全計算時間の78%を消費。
解析的接線剛性は数値微分と3.25e-05の差異で十分正確。

#### 高速化2: バッチ接触幾何

`update_geometry` が各接触ペアに対してPythonループで `closest_point_segments` + `build_contact_frame` を呼んでいた。
numpy einsum/cross のバッチ版を実装し、全ペアを一括処理。
`update_geometry`: 4.6秒 → 0.14秒（33倍高速化）

### 2. 揺動Phase2 収束達成

**問題**: Phase2（揺動）が「Matrix is exactly singular」で発散
- 原因: 各揺動ステップを `n_load_steps=1` で個別に呼び出し、adaptive_timesteppingが無効
- 90度曲げ後の参照配置から2mm変位をワンステップで解くと特異行列が発生

**修正**: Phase2をPhase1と同じ `prescribed_dofs` + `adaptive_timestepping` + `ul_assembler` 方式に変更
- 各揺動ステップの増分z変位をNCP内部の荷重ステッピングで処理
- adaptive_timesteppingが自然に増分を制御

**結果**: 7本撚線90度曲げ+揺動1周期が26.6秒で完全収束（xfail除去）

### 3. Phase2初期時間増分の物理ベース自動推定

Phase2揺動で`n_load_steps`をハードコード分割していた問題を修正。

**問題**: `n_load_steps=max(2, n_steps_per_quarter)`は`adaptive_timestepping=True`と矛盾。ハードコード分割は問題の温床。

**修正**: 曲げPhase1と同等のスケールで初期Δtを自動推定:
- Phase1: `max_angle_per_step_deg`（デフォルト3°）で初期分割数を決定
- Phase2: 各揺動ステップの`|incr_z| / amplitude`を荷重分率とみなし、Phase1の3°相当の等価増分で初期分割数を算出
- 算出された初期値を`n_load_steps`として渡し、`adaptive_timestepping`が自動的に増減

**効果**: 200秒→130秒に改善（無駄なカットバック削減）

### 4. CI修正

`pyproject.toml` の `[dev]` 依存に `pyyaml>=6.0` を追加。
CIの全runで失敗していた `test_task_yaml_roundtrip` を解消。

### 5. CLAUDE.md改訂

- コンテキスト削減: 192行 → 75行
- 新機能の収束検証フロー厳格化:
  - `scripts/` にスクリプト作成 → teeでログ出力 → 2D投影で物理検証 → pytestに移行
- 推奨ソルバー構成にUL+NCP統合・解析的接線剛性を明記

### 6. 検証スクリプト整備

- `scripts/verify_7strand_bending_oscillation.py`: 45度曲げ + 90度曲げ+揺動の収束検証
- `scripts/profile_ncp_7strand.py`: cProfileプロファイリング
- `docs/verification/7strand_bending_oscillation_2d.png`: XY平面2D投影
- `docs/verification/7strand_bending_oscillation_xz.png`: XZ平面2D投影

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | ULCRBeamAssembler: analytical_tangent=True に切替 |
| `xkep_cae/contact/geometry.py` | `closest_point_segments_batch`, `build_contact_frame_batch` 追加 |
| `xkep_cae/contact/pair.py` | `update_geometry` をバッチ版に切替 |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | Phase2をprescribed_dofs+adaptive方式に変更 |
| `tests/contact/test_ncp_bending_oscillation.py` | 7本揺動テストのxfail除去 |
| `pyproject.toml` | pyyaml を dev 依存に追加 |
| `CLAUDE.md` | コンテキスト削減・運用厳格化 |
| `scripts/verify_7strand_bending_oscillation.py` | 新規: 収束検証スクリプト |
| `scripts/profile_ncp_7strand.py` | 新規: プロファイリングスクリプト |

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 備考 |
|--------|--------|-----------|------|
| 数値微分接線(UL) | 解析的接線(UL) | status-132 | `analytical_tangent=True` |
| スカラー版接触幾何 | バッチ版接触幾何 | status-132 | `*_batch()` 関数追加 |
| Phase2手動ループ | Phase2 prescribed+adaptive | status-132 | UL+NCP統合方式 |
| Phase2 n_load_stepsハードコード | 物理ベース初期推定+adaptive | status-132 | 曲げ3°等価のΔtで初期分割 |

## TODO

### 次の優先

- [ ] 19本撚線の曲げ揺動収束確認（scripts/で検証）
- [ ] 要素ループのベクトル化（残りの46%ボトルネック）
- [ ] 19本→37本のスケールアップ

### 確認事項

- 45度曲げで接触力(NCP active)が0: 0.5ピッチの短いモデルでは初期貫入オフセット補正後のギャップ>0で実効的接触が起きにくい。長いモデル(1ピッチ以上)で検証すべき
- 最大貫入比0.237は90度曲げ+揺動後として大きめ → モデル長さ・要素密度の影響を調査
