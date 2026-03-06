# xkep-cae コーディング規約

## 言語・文書化

- 全ての回答・設計仕様は**日本語**で記述する
- すべての markdown 文書には原則 project 直下の `README.md` へのバックリンクを貼る

## 2交代制運用（Codex / Claude Code）

本プロジェクトは **Codex と Claude Code の2交代制**で運用する。常に互いへの引き継ぎを想定すること。

### ステータス管理

- 実装状況は `docs/status/status-{index}.md` に記録する
- **現在の状況**は index が最大の status ファイルに書かれている
- **`docs/status/status-index.md`** にステータス一覧を管理する（新規status作成時に必ず行を追加すること）
- status-001〜096 は `docs/status/archive/` に移動済み
- status に書いた内容は **git の commit メッセージと整合**を取ること
- 実装状況は細かく書き出す（別の AI アシスタントが参照して簡便に状況を把握する目的）

### 作業完了時の必須手順

1. **README.md** を更新（テスト数、マイルストーン進捗）
2. **status ファイル**を新規作成 or 更新（TODO は status に記入）
3. **status-index.md** を更新（新規statusの行を追加）
4. **roadmap.md** を更新（Phase S のチェックボックス、「現在地」）
5. 実装とドキュメントの不整合を発見したら**その場で修正**するか、TODO に追加
6. **feature ごとにコミットを切って**、最後に push

### 確認事項・懸念

- ユーザーへの確認事項や設計上の懸念は **status ファイルに書き出す**こと

## コード規約

- テスト駆動: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
- 後方互換性を保ちながら拡張（既存テストを破壊しない）
- lint/format: `ruff check xkep_cae/ tests/` && `ruff format xkep_cae/ tests/`

### テストの2分類: プログラムテスト vs 物理テスト

テストは以下の2種類を**明示的に分けて両方作成**すること。

#### 1. プログラムテスト（収束・エラー・API正しさ）

- ソルバーが収束するか、例外が正しく発生するか、APIの入出力が仕様通りか
- 粗いメッシュ（計算速度優先）で十分。`min_elems_per_pitch=0` で密度チェックを明示的にスキップ
- クラス名に `Test〇〇API`, `Test〇〇Convergence`, `Test〇〇Validation` など

#### 2. 物理テスト（物理的に当然の性質の検証）

- 人間が目視・経験則で判断する「物理的に当然のこと」をコード化する
- クラス名に `Test〇〇Physics`, `Test〇〇Physical` など
- 具体例:
  - **貫入量**: 弦近似による初期貫入 < ワイヤ直径の2%（16要素/ピッチ以上の場合）
  - **応力・曲率の連続性**: 隣接要素間の応力差が極端に離散的でないか
  - **荷重オーダー**: 出力荷重が剛性×変位の解析解と同オーダーか
  - **変形の対称性**: 対称荷重に対して対称な変形が出るか
  - **エネルギー保存**: 外力仕事 ≈ 内部エネルギー（動的解析）
  - **接触力の方向**: 法線方向を向いているか、大きさが妥当か

#### 3. 視覚的妥当性検証（AI目視検査）

実務の人間エンジニアは3Dモデルを回転・拡大して目視で物理的異常を発見する。
AI駆動開発でもこのプロセスを再現する:

- **変形メッシュの2D投影**: 四元数モジュール（`xkep_cae/math/`）で任意視点からの回転を適用し、matplotlib で2D投影図を生成
- **コンター図**: 応力・曲率・接触力等のスカラー場を色マップで可視化し、不自然な離散値や特異点がないか確認
- **検証スクリプト**: `tests/generate_verification_plots.py` に視覚検証用の図生成を追加
- **判定基準**: コンター図の隣接要素間の値変化率、変形の滑らかさ、対称性の定量指標を物理テストに組み込む
- `pytest` 実行時には図を生成しない。検証図は別スクリプトで生成し `docs/verification/` に保存

> **原則**: 「プログラムが動く」だけでなく「物理的に正しい」ことを常にテストする。
> 人間の目視検査に相当する定量チェックを物理テストとして自動化する。

### 機能実装の互換ヒストリー

機能の置き換え・統合を行った場合は、以下のルールに従うこと。

1. **互換ヒストリーテーブル**を status ファイルに必ず記録する:

   | 旧機能 | 新機能 | 移行status | 備考 |
   |--------|--------|-----------|------|
   | `newton_raphson_with_contact` | `newton_raphson_contact_ncp` | status-107 | NCP統合、旧テストdeprecated化 |

2. **新機能をデフォルトにする**: 旧機能を後方互換のためにデフォルトにし続けない。推奨構成が確立されたら速やかにデフォルトを切り替え、旧機能は `deprecated` マーカーを付与する
3. **旧テストの移行**: deprecated化したテストは対応するNCP移行版テストを作成し、旧テストファイル冒頭にコメントで移行先を明記する（例: `# DEPRECATED: NCP版は test_xxx_ncp.py を参照`）
4. **旧コードの削除判断**: deprecated化から2 status以上経過し、全テストがNCP版で代替されていれば旧コードの削除を検討する。削除時は status に互換ヒストリーとして記録する

#### 現在の互換ヒストリー

| 旧機能 | 新機能 | 移行status | 状態 |
|--------|--------|-----------|------|
| `newton_raphson_with_contact`（ペナルティ/AL） | `newton_raphson_contact_ncp`（NCP） | status-107→108 | NCP移行版テスト作成完了 |

### 検証結果のドキュメント化

- 文献値や解析解との比較結果は `docs/verification/` に**図付き**で残す
- 図は `tests/generate_verification_plots.py` で生成（matplotlib → PNG）
- `pytest` 実行時にはプロットを生成しない（図生成は別スクリプト）

### 3D可視化検証（必須）

新しい要素・接触・撚線機能を実装した際は、以下の3Dレンダリング検証を**必ず**実施すること。

1. **3D梁表面レンダリング**: `_beam_surface_mesh()` で梁中心線からパイプ形状を生成し、`Poly3DCollection` で3D表示。変形メッシュの滑らかさ・対称性を確認
2. **応力/曲率コンター**: 梁表面に応力値・曲率をカラーマップで表示。隣接要素間の値変化が連続的であることを確認
3. **接触力ベクトル**: 接触ペア間の法線力・摩擦力を矢印（quiver3D）で表示し、方向と大きさの妥当性を確認

**実施ルール**:
- 新機能追加時は `tests/generate_verification_plots.py` に対応する3Dプロット関数を追加
- `docs/verification/` に出力PNGを保存し、statusファイルで結果を参照
- 撚線テスト（7本以上）では必ず3D表面レンダリングで変形形状を確認
- 物理テストの定量チェック（隣接要素間の値変化率、変形の滑らかさ指標）も併用

**利用可能な3Dプロット**:
- `plot_twisted_wire_3d_surface()`: 撚線3Dパイプ表面 + 曲率コンター
- `plot_beam_3d_stress_contour()`: 片持ち梁3D応力コンター（変形形状付き）

### NCP接触テストの推奨構成

NCPソルバーを使う接触テストでは、以下のS3改良機能を**原則有効化**すること。

| パラメータ | 推奨値 | 説明 |
|-----------|-------|------|
| `adaptive_timestepping` | `True` | 自動安定時間増分制御 |
| `adjust_initial_penetration` | `True` | 初期貫入オフセット補正 |
| `contact_force_ramp` | `True`（大規模問題） | 接触力ランプ |
| `k_pen_continuation` | `True`（大規模問題） | ペナルティ剛性continuation |

ばねモデルなど単純な接触問題でも `adaptive_timestepping=True` と `adjust_initial_penetration=True` は有効化する。

## プロジェクト構成

```
xkep_cae/
├── core/           # Protocol 定義
├── elements/       # 要素（README.md に一覧）
├── materials/      # 構成則
├── sections/       # 断面モデル
├── math/           # 四元数, SO(3)
├── contact/        # 梁–梁接触（README.md に設計詳細）
├── mesh/           # 撚線メッシュ（README.md に構成）
├── thermal/        # 熱伝導 + GNN/PINN
├── tuning/         # チューニングタスクスキーマ
├── numerical_tests/ # 数値試験フレームワーク
├── output/         # 過渡応答出力
├── io/             # Abaqus .inp パーサー
├── solver.py       # NR, 弧長法
├── assembly.py     # アセンブリ
├── dynamics.py     # 動的解析
├── bc.py           # 境界条件
└── api.py          # 高レベル API
docs/
├── roadmap.md      # ロードマップ + マイルストーン
├── archive/        # 完了済みPhase詳細
├── status/         # アクティブstatus（097〜）
│   └── archive/    # 旧status（001〜096）
├── contact/        # 接触モジュール設計仕様群
└── verification/   # バリデーション文書・検証図
```

## 現在の状態

**2263テスト（fast: 1689 / slow: 356 + deprecated: 218）** — 2026-03-06

FEM基盤（梁/平面/固体要素）、非線形（幾何学的/材料）、動的解析、梁–梁接触（NCP/Mortar/Line contact/摩擦）、撚線モデル（7本撚り収束/被膜/シース）、高速化基盤（COO/CSR/共有メモリ並列/ブロック前処理）、GNN/PINNサロゲートPoC — 全て完了。

### ターゲットマイルストーン

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 推奨ソルバー構成

- `newton_raphson_contact_ncp`（`solver_ncp.py`）
- Line-to-line Gauss積分 + Mortar + 同層除外 + Fischer-Burmeister NCP
- DOF閾値で直接法/GMRES自動切替 + ブロック前処理

### 次の課題

**S3: 19本NCP収束達成** → 37本以上 → S4: 剛性比較 → S5: ML → S6: 1000本6時間 → S7: GPU

詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。
