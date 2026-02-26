# xkep-cae

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

## 現在の状態

**Phase 1〜3 + Phase 4.1〜4.2 + Phase 5.1〜5.4 + Phase C0〜C5 + 過渡応答出力 + FIELD ANIMATION出力 + GIFアニメーション出力 + CR梁定式化 + CR梁ファイバー弾塑性 + 摩擦接触バリデーション + 梁梁接触貫入テスト + 適応的ペナルティ増大 + 実梁要素接触テスト + 長距離スライドテスト + 接触バリデーションドキュメント + 大規模マルチセグメント性能評価 + 撚線メッシュファクトリ + 多点接触撚線テスト + 接触グラフ表現 + k_pen自動推定 + 段階的接触アクティベーション + ヘリカル摩擦安定化 + 接触グラフ可視化・時系列収集 + 7本撚り接触NR収束達成 + 撚線ヒステリシス観測 + 接触グラフ統計分析 + ヒステリシス可視化 + 統計ダッシュボード + 被膜モデル + シースモデル + シース挙動設計（解析的リングコンプライアンス+ペナルティ接触, Stage S1〜S4）+ Stage S1 解析的リングコンプライアンス行列 + HEX8 要素ファミリ拡充（C3D8/C3D8B/C3D8R/C3D8I + B-bar平均膨張法 + SRI+B-bar併用デフォルト + アセンブリ統合 + Abaqus命名準拠） + シース曲げモードバリデーション + Protocol 定義 3D 解析対応拡張 + 撚撚線（被膜付き撚線）統合解析テスト + Stage S2 シース内面Fourier近似 + 接触接線モード（contact_tangent_mode） + 2D定常熱伝導FEM + GNNサロゲートモデル + サロゲートモデル追加検証 + 全結合GNN + PINN導入検証 + 不規則メッシュGNN比較 + 接触ML設計仕様 完了。1612テスト。**
Phase 3.4: Q4要素の幾何学的非線形（TL定式化 + Updated Lagrangian）実装完了。
Phase 5: 陽解法（Central Difference）、モーダル減衰、非線形動解析ソルバー実装完了。
Phase C0: 梁–梁接触モジュール骨格（ContactPair/ContactState/geometry）実装完了。
Phase C1: Broadphase（AABB格子）+ ContactManager幾何更新 + Active-setヒステリシス実装完了。
Phase C2: 法線AL接触力 + 接触接線剛性（主項）+ 接触付きNRソルバー（Outer/Inner分離）実装完了。
Phase C3: Coulomb摩擦 return mapping + μランプ + 摩擦接線剛性 + 接線相対変位追跡 実装完了。
Phase C4: merit line search（Armijo backtracking） + merit-based Outer終了判定 + step length適応制御 実装完了。
Phase C5: 幾何微分込み一貫接線（K_geo） + slip consistent tangent（v0.2） + PDAS active-set（実験的） + 平行輸送フレーム更新 実装完了。
過渡応答出力: Abaqus準拠のStep/Increment/Frame階層 + CSV/JSON/VTK(ParaView)出力。
ステップ列自動実行（run_transient_steps）、非線形反力計算、VTKバイナリ出力、要素データ出力、.inpパーサー統合。
.inpパーサー拡張: *ELSET, *BOUNDARY, *OUTPUT FIELD ANIMATION, *MATERIAL, *ELASTIC, *DENSITY, *PLASTIC キーワード追加。
テーブル補間型硬化則: *PLASTIC テーブル → Plasticity1D/PlaneStrainPlasticity 変換（区分線形、コンバータ関数）。
HARDENING=KINEMATIC テーブル → Armstrong-Frederick 移動硬化パラメータ変換（線形/非線形AF近似）。
FIELD ANIMATION出力: 梁要素のx/y/z軸方向2Dプロット（要素セット色分け・凡例対応）。
GIFアニメーション出力: Pillow連携、ビュー方向ごとのGIF生成、フレーム間描画範囲固定。
サンプル入力ファイル: 5つの `.inp` ファイルを `examples/` に追加。
.inp実行スクリプト: `examples/run_examples.py` でサンプル .inp の解析実行・解析解比較。
Abaqus三点曲げバリデーション: `assets/test_assets/Abaqus/1-bend3p/` の結果と比較（剛性差異1.09%）。
三点曲げ非線形動解析スクリプト: Abaqus準拠パラメータで非線形動解析を実施、GIFアニメーション・比較プロット出力。
CR梁定式化: Timoshenko 3D梁のCorotational定式化による幾何学的非線形（大変形・大回転）対応。dynamic_runner統合（nlgeom=True）。
CR梁ファイバー弾塑性: FiberIntegrator + B行列定式化 + 解析的接線剛性。Abaqus B31弾塑性三点曲げ（idx2）とのバリデーション。
撚線メッシュファクトリ: 理想ヘリカル配置に基づく撚線梁メッシュ生成（3/7/19/37/61/91本対応）。
多点接触撚線テスト: 3本撚り5荷重タイプ + 摩擦3荷重タイプ成功、7本はNR収束限界でxfail。
接触グラフ表現: 多点接触の無向グラフ表現（トポロジー変遷追跡、連結成分分析、隣接行列出力）。
k_pen自動推定: EI/L³ベースのペナルティ剛性自動推定（接触ペア数スケーリング）。段階的接触アクティベーション。
ヘリカル摩擦安定化: 摩擦履歴の平行輸送（rotate_friction_history）でヘリカル接触幾何の収束を実現。
接触グラフ可視化: matplotlib描画（plot_contact_graph/history）+ GIFアニメーション（save_contact_graph_gif）。時系列自動収集。
7本撚り接触NR収束達成: AL乗数緩和(omega) + 反復ソルバー(GMRES+ILU) + Active Set Freeze + Pure Penalty方式で全3荷重ケース（引張・ねじり・曲げ）収束。
撚線ヒステリシス観測: サイクリック荷重ランナー（run_contact_cyclic）+ CyclicContactResult + 3本撚り引張/曲げ/ねじり往復荷重テスト。
接触グラフ統計分析: stick/slip比率、法線力統計、連結成分数、接触持続マップ、累積散逸エネルギー、サマリーメソッド。
ヒステリシス可視化: plot_hysteresis_curve（荷重-変位曲線） + compute_hysteresis_area（ループ面積）。
統計ダッシュボード: plot_statistics_dashboard（6パネル統計描画）。
被膜モデル: CoatingModel（剛性寄与 + 摩擦制御）。環状断面特性・複合断面剛性・被膜込み接触半径。
シースモデル: SheathModel（撚線全体を覆う円筒外被）。エンベロープ半径・円筒管断面特性・等価梁剛性・最外層素線特定・径方向ギャップ。
シース挙動設計: 解析的リングコンプライアンス行列+ペナルティ接触（DOF追加ゼロ、Fourier近似内面、有限滑り対応）。Stage S1〜S4ロードマップ策定済み。
HEX8 要素ファミリ: C3D8（SRI+B-bar併用+アワーグラス制御）、C3D8B（B-bar平均膨張法）、C3D8R（低減積分+アワーグラス制御）、C3D8I（非適合モード）。3D等方弾性テンソル。アセンブリ統合済み。Abaqus準拠命名。
シース曲げモードバリデーション: 3Dチューブメッシュ + HEX8 FEM で軸引張/圧縮・純曲げ・横せん断を検証。梁理論解析解と比較。
Protocol 定義の 3D 解析対応拡張: NonlinearElementProtocol（内力・接線剛性）、DynamicElementProtocol（質量行列）、PlasticConstitutiveProtocol（return mapping）を追加。既存 Protocol は後方互換で維持。
バリデーションテスト結果は[検証文書](docs/verification/validation.md)に図付きで文書化済み。

詳細は[ロードマップ](docs/roadmap.md)を参照。

## ドキュメント

- [ロードマップ](docs/roadmap.md) — 全体開発計画（Phase 1〜8 + Phase C）
- [使用例](docs/examples.md) — API・梁要素・非線形・弾塑性のコード例
- [バリデーション文書](docs/verification/validation.md) — 全Phase の解析解・厳密解との比較検証
- [検証図](docs/verification/) — 解析解比較の検証プロット（15枚）
- [接触テストカタログ](docs/verification/contact_test_catalog.md) — 全接触テスト（~240テスト）の系統的一覧
- [Cosserat rod 設計仕様書](docs/cosserat-design.md) — 四元数回転・Cosserat rod の設計
- [Abaqus差異](docs/abaqus-differences.md) — xkep-cae と Abaqus の既知の差異
- [梁–梁接触モジュール仕様書](docs/contact/beam_beam_contact_spec_v0.1.md) — 接触アルゴリズムの実装指針
- [過渡応答出力設計仕様](docs/transient-output-design.md) — Step/Increment/Frame + 出力インターフェースの設計
- [サンプル入力ファイル](examples/README.md) — `.inp` ファイルのサンプル集（片持ち梁、3点曲げ、門型フレーム等）
- [接触付き弧長法設計検討](docs/contact/arc_length_contact_design.md) — 接触問題でのリミットポイント追跡の設計方針
- [実装状況](docs/status/status-068.md) — 最新のステータス（PINN導入検証 + 不規則メッシュGNN比較 + 接触ML設計仕様）
- [ステータス一覧](docs/status/status-index.md) — 全ステータスファイルの一覧とテスト数推移

## インストール

```bash
pip install -e ".[dev]"
```

## テスト実行

```bash
pytest tests/ -v -m "not external"
```

## Lint / Format

```bash
ruff check xkep_cae/ tests/
ruff format xkep_cae/ tests/
```

## クイックスタート

```python
from xkep_cae.api import solve_plane_strain

u_map = solve_plane_strain(
    node_coord_array=nodes,
    node_label_df_mapping={1: (False, False), 2: (False, False)},
    node_label_load_mapping={5: (1.0, 0.0)},
    E=200e3, nu=0.3, thickness=1.0,
    elem_quads=elem_q4, elem_tris=elem_t3,
)
```

その他の使用例（梁要素、非線形解析、弾塑性解析、数値試験フレームワーク等）は
[docs/examples.md](docs/examples.md) を参照。

## 依存ライブラリ

- numpy, scipy（必須）
- pyamg（大規模問題時のAMGソルバー、オプション）
- numba（TRI6高速化、オプション）
- matplotlib（FIELD ANIMATION出力、オプション）
- Pillow（GIFアニメーション出力、matplotlib導入時に自動インストール）
- ruff（開発時lint/format）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
