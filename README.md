# xkep-cae

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

## 現在の状態

**Phase 1〜3 + Phase 4.1〜4.2 + Phase 5.1〜5.4 + Phase C0〜C5 + 過渡応答出力 + FIELD ANIMATION出力 + GIFアニメーション出力 + CR梁定式化 + CR梁ファイバー弾塑性 + 摩擦接触バリデーション完了。1009テスト。**
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
バリデーションテスト結果は[検証文書](docs/verification/validation.md)に図付きで文書化済み。

詳細は[ロードマップ](docs/roadmap.md)を参照。

## ドキュメント

- [ロードマップ](docs/roadmap.md) — 全体開発計画（Phase 1〜8 + Phase C）
- [使用例](docs/examples.md) — API・梁要素・非線形・弾塑性のコード例
- [バリデーション文書](docs/verification/validation.md) — 全Phase の解析解・厳密解との比較検証
- [検証図](docs/verification/) — 解析解比較の検証プロット（12枚）
- [Cosserat rod 設計仕様書](docs/cosserat-design.md) — 四元数回転・Cosserat rod の設計
- [Abaqus差異](docs/abaqus-differences.md) — xkep-cae と Abaqus の既知の差異
- [梁–梁接触モジュール仕様書](docs/contact/beam_beam_contact_spec_v0.1.md) — 接触アルゴリズムの実装指針
- [過渡応答出力設計仕様](docs/transient-output-design.md) — Step/Increment/Frame + 出力インターフェースの設計
- [サンプル入力ファイル](examples/README.md) — `.inp` ファイルのサンプル集（片持ち梁、3点曲げ、門型フレーム等）
- [接触付き弧長法設計検討](docs/contact/arc_length_contact_design.md) — 接触問題でのリミットポイント追跡の設計方針
- [実装状況](docs/status/status-047.md) — 最新のステータス（摩擦接触バリデーション + 接触付き弧長法設計検討）
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
