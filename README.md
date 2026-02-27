# xkep-cae

[![CI](https://github.com/gyp0bt/xkep-cae/actions/workflows/ci.yml/badge.svg)](https://github.com/gyp0bt/xkep-cae/actions/workflows/ci.yml)

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

## 現在の状態

**1629テスト（fast: 1344 / slow: 283 / external: 2）**

| フェーズ | 状態 |
|---------|------|
| Phase 1: アーキテクチャ（Protocol/ABC） | ✓ |
| Phase 2: 空間梁要素（EB/Timo/Cosserat） | ✓ |
| Phase 3: 幾何学的非線形（NR, 弧長法, CR, TL/UL） | ✓ |
| Phase 4.1-4.2: 弾塑性 + ファイバーモデル | ✓ |
| Phase 4.3: von Mises 3D | 凍結 |
| Phase 5: 動的解析（Newmark-β, 陽解法, モーダル減衰） | ✓ |
| Phase C0-C5: 梁–梁接触（AL, 摩擦, merit LS, PDAS） | ✓ |
| Phase 4.7 L0: 撚線基礎（7本撚り収束, ヒステリシス） | ✓ |
| Phase 4.7 L0.5 S1-S2: シース挙動 | ✓ |
| HEX8要素ファミリ（C3D8/C3D8B/C3D8R/C3D8I） | ✓ |
| 過渡応答出力 + FIELD ANIMATION + GIF | ✓ |
| Phase 6.0: 2D熱伝導GNN/PINNサロゲート（PoC） | ✓ |
| GitHub Actions CI + slowテストマーカー | ✓ |

詳細は[ロードマップ](docs/roadmap.md)および最新の[ステータス](docs/status/status-index.md)を参照。

## ドキュメント

### 計画・設計

- [ロードマップ](docs/roadmap.md) — 全体開発計画と TODO
- [ステータス一覧](docs/status/status-index.md) — 全ステータスファイルとテスト数推移

### 設計仕様

- [Cosserat rod 設計](docs/cosserat-design.md) — 四元数回転・B行列定式化
- [過渡応答出力設計](docs/transient-output-design.md) — Step/Increment/Frame + 出力I/F
- [梁–梁接触仕様群](docs/contact/) — 接触アルゴリズム・撚線改善・ML設計

### バリデーション

- [検証文書](docs/verification/validation.md) — 解析解・厳密解との比較（検証図15枚）
- [接触テストカタログ](docs/verification/contact_test_catalog.md) — 全接触テスト（~240テスト）

### 利用ガイド

- [使用例](docs/examples.md) — API・梁要素・非線形・弾塑性のコード例
- [Abaqus差異](docs/abaqus-differences.md) — xkep-cae と Abaqus の既知の差異
- [サンプル入力ファイル](examples/README.md) — `.inp` ファイルのサンプル集

## インストール

```bash
pip install -e ".[dev]"
# ML機能（GNN/PINN）を使う場合
pip install -e ".[dev,ml]"
```

## テスト実行

```bash
# 高速テストのみ（~3分）
pytest tests/ -v -m "not slow and not external"

# 全テスト（~30分, slow含む）
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

その他の使用例は [docs/examples.md](docs/examples.md) を参照。

## プロジェクト構成

```
xkep_cae/
├── core/           # Protocol 定義（Element, Constitutive, State）
├── elements/       # 要素（Q4, TRI3/6, Beam, Cosserat, HEX8）
├── materials/      # 構成則（弾性, 1D/3D弾塑性）
├── sections/       # 断面モデル（BeamSection, FiberSection）
├── math/           # 数学ユーティリティ（四元数, SO(3)）
├── contact/        # 梁–梁接触（Broadphase, AL, 摩擦, グラフ）
├── mesh/           # メッシュ生成（撚線, シース, チューブ）
├── thermal/        # 熱伝導FEM + GNN/PINNサロゲート
├── numerical_tests/ # 数値試験フレームワーク
├── output/         # 過渡応答出力（CSV/JSON/VTK/GIF）
├── io/             # Abaqus .inp パーサー
├── solver.py       # 線形/非線形ソルバー
├── assembly.py     # アセンブリ
├── dynamics.py     # 動的解析（Newmark-β, HHT-α, 陽解法）
├── bc.py           # 境界条件
└── api.py          # 高レベル API
docs/
├── roadmap.md      # 全体ロードマップ
├── archive/        # 完了済みPhase詳細設計
├── status/         # ステータスファイル群（73個）
├── contact/        # 接触モジュール仕様群
└── verification/   # バリデーション文書・検証図
```

## 依存ライブラリ

- numpy, scipy（必須）
- pyamg（大規模問題AMGソルバー、オプション）
- numba（TRI6高速化、オプション）
- matplotlib, Pillow（可視化・GIF出力、オプション）
- torch, torch-geometric（GNN/PINNサロゲート、`[ml]`オプション）
- ruff（開発時lint/format）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
