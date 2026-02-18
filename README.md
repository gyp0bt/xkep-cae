# xkep-cae

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

## 現在の状態

**Phase 1〜3 + Phase 4.1〜4.2 + Phase 5.1〜5.4 完了。556テストパス。**
Phase 3.4: Q4要素の幾何学的非線形（TL定式化, Green-Lagrangeひずみ, S2PK応力）実装完了。
Phase 5.2: 梁要素に mass_matrix() メソッド追加。Phase 5.4: 非線形動解析ソルバー実装完了。
数値三点曲げ試験の非線形動解析対応（dynamic_runner）実装完了。
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
- [実装状況](docs/status/status-028.md) — 最新のステータス（Phase 3.4 + 5.4 + 動的三点曲げ）
- [ステータス履歴](docs/status/) — 全ステータスファイル一覧

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
- ruff（開発時lint/format）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
