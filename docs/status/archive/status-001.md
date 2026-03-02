# status-001: プロジェクト棚卸しとロードマップ策定

[← README](../../README.md) | [ロードマップ](../roadmap.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/review-pycae-roadmap-D0h71`

---

## 実施内容

### コードベース棚卸し

プロジェクト全体を精査し、現状を把握した。

#### 現在のプロジェクト構成

```
pycae/
├── api.py              (218行) ラベルベース高レベルAPI
├── assembly.py         (437行) 全体剛性行列構築（COO形式）
├── bc.py               (121行) 境界条件（消去法/Penalty法）
├── solver.py           (117行) 線形ソルバー（spsolve/pyamg）
├── elements/
│   ├── quad4.py        (66行)  Q4双線形四角形
│   ├── quad4_bbar.py   (110行) Q4 + B̄法（体積ロッキング対策）
│   ├── tri3.py         (52行)  TRI3一次三角形
│   └── tri6.py         (256行) TRI6二次三角形（Numba対応準備済み）
└── materials/
    └── elastic.py      (20行)  線形弾性（平面ひずみ）

tests/
├── test_linear_elastic_mixed.py              製造解テスト
├── test_with_abaqus_data_manually_tensile.py 引張試験ベンチマーク
├── test_with_abaqus_data_manually_shear.py   せん断試験ベンチマーク
├── test_with_abaqus_data_manually_cutter_sample1.py  実メッシュテスト
└── test_with_abaqus_data_manually_cutter_sample5.py  TRI6メッシュテスト
```

**総コード行数**: 約1,397行

#### 実装済み機能

- **要素**: Q4, TRI3, TRI6, Q4_BBAR（すべて平面ひずみ）
- **材料**: 線形弾性（平面ひずみ）
- **ソルバー**: 直接法（spsolve）+ AMG反復法（pyamg）
- **境界条件**: Dirichlet消去法、Penalty法
- **アセンブリ**: COO形式→CSR変換、混在メッシュ対応
- **API**: ラベル→内部インデックス自動変換、エンドツーエンド解法
- **検証**: 製造解テスト、Abaqus比較、実メッシュ統合テスト

#### リポジトリ外で完了済み（未コミット）

- CalculiXのCPE4を参考にした内部ひずみ空間による剛性補正付き修正四角形要素
- Abaqus/CalculiXとの精度ベンチマーク

#### 未実装

- 梁要素（1D）
- 3次元要素
- 非線形解析（幾何学/材料）
- 動的解析
- 応力・ひずみポスト処理
- pyproject.toml等のプロジェクト標準構成
- docs/ディレクトリ

### ロードマップ策定

`docs/roadmap.md` を作成。全8フェーズ構成。

| Phase | 内容 | 前提 |
|-------|------|------|
| 1 | アーキテクチャ再構成（Protocol導入、プロジェクト基盤整備） | なし |
| 2 | 空間梁要素（E-B梁→Timoshenko 2D→Timoshenko 3D） | Phase 1 |
| 3 | 幾何学的非線形（N-R法、共回転定式化） | Phase 2 |
| 4 | 材料非線形（弾塑性、粘弾性、異方性、ファイバーモデル） | Phase 3 |
| 5 | 動的解析（Newmark-β、HHT-α法） | Phase 3 |
| 6 | NNサロゲートモデル対応 | Phase 4 |
| 7 | モデルレジストリ・パラメータフィッティング | Phase 2-6 |
| 8 | 応用展開（接触、ミクロ-マクロ、3D固体） | 随時 |

**クリティカルパス**: Phase 1 → 2 → 3 → 4

---

## 作成ドキュメント

| ファイル | 内容 |
|---------|------|
| `docs/roadmap.md` | 全体ロードマップ |
| `docs/status/status-001.md` | 本ファイル（棚卸し結果） |

---

## TODO（次回以降の作業）

- [ ] Phase 1.1: `pyproject.toml` の作成
- [ ] Phase 1.1: テストフレームワーク（pytest）統一
- [ ] Phase 1.2: `core/` ディレクトリ作成、ElementProtocol / ConstitutiveProtocol 設計
- [ ] Phase 1.3: 既存要素・材料のProtocol適合

---

## ユーザーへの確認事項

1. **Phase 1（アーキテクチャ再構成）の着手タイミング**：ロードマップ承認後すぐに開始してよいか？
2. **既存の補正四角形要素**：リポジトリ外で完了している修正四角形要素のコードは、このリポジトリに取り込む予定はあるか？取り込む場合はPhase 1の移行時に一括で対応可能。
3. **依存ライブラリの方針**：pymeshは社内ツールと見受けられるが、外部依存として維持するか、メッシュI/Oを自前実装するか？
4. **NN関連のフレームワーク選定**：Phase 6でPyTorchを想定しているが、JAXやその他の選択肢はあるか？
5. **CI/CD環境**: GitHub Actions等のCI基盤は利用可能か？

---

## 設計上の懸念

1. **Protocol vs ABC**: Python 3.8+のProtocolはランタイムチェックが弱い。ABCの方が実装漏れを早期検出できるが、鋳型への強制が強い。構成則のNN代替を考えるとProtocolの柔軟性が有利。両方を検討の上、Phase 1で判断する。
2. **回転のパラメタ化**: 空間梁の大回転を扱う際、四元数 vs 回転ベクトル vs 方向余弦行列の選択が必要。四元数は特異点がないが直感性に欠ける。共回転定式化では回転ベクトル（指数写像）が一般的。Phase 3の設計時に決定する。
3. **ファイバーモデルの計算コスト**: 断面のファイバー分割数が多いと各要素の構成則評価が高コストになる。NNサロゲートの第一候補になりうる。
