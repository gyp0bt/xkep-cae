# status-300: 変形メッシュ2D投影可視化スクリプト実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-07
- **ブランチ**: `claude/check-status-todos-53DoM`
- **テスト数**: 442+ passed（既存テスト全合格）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-299のTODO「変形メッシュの2D投影可視化」を実装。
`contracts/visualize_deformed_mesh_2d.py` に4パネル2D投影図（XZ側面 + XY端面）と
変形過程スナップショット（6フレーム時系列）の描画機能を追加した。

---

## 実装内容

### contracts/visualize_deformed_mesh_2d.py

1. **`plot_2d_projection()`**: 4パネル構成の2D投影図
   - 左上: 初期配置 XZ平面（側面図）
   - 右上: 変形後 XZ平面 + エラスティカ理論線
   - 左下: 初期配置 XY平面（端面図）
   - 右下: 変形後 XY平面（端面図）
   - 素線ごとに色分け（7色）、アスペクト比等倍

2. **`plot_2d_snapshots()`**: 変形過程の時系列スナップショット
   - displacement_history から等間隔でフレーム抽出（デフォルト6フレーム）
   - 各フレームのfrac値をタイトルに表示
   - XZ平面に投影

3. **CLI**: コマンドライン引数で揺動振幅を指定可能
   ```bash
   # 90度曲げのみ
   python contracts/visualize_deformed_mesh_2d.py
   # 90度曲げ + ±48mm揺動
   python contracts/visualize_deformed_mesh_2d.py --oscillation 48
   ```

### 動作確認

ダミーデータ（7本素線×8要素、理論的90度曲げ変形）で両関数の正常動作を確認済み。
- 2D投影図: XZ平面でエラスティカ理論線との一致確認
- スナップショット: frac=0→1の変形過程が正しく描画されることを確認

フルソルバー実行で4枚の検証画像を生成し `docs/verification/` に保存済み:
- `deformed_mesh_2d_90deg_bend.png`: 90度曲げ（frac=1.0, 727.6s）
- `deformed_mesh_2d_90deg_bend_snapshots.png`: 90度曲げ変形過程スナップショット
- `deformed_mesh_2d_90deg_osc48mm.png`: 90度曲げ+±48mm揺動（frac=1.0, 1459.1s）
- `deformed_mesh_2d_90deg_osc48mm_snapshots.png`: 曲げ+揺動変形過程スナップショット

---

## 変更ファイル

- `contracts/visualize_deformed_mesh_2d.py`: 新規作成

---

## TODO

- [x] フルソルバー実行で90度曲げ2D投影図を生成し `docs/verification/` に保存 → 完了（frac=1.0000, incr=535, cutback=45, 727.6s）
- [x] フルソルバー実行で90度曲げ+±48mm揺動2D投影図を生成し `docs/verification/` に保存 → 完了（frac=1.0000, incr=1900, cutback=72, 1459.1s）
- [ ] cutback数削減（72→30以下）: 計算効率改善
- [ ] 揺動フェーズの物理的妥当性検証（応力分布、接触力履歴）
- [ ] MPC+contact: ローカルMPC（ワイヤ単位の端部結合）の検討

---

## 次の担当者向け

### 可視化スクリプトの使い方

```bash
# 90度曲げのみ → docs/verification/deformed_mesh_2d_90deg_bend.png
python contracts/visualize_deformed_mesh_2d.py 2>&1 | tee /tmp/log-viz-bend-$(date +%s).log

# 90度曲げ + ±48mm揺動 → docs/verification/deformed_mesh_2d_90deg_osc48mm.png
python contracts/visualize_deformed_mesh_2d.py --oscillation 48 2>&1 | tee /tmp/log-viz-osc48-$(date +%s).log
```

描画関数 `plot_2d_projection()` と `plot_2d_snapshots()` は独立して使用可能。
ソルバー結果のcoords, coords_def, conn, strand_ids を渡すだけでよい。

### 残りのTODO優先度

1. **cutback削減**（最大インパクト）: 中盤37回の散発的カットバック削減が最優先
2. **物理妥当性検証**: 応力分布・接触力の可視化（別スクリプトで実装推奨）
3. **2D投影図生成**: フルソルバー実行のみ必要（コード変更不要）

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: ダミーデータテストの結果をそのまま記録
- [x] **回帰なし**: 新規ファイルのみ、既存コード変更なし
- [x] **再現手順記載**: ダミーデータテスト手順を記録
