# status-313: 撚線ファイバー梁モデル 設計仕様策定（work/beam_hysteresis 結果の統合）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-10
- **ブランチ**: `claude/merge-fiber-beam-model-tFcuM`
- **テスト数**: 459 passed（変更なし。設計ドキュメントのみの追加）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

`work/beam_hysteresis/` ディレクトリで蓄積された撚線ヒステリシスの概念検証結果を
master にマージするための **設計仕様ドキュメント** を新規作成した。

コード実装は本 status ではまだ行わず、後続の Phase F1〜F6 で段階的に実装する計画。
本 status では以下を成果物とする：

1. **設計仕様書** `xkep_cae/elements/docs/fiber_beam_strand.md` 新規作成
2. **work/beam_hysteresis/README.md** 新規作成（概念検証の結論要約）
3. **docs/design/README.md** に設計文書リンク追加
4. **README.md / docs/status/status-index.md / docs/roadmap.md** 更新

---

## 背景

`work/beam_hysteresis/` では Stage 01〜08 で以下が確認済みだった：

| Stage | 結論 |
|-------|------|
| 01 | 1D 移動硬化 ≡ 1D 撚線摩擦（数学的同型） |
| 02-03 | 傾き非対称 U/L < 1 には接触剛性劣化（β ≈ 0.25）が必要 |
| 04-05 | N=150 多層摩擦＋繊維断面で角のない丸いティアドロップ |
| 06 | ジグ摩擦は乗算的、内部摩擦は加算的。分離同定可能 |
| 07 | 劣化ありモデルは Cycle 2 以降シェイクダウン |
| 08 | 7本撚線陽接触解析で散逸ループを確認（キャリブレーション目標） |

この知見を **1本のファイバー梁要素** に落とし込み、
現行の陽接触モデル（`StrandBendingOscillationProcess` 等）と相補する
高速近似モデルを追加するのが本設計の狙い。

---

## 成果物

### 1. `xkep_cae/elements/docs/fiber_beam_strand.md`（新規 457行）

以下の節を含む正式な設計仕様：

- **背景**: `work/beam_hysteresis/` Stage 01〜08 の数値的裏付けを数式付きで要約
- **スコープ / 非スコープ**
- **モジュール構成**: `xkep_cae/elements/fiber/` 配下の 5 ファイル構成
- **Strategy Protocol**: `Fiber1DMaterialStrategy` 追加（既存 Penalty/Friction と同作法）
- **状態 dataclass**: `Fiber1DState` / `SectionState`（frozen、C17 準拠）
- **ファイバー断面ジェネレータ**: `CircularFiberSection.strip / polar`
- **セクション積分 Process**: `FiberSectionIntegratorProcess`（軸–曲げカップリング接線行列込み）
- **梁要素ラッパ**: `StrandFiberBeamProcess`（CR Timoshenko に差し替え可能な材料層）
- **既存コードへの組み込みポイント**: `_beam_assembler.ULCRBeamAssembler` の分岐追加
- **テスト計画**: 4 クラス（API / Physics / Convergence / Integration）
- **実装フェーズ F1〜F6**
- **既知のリスク**: 接線 FD、状態肥大化、同定非一意性、動的積分結合

### 2. `work/beam_hysteresis/README.md`（新規）

- 概念検証の結論要約（6 項目）
- スクリプト一覧と主図の対応表
- キー方程式（撚線摩擦、剛性劣化、ジグ摩擦補正）
- キャリブレーション対象パラメータ表（自由度 3 に限定）
- 実行方法

### 3. `docs/design/README.md`

`fiber_beam_strand.md` を索引表に追加（状態: **仕様策定**）。

### 4. `README.md`

- 現在の状態バナーを status-313 に更新
- ドキュメント表に撚線ファイバー梁の 2 つのリンクを追加

### 5. `docs/status/status-index.md`（本ファイル）

- status-313 エントリ追加

### 6. `docs/roadmap.md`

- Phase 4.4–4.6 欄に進捗メモ追加
- 完了済み表に本 status を接続

---

## 実装計画（未着手）

本 status はドキュメントのみ。コードは **Phase F1〜F6** で段階実装：

| Phase | 内容 | 完了判定 |
|-------|------|---------|
| **F1** | `Elastic1D`, `BilinearKinematicHardening1D` | Physics テスト 6 件合格 |
| **F2** | `MultiLayerFrictionDegrading1D`（frozen 化） | `05_smooth_teardrop.py` rtol 1% |
| **F3** | `CircularFiberSection` + `FiberSectionIntegratorProcess` | FD 接線 atol 1e-5 |
| **F4** | `StrandFiberBeamProcess` + `_beam_assembler` 配線 | 弾性 EI 一致 < 0.1% |
| **F5** | `StrandBendingOscillationProcess` に `use_fiber_beam` | 散逸エネルギー一致 < 10% |
| **F6** | キャリブレーション Process（`tuning/`） | BenchmarkRunnerProcess マニフェスト |

---

## 再現手順

```bash
# 設計ドキュメントのプレビュー
cat xkep_cae/elements/docs/fiber_beam_strand.md
cat work/beam_hysteresis/README.md

# 概念検証スクリプトの再実行（裏付けの確認）
cd work/beam_hysteresis
python 05_smooth_teardrop.py 2>&1 | tee /tmp/log-$(date +%s).log
python 06_jig_friction.py 2>&1 | tee /tmp/log-$(date +%s).log

# テストスイート（回帰がないことの確認）
pytest tests/ -v -m "not slow and not external"
```

---

## 次にやること

1. **Phase F1 開始**: `xkep_cae/elements/fiber/` を切って
   `Elastic1D` / `BilinearKinematicHardening1D` 実装＋Physics テスト
2. `docs/design/README.md` の状態を **仕様策定 → 実装中** に更新
3. 既存の 7本撚線 `StrandBendingOscillationProcess` 結果を
   キャリブレーション真値として `tests/fixtures/` に固定化

---

## コミットメッセージ（予定）

```
feat(design): 撚線ファイバー梁モデル設計仕様策定 (status-313)

- xkep_cae/elements/docs/fiber_beam_strand.md 新規作成
  (Strategy/状態/積分/テスト計画/F1-F6 フェーズ)
- work/beam_hysteresis/README.md 新規作成（Stage 01-08 結論要約）
- docs/design/README.md に設計文書リンク追加
- README.md / status-index.md / roadmap.md 更新

概念検証（N=150 多層摩擦 + β=0.25 接触劣化 + 繊維断面）で確認済みの
ヒステリシスを 1 本のファイバー梁要素に等価化し、陽接触モデルと
相補する高速近似モデルを追加する設計。コード実装は F1-F6 で段階的に行う。
```
