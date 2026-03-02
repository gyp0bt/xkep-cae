# status-016: 梁–梁接触モジュール仕様追加とロードマップ統合

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-015](./status-015.md)

**日付**: 2026-02-14
**作業者**: Codex
**ブランチ**: `work`

---

## 実施内容

status-015 時点の Phase 3 開始状態を受け、次の実装ターゲットである梁–梁接触について、
実装可能な粒度まで仕様を落とし込み、プロジェクト計画へ編入した。

1. `docs/contact/beam_beam_contact_spec_v0.1.md` を新規作成
   - AL（法線）
   - Active-set（ヒステリシス）
   - return mapping（摩擦）
   - Outer/Inner 分離（探索と求解の分離）
   - merit line search
   を中核に据えた接触設計を定義。
2. `docs/roadmap.md` を更新
   - 現在地を「Phase 3 + 接触モジュール設計追加」に更新
   - 未実装項目を現状に合わせて是正
   - 接触モジュール専用の実装フェーズ（C0〜C4）を明示
3. `README.md` を更新
   - 最新statusリンクを `status-016` に更新
   - 接触仕様書への導線を追加

---

## 設計仕様（要点）

### 1) 収束最優先の方針
- 最近接を毎Newtonで更新すると不連続ジャンプで破綻しやすいため、
  **Inner Newtonでは最近接 `(s,t)` を固定**し、Outerでのみ更新する。
- 法線は penalty単独ではなく **AL** を採用し、過剛性化と残留貫通の両リスクを抑える。
- 摩擦は経験則分類ではなく **return mapping（投影）** で一貫更新する。

### 2) v0.1での実装割り切り
- `K_c` は主項中心（penalty/AL支配項、stick接線）で先行。
- 幾何微分込みの厳密接線は v0.2（C5）へ後送。
- C¹正則化は切替式（デフォルトOFF）で導入し、最初は単純系で安定性を確認。

### 3) テスト先行順序
- 幾何 → 法線 → 摩擦 → 統合
- 特に「探索固定 vs 毎Newton探索更新」の比較テストを必須とし、
  アーキテクチャ選定の妥当性を再現可能にする。

---

## 次作業（TODO）

### 優先度A（次担当で即着手）
- [ ] Phase C0 開始: `xkep_cae/contact/` 配下に `ContactPair`, `ContactState`, `solver_hooks` 骨格を追加
- [ ] `tests/contact/test_geometry_segment_segment.py` を先行追加（解析解ケース、平行・端点ケース）
- [ ] `newton_raphson` 呼び出しに接触フックを注入する最小I/Fを定義

### 優先度B（C1〜C2）
- [ ] AABB格子 broadphase 実装（まずは単純格子）
- [ ] 法線AL（`lambda_n` 更新、`k_pen` クリップ）
- [ ] Active-set ヒステリシス（`g_on`, `g_off`）

### 優先度C（C3〜C4）
- [ ] 摩擦return mapping + `μ`ランプ
- [ ] merit function による line search
- [ ] Outer/Inner 反復統計ログ（反転回数、line search発動率）

---

## 確認事項・設計上の懸念

- [ ] `k_pen` 自動初期化スケール（`EA/L` 基準）の係数は、梁断面差が大きい混在ケースで再調整が必要。
- [ ] 接線フレーム `t1,t2` の履歴更新（最短回転）をC3で入れない場合、滑り方向が振動する懸念あり。
- [ ] broadphase はC1で格子実装としたが、要素密度が偏るメッシュではBVH移行を早める判断が必要。

---

## 引き継ぎメモ（Codex/Claude 2交代運用）

- 仕様書の優先順位は「落ちにくさ > 厳密性」。
- まずC0/C1で探索とデータ構造を安定化し、C2以降で接触法則を重ねる順番を崩さないこと。
- 実装コミットは **phase単位で小さく分割**し、statusに対応する検証ログを都度残すこと。
