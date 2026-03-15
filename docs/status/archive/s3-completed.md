# S3: 大規模収束改善 — 完了済み項目

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← status-index](status-index.md)

> S3フェーズで完了した全項目の詳細記録。roadmap.md からの分離（status-149）。

## 完了済みマイルストーン

| カテゴリ | 項目 | status |
|---------|------|--------|
| **収束達成** | 7本 NCP収束（adaptive omega + λ_nキャッピング） | 097 |
| | 19本 NCP収束（径方向圧縮、24ペアアクティブ） | 112 |
| | 37本 NCP収束（Layer1径方向圧縮） | 121 |
| | 7本90°曲げ+揺動1周期 完全収束（130秒） | 132 |
| | 19本曲げ揺動収束（45°+90°+揺動） | 135 |
| | 7本NCP曲げ揺動 接触あり収束（Point contact + mesh_gap） | 143 |
| | 摩擦あり（μ=0.1）45°曲げ収束 | 143 |
| | smooth penalty+Uzawaで摩擦90°曲げ+揺動収束 | 147 |
| **ソルバー改善** | NCP収束安定化（line search/MNR/接線予測子/エネルギー収束） | 103 |
| | ILU drop_tol 適応制御 | 107 |
| | Schur正則化改善（対角最大値ベース） | 107 |
| | GMRES restart適応 | 107 |
| | λウォームスタート | 107 |
| | Active setチャタリング抑制（過半数投票） | 107 |
| | 適応時間増分制御 | 109 |
| | AMG前処理（PyAMG SA） | 109 |
| | k_pen continuation | 109 |
| | k_pen自動推定（beam EI ベース） | 109 |
| | omega回復メカニズム | 109 |
| | 残差スケーリング | 110 |
| | 接触力ランプ | 110 |
| | n_load_steps=1対応 + 安定化成長戦略 | 133 |
| **高速化** | NCP 6x高速化（解析的接線+バッチ接触幾何、49.6s→8.3s） | 132 |
| | 要素ループバッチ化（12.6x高速化） | 134 |
| **アーキテクチャ** | ソルバー一本化（n_load_steps廃止、NCPSolverInput導入） | 134 |
| | ステップ二分法deprecated化（adaptive_timestepping統合） | 110 |
| | レガシーテストdeprecated化（旧ソルバー5ファイル） | 107,116 |
| | NCPソルバー段階的活性化移植 | 126 |
| | UL+NCP統合（adaptive_timestepping連動） | 131 |
| **梁要素** | CR梁収束問題の根本原因診断（寄生軸力 EA/(EI/L²)≈9000） | 129 |
| | Updated Lagrangian CR梁アセンブラ（~13°障壁解消） | 130 |
| | CR vs Cosserat比較（物理・収束性・コスト定量） | 117 |
| | Generalized-α法（Chung-Hulbert 1993、14テスト） | 117 |
| **接触モデル** | NCP版摩擦バリデーション16件移行 | 121 |
| | NCP版ヒステリシス9件移行 | 121 |
| | NCP摩擦接触の行列特異化修正（J_t_t正則化） | 128 |
| | Mortarギャップ計算バグ修正（pair.state.radius→pair.radius） | 142 |
| | 撚線メッシュ初期貫入の解消（mesh_gap方式） | 143 |
| **被膜** | gap_offset手法廃止→被膜厚考慮メッシュ+被膜スプリング | 137 |
| | 被膜接線剛性実装（k=1e6完全収束） | 139 |
| | mm-ton-MPa移行 + Kelvin-Voigt粘性減衰 + k_pen材料ベース強制 | 140 |
| | 被膜Coulomb摩擦モデル実装 + 摩擦core関数抽出 | 141 |
| **可視化** | 接触診断2D投影可視化（四元数回転） | 123 |
| | 変形前後3Dレンダリング | 123 |
| | 3Dプロット2D投影完全移行（mplot3d廃止） | 126 |
| | 3Dチューブレンダリング（円形断面付き表面描画） | 144 |
| **チューニング** | TuningTaskスキーマ + 検証プロット6種 | 114,115 |
| | チューニング実行エンジン + Optuna連携 | 115 |
| **テスト** | 動的解析物理テスト13件 | 117 |
| | 応力・曲率連続性テスト11件 | 115 |
| | NCP曲げ揺動テスト8件 | 126 |
| | メッシュ非貫入制約 | 104 |

## 完了済みTODO

- [x] 7本NCP曲げ揺動のCI確認 → xfailで安定化（status-127）
- [x] NCP摩擦接触の行列特異化修正（status-128）
- [x] CR梁ヘリカル要素の収束問題の根本原因診断（status-129）
- [x] Updated Lagrangian実装で7本45°/90°曲げ収束達成（status-130）
- [x] UL+NCP統合: adaptive_timesteppingとUL参照更新の一体化（status-131）
- [x] UL Phase 2（揺動）の特異行列問題修正（status-132）
- [x] NCP 6x高速化: 解析的接線剛性+バッチ接触幾何（status-132）
- [x] 非線形接触動解析ソルバーモジュール完全一本化（status-134）
- [x] 要素ループのベクトル化（12.6x高速化、status-134）
- [x] 19本撚線の曲げ揺動収束確認（status-135）
- [x] gap_offset手法の廃止と被膜接触モデルの再構築（status-137）
- [x] 被膜接線剛性実装 + 6DOFバグ修正 + 収束検証（status-139）
- [x] mm-ton-MPa移行 + Kelvin-Voigt粘性減衰 + k_pen材料ベース強制（status-140）
- [x] 被膜Coulomb摩擦モデル実装 + 摩擦core関数抽出（status-141）
- [x] Mortarギャップ計算バグ修正（status-142）
- [x] 撚線メッシュ初期貫入の解消（status-143）
- [x] 7本NCP曲げ揺動の接触あり収束達成（status-143）
- [x] 摩擦あり（μ=0.1）曲げ揺動収束検証（status-143）

---
