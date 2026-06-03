[[ようこそ]] / [[ポエム]] / [[../../../README|README]]

# wiggle カンバン

撚線工程 3D CG プロジェクトの作業ボード。

## 🔥 Next（次セッションでやる）

### ★ 2ボビン巻取りの保存則 + kinematic 管理（第八航海の宿題・最優先）
- [ ] **質量保存**: twobobbin.html は供給リール無しで slack 超過→伸び発散（slack=1 で 3 周 2.2×）。
      恒久対策 = 供給リールから払い出し（巻取分の rest 長を供給して N を増やす）or XPBD。
- [ ] **kinematic 管理**: `[KINEMATIC]` タグ + HUD 凡例は導入済み（①回転θ②軸トラバース③端クランプ）。
      新規駆動を足すたび「機械駆動か?」を判定し、タグと凡例を更新する運用を維持。
- [ ] **運動量保存**（未検証）: 手回し θ を止めた後の慣性・反動が物理的か（現状 Verlet + damping=0.99）。
- [ ] **エネルギー保存**（未検証）: 摩擦散逸・PBD 数値減衰・bend relaxation のエネルギー収支を測る。
- [ ] sim 側 +6% ドリフト: 距離拘束 GS iters=12 を増やす or XPBD compliance→0 で長鎖の伸びを締める。
- [ ] 検証: `node wiggle/web/twobobbin_probe.mjs` / frames.bin の arc 全長を Python で実測（伸び＝質量）。

> 備考: 下記 0〜2（winding studio / cradle / lay plate）は第七航海の **2ボビン刷新**以前の
> strander/winding 系レガシー。軸足が `twobobbin.*` に移ったため優先度低。

### 0. winding studio: クリーン helix 化（flange スランプ残）（最優先）
- [ ] **現状**: Coulomb 摩擦 + 固定着地点 + feedScale 0.7 で胴体に巻き上がる（bodyN:flangeN
      ≈ 2.4、yMax まで climb）が、**下フランジ面への山積みが残る**（縦軸重力で
      slack 鎖が grip 前に slump）。胴体巻が多数派だが視覚的にまだ messy
- [ ] 案: ① 巻き始めの first-layer を tighter に（早期 supply を更に絞る ramp 延長）
      ② feedScale を時間で可変（巻き始めだけ強 taut）③ flange 近傍の slump 抑制
      （flange 面接触で radial 内向き nudge）④ 弱め gravity option
- [ ] 真の解は「素線を tension で常に張る」= 実機の張力制御。slack PBD の限界
- [ ] 検証: `node wiggle/web/winding_core.test.mjs`（5 gate: 安定/起動/巻取/capstan/半径分布）

### 1. クレードルの支持アーム（strander 側）
- [ ] cradle ring から各ボビンへの放射状の腕（CylinderGeometry 細棒 6 本）
- [ ] ボビンを支持枠ごと回転させる group 構造に再編
- [ ] cradle 全体が ω で回ると、ボビンが宙に浮いて見える違和感が消える

### 2. lay plate（撚りダイスの手前ガイド板）
- [ ] ダイスの少し上流に薄い円盤（穴 6 つ）を置く
- [ ] 各穴の中心が outerStrand の z_lay 直前の位置と一致するよう配置
- [ ] 穴は ExtrudeGeometry か CSG（複雑なら省略可）
- [ ] これでボビン→lay plate→ダイス の流れがちゃんと「装置っぽく」なる

## 🟡 Backlog（次々セッション以降の候補）

- [ ] HDRI 差し替え（polyhaven の `studio_small_*` あたり、`.hdr` 1 枚）
- [ ] 導糸経路をベジエに（今は bobbin → lay plate が直線）
- [ ] ボビン自体の自転（cradle 内で水平軸まわりに回る — 素線が解けていく感）
- [ ] 1+6+12=19 本構成（2 層）への拡張
- [ ] S/Z lay 切替 GUI
- [ ] 巻線の helical texture（ボビン上に巻きの溝が見える）
- [ ] postprocessing の SSR で床反射をピカピカに
- [ ] glTF export → Blender / Three.js Editor で素材詰め
- [ ] FEM 連成（micro-macro）: helix の曲率から既存梁ソルバで局所応力、色マップ
- [ ] 巻取り満タンで自動的に simT をリセット → 連続稼働ループ
- [ ] 速度ゲージ HUD（v_feed / ω_takeup / 推定生産量 m/h を画面隅に表示）

## ⏸ 凍結（やらない）

- [ ] カラフル虹色パレット（次世代に戻したくなったら復活）
- [ ] PyVista 側の積極開発（sanity check で十分）
- [ ] CLAUDE.md の MCDD 規約系の wiggle への適用
- [ ] FEM のような厳密な接触解析（視覚的接触で十分、≠FEM のやばい接触）

## ✅ Done（直近セッション分）

### 第八航海（2026-06-03）— 名づけと、破れを灯す

- [x] **Ubuntu クラッシュ診断**: 30 周 replay 確認中の本体ハングは描画/GPU でなく
      `iwlwifi` ファーム死（`cfg80211` soft lockup）と特定。43MB の WiFi 転送が引き金。
      localhost 経路は WiFi を通らず安全。frames.bin は無傷（30.17 回転）。無線本復旧は tibby
- [x] **kinematic 規約導入**: kinematic = 機械駆動のみ（①回転θ②軸トラバース③端クランプ）。
      コードに `[KINEMATIC]` タグ、HUD に機械駆動凡例。`grep [KINEMATIC]` で台本/創発の境界が一目。
      twobobbin.js / twobobbin_core.js / twobobbin_sim.mjs / replay.js に適用。travB の横ズレは
      創発でなく処方トラバースと判明（マジックではなく台本）
- [x] **質量保存の実測 + 明示**: 計測 → twobobbin.html は供給無しで slack 超過→伸び発散（3 周 2.2×）、
      sim は GS 不足 + 摩擦ロックで +6% ドリフト（1.004→1.064）。誤コメント「stretch≈1.01 で保存」を
      実測値に修正、HUD に質量保存インジケータ（緑/橙/赤）追加。**破れを隠さない**原則を確立
- [x] **未記載航海の橋渡し**: 第六（capstan/conveyor 移植, 06-01）・第七（2ボビン刷新, 06-02）は
      日誌未記載のまま走っていた旨を [[ポエム]] 第八航海に明記

### 第五航海（2026-05-31）— Coulomb 摩擦と固定着地点

- [x] **純物理コア `winding_core.js` 抽出**（THREE 非依存）: `estimateCapacity` /
      `wrappedArcLength` / `bobbinContact`（接触+摩擦+flange+自己接触+eye+guide）/
      `supplyStep` / `traverseStep`。winding.js（描画）と headless テストが**同一の真実**を
      import → `/tmp/*.mjs` コピペによる数式ドリフトを撲滅
- [x] **co-rotation stick → Coulomb 摩擦に置換**（ユーザ要望）: 接線補正を
      `μ·(法線貫入 + seating)` でクランプ。滑り限界が生まれ **capstan 式 e^(μθ) が
      emergent**（μ↑ で巻取り成立、μ=0 比 13×、μ0.8 sweet spot）。**軸方向 Coulomb** も
      追加（縦軸重力での wrap 滑落を防止）
- [x] **回転 hard-pin → 世界固定着地点モデル**: 「particle 0 をボビンと一緒に回す」と
      敷き済み素線が摩擦で後落ち→constraint と綱引きで**半周テレポート**して裂ける問題を
      解決。**ボビンが素線の下で回り、摩擦が敷き済みを引きずる**（実機の巻取り。Coulomb の
      滑りと両立）。巻き始めは下フランジ際から build
- [x] **起動バーストの真因 = flange clip テレポート**: ボビン上方を昇る free span が
      `r<FLANGE_R` に入った瞬間 `|y|` を innerY_clip へ瞬間移動（速度6006mm/s）。フランジを
      薄い円盤として `|y|<innerY+wr` だけブロックに修正 + smoothstep 供給ランプ + 供給上限を
      `ω·R·dt`（旧 1.5× 撤廃）
- [x] **feedScale 0.7（under-feed）で taut 化**: 巻取より3割少なく供給 → 鎖が張って胴体に
      張り付き flange 山積み減（bodyN:flangeN 1.6→2.4）。負帰還で安定（張力サーボの正帰還
      発散とは逆）
- [x] **入口ガイドローラ**（カピストン風プーリー、軸 x の接触シリンダ + 視覚）+
      **active-winding リング指示子** + **eye を巻線面近く(z=FLANGE_R+6)へ下げ** wander 低減
- [x] **headless 物理検証 `winding_core.test.mjs` 常設**: 5 gate（安定 / 起動鎮静 /
      巻取り / capstan / 半径分布）AND で ✅ ALL PASS。`node` 1 発で回帰検出
- [x] **`?warmup=N` URL パラメータ**: 初回描画前に N step 空回し → 巻取り進行状態へ直リンク
      （headless スクショ / デバッグ用。rAF が headless で激しく throttle される対策）
- [x] **残課題**: 下フランジ面への山積みが残る（縦軸重力 × slack 鎖の限界、Next 0 へ）

### 第四航海（2026-05-31）— 暴れ電線と摩擦という回答

- [x] **暴れ電線の真因特定**: 数値振動ではなく「際限のないスラック」。両端 pin 間
      距離一定 + `restLength` 単調成長 + 巻取退場機構なし → 60s で chord の 16 倍長
      → 物理的にフロッピー。前航海の数値対策は二次的問題しか直していなかった
- [x] **kinematic helix ハイブリッド → 撤回**（ユーザ却下、「なりゆき」精神に反する）
- [x] **capstan 摩擦**を接触に実装: 接触帯の粒子をボビン表面と co-rotate
      （前位置角 + ω·Δt へ stickK 寄せ）。保持と巻取り(take-up)を同時に生む
- [x] `physics.js` `contactProject(pos,N,fixed,prev)` に prev 追加（後方互換、
      main.js 無影響）
- [x] **供給 = 巻取（demand-driven）**: `feedScale` 廃止。巻き付いた弧長の増分だけ
      繰り出す。rate-limit（`ω·R·dt·1.5`）+ `maxStep:8` 絶対速度 cap で正帰還を断つ
- [x] 張力サーボ / 巻付増分そのまま供給は**発散する**ことを実証（教訓として記録）
- [x] GUI: 「摩擦 (capstan stick)」スライダ（既定 0.85）、`chain` iters 6→8
- [x] ヘッドレス物理検証パターン確立（`node` で StrandChain を回し暴走/安定を数値判定）
- [x] memory `wiggle-next-session` 更新、[[ポエム]] 第四航海 追記

### 第三航海（2026-05-28 〜 05-30）— パチもん物理三本柱

- [x] **有限長 supply→takeup 巻き替え**: `L_supply + L_transit + L_takeup = L_total`
      の物質保存、phase 1/2/3、s_trail で後端後退、`rewindReset` ボタン
- [x] **多層 level-wind 螺旋**（`buildLevelWindHelix`）でサプライ / 巻取りボビン胴体に
      実際の巻線見える化（軸往復 + 半径 step）
- [x] **PBD 鎖物理 `physics.js` 新設**（`StrandChain` = Verlet + 距離拘束 Gauss-Seidel）
- [x] 外周 6 本 + bundle + core すべて PBD 鎖化（kinematic helix 廃止、両端 pin で
      重力下にだらしなく垂れる）
- [x] **Bishop frame**（parallel-transport）で下流 helix wrap の安定化
- [x] **駆動張力 = rest 長短縮**: `rest_per_segment < chord/N` で boundary tension
      代用。chain type 別 gain（bundle 1.5×）
- [x] **遠心 wrap pretension** `wrapPretensionSlack(...)` + ω² scaling `OMEGA_TAUT`
      で回転に応じた taut 化
- [x] **ω-ramp damping**（`DAMPING_REST=0.992 → DAMPING_RUN=0.94` 補間）で
      高速回転時のチャタリング抑制
- [x] **パチもん EI = Laplacian 引き寄せ** `applyBending(k)` を距離拘束 Gauss-Seidel
      反復に共存。Cosserat / Timoshenko 不要で「真っ直ぐにしたがる」性質を獲得
- [x] cache buster `?v=N` を index.html + import 文に付与（python http.server で
      module キャッシュ強くなる問題の対策）
- [x] GUI: `駆動張力` / `wrap pretension k_ω` / `パチもん EI` / `素線 PBD 物理` トグル
- [x] memory に `wiggle-next-session` / `wiggle-pbd-patterns` 保存

### 第二航海（2026-05-28）

- [x] サプライボビンに**素線の巻線**（copper cylinder + flanges + body）
- [x] **撚りダイス**（LatheGeometry のファンネル形状、メタリック仕上げ）
- [x] **巻取りスプール**（core + flanges + 巻線、Z軸 90° 回転でスプール軸を X 方向に）
- [x] 回転速度倍速（omega 2π/5 → 2π/2、GUI で最大 ±8 rad/s まで）
- [x] **巻取りスプールの回転物理**（超簡易: v_feed/R_wound、解析的時間積分）
- [x] **サプライボビン距離拡大**（L_tail_in 250→450mm）
- [x] **ダイス-素線視覚接触**（DIE_BORE_R を素線束外径 + 0.5mm に絞る）
- [x] strand 端→巻取りの接続チューブ追加
- [x] カメラ FOV 32°→38°、長くなった機械に合わせ画角再調整
- [x] cradle ring（torus）追加で「ボビン群がドーナツ状に配置」を視覚化
- [x] 巻取りが見えなかった問題のデバッグ（背景壁が物理的オクルーダーだった）

### 第一航海（2026-05-22〜27）

- [x] xkep-cae 方向転換の合意形成（FEM 厳密 → 撚線 3D CG）
- [x] `wiggle/` パッケージ作成（kinematics + render + demo）
- [x] PyVista 1+6 撚線の MP4 / PNG 出力（headless）
- [x] 撚りほどき → 撚り工程の符号バグ修正（imprint 位相の物理導出 docstring 付き）
- [x] three.js 版立ち上げ（importmap で CDN、ビルド不要）
- [x] `MeshPhysicalMaterial` + bloom + RoomEnvironment + 3-spot lighting + shadow
- [x] lil-gui で L_pitch / ω / 本数 / R_layer / 速度 / bloom / 露出 を runtime 調整
- [x] chromium headless での自動スクショパス確立（`--enable-unsafe-swiftshader`）
- [x] http.server (port 8780) でローカル配信
- [x] memory に方向転換と「厳格規約は外す」を保存
- [x] 単位を mm 固定に（StranderConfig 全パラメータ）
- [x] 銅色単色化（COPPER_BASE = 0xb87333、素線ごとに ±5% lightness）
- [x] OrbitControls + autoRotate + 視点プリセット（俯瞰/側面/ダイス近景/先端ドリーイン）
- [x] 時間スライダー（GUI で 0〜120s スクラブ可能）
- [x] GUI フォルダ構造化（機械パラメータ / 再生 / ビジュアル / カメラ）

## 環境メモ

- 開発機: Linux, Python 3.10+, uv で管理
- ブラウザ確認: `python -m http.server 8780` を `/home/nishioka/work/xkep-cae` から起動、
  `http://localhost:8780/wiggle/web/index.html`
- スクショ: `chromium --headless=new --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader`
  でないと WebGL コンテキストが取れない
- 依存: `pyvista 0.48`, `imageio[ffmpeg]`, three.js 0.170 (CDN), lil-gui 0.19 (CDN)

## ファイルマップ

```
wiggle/
  __init__.py
  kinematics.py     # Python 版運動学（L_tail_in=450、符号バグ修正済み）
  render.py         # PyVista レンダラ
  demo.py           # 1+6 PNG/MP4 出力デモ
  web/
    index.html      # strander studio: importmap + HUD
    main.js         # strander: シーン構築 + 毎フレーム TubeGeometry + 巻取り回転物理
    kinematics.js   # Python 版と完全対応 + buildLevelWindHelix / Bishop frame 等
    physics.js      # StrandChain（Verlet + 距離拘束 GS + bending + contactProject(…,prev)）
    winding.html    # winding studio（サプライボビン製造工程）
    winding.js      # winding: 描画 + GUI。物理は winding_core を import。?warmup=N 対応
    winding_core.js # 純物理コア（THREE 非依存）: 接触+Coulomb摩擦+supply+traverse。
                    #   winding.js とテストが共有する単一の真実
    winding_core.test.mjs  # headless 物理検証（node 実行、5 gate AND）
```

```
results/wiggle/
  strander_1plus6_t0.png    # PyVista t=0
  strander_1plus6_t3.png    # PyVista t=3
  strander_1plus6.mp4       # PyVista 6 秒アニメ
  web_t0.png  web_t3.png  web3.png  # three.js 初期版
  v3_wall_fix.png            # 巻取り visibility 修正版（背景壁を後退）
  v4_physics.png             # 回転物理 + ダイス接触 + L_tail_in=450 版
  v4_reframe.png             # FOV 38° + カメラ再フレーミング
  v4_takeup_big.png          # 巻取り拡大 + 初期 simT=10
```

## デバッグ Tips（第二航海で得た）

- **「見えないオブジェクト」の特定順序**: ① シーンに add してるか →
  ② カメラ frustum 内か → ③ **手前に何か遮蔽してないか** ← 今回ここで詰まった →
  ④ マテリアルが暗背景に溶け込んでないか → ⑤ scale や opacity が 0 になってないか
- **wireframe デバッグ**: `mat.wireframe = true` を一時的に GUI トグルにすると
  オクルーダー特定が一瞬で済む。次セッションで実装したい
