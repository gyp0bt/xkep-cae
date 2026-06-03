// wiggle — winding studio
// ボビンに素線を巻き付けて「サプライボビン」を製造する工程 CG。
// 物理: 素線は PBD chain（重力 + 距離拘束 + bending + 接触 + capstan 摩擦）。
//   形状は一切指定しない（kinematic helix なし・freeze なし）= 純なりゆき。
//   巻き付き・level-wind 模様は 摩擦 + 接触 + 重力 から emergent に決まる。
// 座標系: 機械座標 +y が上、ボビン軸 = +y 鉛直、bobbin 中心 = 原点
// 単位: mm / rad / s

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import GUI from "lil-gui";

// 親 HTML が ?v=Date.now() を付けて winding.js を呼んでくれる前提で、
// import.meta.url の search を local module 動的 import に引き継ぎ全ファイル毎回 fresh
const _ver = new URL(import.meta.url).search;
const { buildLevelWindHelix } = await import("./kinematics.js" + _ver);
const { StrandChain } = await import("./physics.js" + _ver);
const {
  estimateCapacity, wrappedArcLength,
  bobbinContact, supplyStep, traverseStep,
} = await import("./winding_core.js" + _ver);

// ---------- 寸法 ----------
const COPPER = 0xb87333;
const cfg = {
  BODY_R: 40,        // mm — 胴体半径
  BODY_H: 140,       // mm — 胴体長
  FLANGE_R: 78,      // mm — フランジ外径
  FLANGE_H: 8,       // mm — フランジ厚
  WIND_R_IN: 41,     // mm — 巻線内径（胴体表面より僅か外）
  WIND_R_OUT: 73,    // mm — 巻線最外径（フランジ内側より僅か内）
  WIND_H: 120,       // mm — 巻線可用軸長
  wire_r: 1.8,       // mm — 素線半径
  omega: 1.5,        // rad/s — ボビン自転
  L_capacity: 0,     // mm — 計算後設定
};

cfg.L_capacity = estimateCapacity(cfg);

// ---------- Renderer / Scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
scene.fog = new THREE.Fog(0x0b0d10, 500, 1800);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.035).texture;

// ---------- Camera ----------
const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 1, 5000);
camera.position.set(220, 110, 240);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

// ---------- Lights ----------
scene.add(new THREE.AmbientLight(0xffffff, 0.18));

const key = new THREE.SpotLight(0xfff2dd, 12000, 1000, Math.PI / 5, 0.35, 1.4);
key.position.set(-180, 320, 200);
key.target.position.set(0, 0, 0);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.bias = -0.002;
scene.add(key, key.target);

const fill = new THREE.SpotLight(0x88aacc, 4500, 1000, Math.PI / 4, 0.6, 1.2);
fill.position.set(280, 160, -120);
fill.target.position.set(0, 0, 0);
scene.add(fill, fill.target);

const rim = new THREE.DirectionalLight(0xffe7cf, 1.2);
rim.position.set(50, 80, -240);
scene.add(rim);

// ---------- Floor ----------
const floorGeo = new THREE.PlaneGeometry(1200, 1200);
const floorMat = new THREE.MeshPhysicalMaterial({
  color: 0x12141a, metalness: 0.25, roughness: 0.42,
  clearcoat: 0.3, clearcoatRoughness: 0.6,
});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -(cfg.FLANGE_R + 24);
floor.receiveShadow = true;
scene.add(floor);

// ---------- Materials ----------
const copperMat = new THREE.MeshPhysicalMaterial({ color: COPPER, metalness: 0.95, roughness: 0.18 });
const bobbinBodyMat = new THREE.MeshPhysicalMaterial({ color: 0x3b2a1d, metalness: 0.15, roughness: 0.65 });
const bobbinFlangeMat = new THREE.MeshPhysicalMaterial({ color: 0x1f1610, metalness: 0.55, roughness: 0.4 });
const frameMat = new THREE.MeshPhysicalMaterial({ color: 0x5a6068, metalness: 0.8, roughness: 0.28 });
const guideMat = new THREE.MeshPhysicalMaterial({ color: 0x7a8088, metalness: 0.85, roughness: 0.25 });

// ---------- Bobbin (spin around +y) ----------
const bobbinGroup = new THREE.Group();
scene.add(bobbinGroup);

const bobbinSpin = new THREE.Group();
bobbinGroup.add(bobbinSpin);

const body = new THREE.Mesh(
  new THREE.CylinderGeometry(cfg.BODY_R, cfg.BODY_R, cfg.BODY_H, 36),
  bobbinBodyMat,
);
body.castShadow = true; body.receiveShadow = true;
bobbinSpin.add(body);

const flA = new THREE.Mesh(
  new THREE.CylinderGeometry(cfg.FLANGE_R, cfg.FLANGE_R, cfg.FLANGE_H, 40),
  bobbinFlangeMat,
);
flA.position.y = -(cfg.BODY_H - cfg.FLANGE_H) / 2;
flA.castShadow = true; flA.receiveShadow = true;
bobbinSpin.add(flA);

const flB = new THREE.Mesh(
  new THREE.CylinderGeometry(cfg.FLANGE_R, cfg.FLANGE_R, cfg.FLANGE_H, 40),
  bobbinFlangeMat,
);
flB.position.y = +(cfg.BODY_H - cfg.FLANGE_H) / 2;
flB.castShadow = true; flB.receiveShadow = true;
bobbinSpin.add(flB);

// 巻線済み銅の視覚は PBD chain そのもの（procedural mesh は持たない）。
// bobbinSpin に body + flange だけが入る。chain は world 系で render される。

// ---------- Bobbin support frame (両側スピンドル軸受) ----------
const spindleGeo = new THREE.CylinderGeometry(8, 8, cfg.BODY_H * 1.4, 16);
const spindle = new THREE.Mesh(spindleGeo, frameMat);
spindle.castShadow = true;
scene.add(spindle);

const bearingA = new THREE.Mesh(new THREE.TorusGeometry(14, 4, 10, 24), frameMat);
bearingA.rotation.x = Math.PI / 2;
bearingA.position.y = cfg.BODY_H * 0.65;
bearingA.castShadow = true;
scene.add(bearingA);

const bearingB = new THREE.Mesh(new THREE.TorusGeometry(14, 4, 10, 24), frameMat);
bearingB.rotation.x = Math.PI / 2;
bearingB.position.y = -cfg.BODY_H * 0.65;
bearingB.castShadow = true;
scene.add(bearingB);

// 横腕の支持柱（軸受台）— 2 本
const postH = cfg.BODY_H + 80;
for (const xz of [[-cfg.FLANGE_R - 30, 0], [+cfg.FLANGE_R + 30, 0]]) {
  const postGeo = new THREE.BoxGeometry(14, postH, 22);
  const post = new THREE.Mesh(postGeo, frameMat);
  post.position.set(xz[0], -cfg.FLANGE_R - 24 + postH / 2 - 30, xz[1]);
  post.castShadow = true;
  scene.add(post);
}

// ---------- Level-wind traverse guide ----------
// 巻線進行位置 y_t に追従する横スライダ + 素線通し穴 (eye)
const traverseGroup = new THREE.Group();
scene.add(traverseGroup);

// レール (固定): bobbin の手前 +z 側に水平
const railGeo = new THREE.CylinderGeometry(3, 3, cfg.WIND_H + 40, 12);
const rail = new THREE.Mesh(railGeo, frameMat);
rail.position.set(0, 0, cfg.FLANGE_R + 36);
rail.castShadow = true;
scene.add(rail);

// スライダ本体
const sliderBody = new THREE.Mesh(
  new THREE.BoxGeometry(28, 18, 22), guideMat,
);
sliderBody.castShadow = true;
traverseGroup.add(sliderBody);

// アイ (素線通し穴) — 小さなリング、bobbin 寄り
// torus 既定軸 = z。素線は source(y=232) → bobbin(y=0) と下降して通るので
// リング軸 = y にして水平の穴にする → rotation.x = π/2 で z→y 回転
const eye = new THREE.Mesh(
  new THREE.TorusGeometry(3.6, 0.8, 8, 24),
  new THREE.MeshPhysicalMaterial({ color: 0xc8c8d0, metalness: 0.95, roughness: 0.15 }),
);
eye.rotation.x = Math.PI / 2;
// eye world z = FLANGE_R + 6（巻線面近くまで下げ、着地点までの free span を短くして
// wander を抑える）。traverseGroup は z=FLANGE_R+36 なので local z = -30。
const EYE_WORLD_Z = cfg.FLANGE_R + 6;
const EYE_LOCAL_Z = EYE_WORLD_Z - (cfg.FLANGE_R + 36);   // = -30
eye.position.set(0, 0, EYE_LOCAL_Z);
eye.castShadow = true;
traverseGroup.add(eye);
const EYE_R_INNER = 3.6 - 0.8;   // ring 穴半径（torus.radius - torus.tube）

// スライダから bobbin 寄りに突き出す細いアーム
const armLen = -EYE_LOCAL_Z;
const armGeo = new THREE.CylinderGeometry(1.6, 1.6, armLen, 12);
const arm = new THREE.Mesh(armGeo, guideMat);
arm.rotation.x = Math.PI / 2;
arm.position.set(0, 0, EYE_LOCAL_Z / 2);
arm.castShadow = true;
traverseGroup.add(arm);

traverseGroup.position.set(0, 0, cfg.FLANGE_R + 36);

// ---------- Strand feed (上流素線リール) ----------
// 簡素な「供給リール」: アイの上方斜め後ろから素線を引き出す体
const FEED_POS = new THREE.Vector3(0, 250, 120);

const feedReelGroup = new THREE.Group();
feedReelGroup.position.copy(FEED_POS);
scene.add(feedReelGroup);

const feedAxis = new THREE.Mesh(
  new THREE.CylinderGeometry(3, 3, 50, 16), frameMat,
);
feedAxis.rotation.z = Math.PI / 2;
feedReelGroup.add(feedAxis);

const feedReelBody = new THREE.Mesh(
  new THREE.CylinderGeometry(22, 22, 28, 24), bobbinBodyMat,
);
feedReelBody.rotation.z = Math.PI / 2;
feedAxis.castShadow = true;
feedReelBody.castShadow = true;
feedReelGroup.add(feedReelBody);

const feedReelFlangeMat = new THREE.MeshPhysicalMaterial({ color: 0x252028, metalness: 0.55, roughness: 0.4 });
for (const dx of [-15, +15]) {
  const fl = new THREE.Mesh(
    new THREE.CylinderGeometry(32, 32, 3, 24), feedReelFlangeMat,
  );
  fl.rotation.z = Math.PI / 2;
  fl.position.x = dx;
  fl.castShadow = true;
  feedReelGroup.add(fl);
}

// 供給リールにもざっくり銅が見える tube（procedural）
const feedReelWound = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
feedReelGroup.add(feedReelWound);
(function fillFeedReelVisual() {
  // 適当に多周巻きの helix を一回だけ生成（演出用）
  const pts = buildLevelWindHelix(2000, 23, 30, 24, cfg.wire_r * 2, { segs_per_turn: 22 });
  if (pts.length < 4) return;
  const vs = pts.map(p => new THREE.Vector3(p.y, p.z, p.x));  // local rotation (axis = +x で配置)
  const curve = new THREE.CatmullRomCurve3(vs, false, "catmullrom", 0.0);
  feedReelWound.geometry.dispose();
  feedReelWound.geometry = new THREE.TubeGeometry(curve, 600, cfg.wire_r, 10, false);
})();

// ---------- 入口ガイドローラ（カピストン風プーリー） ----------
// feed reel → eye の経路に置く水平ローラ（軸 = x）。素線がここで一度曲がって
// 巻取りへ入る「装置っぽさ」を出す + 自由スパンの経路を安定化する。
const GUIDE = { x: 0, y: 150, z: 100, r: 10, hx: 20 };
const guideGroup = new THREE.Group();
guideGroup.position.set(GUIDE.x, GUIDE.y, GUIDE.z);
scene.add(guideGroup);

const guideRoller = new THREE.Mesh(
  new THREE.CylinderGeometry(GUIDE.r, GUIDE.r, GUIDE.hx * 2, 24),
  new THREE.MeshPhysicalMaterial({ color: 0x9aa0a8, metalness: 0.9, roughness: 0.2 }),
);
guideRoller.rotation.z = Math.PI / 2;   // 軸を x へ
guideRoller.castShadow = true;
guideGroup.add(guideRoller);
for (const dx of [-GUIDE.hx - 1.5, GUIDE.hx + 1.5]) {
  const fl = new THREE.Mesh(
    new THREE.CylinderGeometry(GUIDE.r + 3, GUIDE.r + 3, 3, 24), guideMat,
  );
  fl.rotation.z = Math.PI / 2;
  fl.position.x = dx;
  fl.castShadow = true;
  guideGroup.add(fl);
}
// ガイド軸受柱
const guidePost = new THREE.Mesh(new THREE.BoxGeometry(8, GUIDE.y * 0.5, 14), frameMat);
guidePost.position.set(GUIDE.hx + 8, GUIDE.y * 0.75, GUIDE.z);
guidePost.castShadow = true;
scene.add(guidePost);

// ---------- アクティブ巻き位置インジケータ ----------
// 現在巻かれている軸位置 y=traverse_y を示す細いリング（巻線面のすぐ外）。
const activeRing = new THREE.Mesh(
  new THREE.TorusGeometry(cfg.BODY_R + cfg.wire_r + 1.5, 0.5, 8, 48),
  new THREE.MeshBasicMaterial({ color: 0xffcf88 }),
);
activeRing.rotation.x = Math.PI / 2;   // xz 平面（軸 = y）
scene.add(activeRing);

// ---------- 素線 (PBD chain): 物理巻き取り + 強いガイド拘束 ----------
// 純物理 PBD（重力 + 距離拘束 + bending + 接触 + co-rotation 摩擦）で巻く。neat さは
// **強いガイド拘束**で各巻きを精密配置して得る（実機:「機械が wire を置くが wire は物理で従う」）:
//   ① アンカー(particle 0) が **ヘリックス経路をなぞる**: (R·sinθ, traverse_y, R·cosθ)。
//      θ=spin・y=traverse が進むとアンカーがらせんを描き、wire を 1 巻きずつ積む。
//   ② **laid-height pin**: 接触帯に入った瞬間の traverse 高さを刻み、以降その高さへ強拘束
//      → 各巻きが着地高さに固定され隣接して並ぶ（重力でばらけない）。
//   ③ co-rotation 摩擦（周方向 no-slip）が表面と一緒に回す。
// headless 検証: pin=1.0/stick=0.9 → gripped 187/200・単調 98%・大ジャンプ 1・
//   ピッチ≈wire 径（54mm/14 巻き）・半径 47.8≈R(41.8)。
const CHAIN_N = 200;
const R_lay = cfg.BODY_R + cfg.wire_r;
const sourcePin = [FEED_POS.x, FEED_POS.y, FEED_POS.z];
const TRAV_LO = -((cfg.BODY_H - 2 * cfg.FLANGE_H) / 2 - cfg.wire_r);  // 下フランジ際
const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const bandR = R_lay + cfg.wire_r * 4;     // 接触帯（laid-pin 判定）

// ヘリックスをなぞるアンカー（lay point）= 巻き取りの prime mover + 強いガイド。
function anchorAt(spinTheta, y) {
  const cs = Math.cos(spinTheta), sn = Math.sin(spinTheta);
  return [R_lay * sn, y, R_lay * cs];
}
const pin0Init = anchorAt(0, TRAV_LO);
const initChord = Math.hypot(
  sourcePin[0] - pin0Init[0], sourcePin[1] - pin0Init[1], sourcePin[2] - pin0Init[2],
);
const chain = new StrandChain(CHAIN_N, initChord / (CHAIN_N - 1), { damping: 0.985, iters: 8 });
chain.bendK = 0.25;
chain.initLine(pin0Init, sourcePin);
const laidY = new Float64Array(CHAIN_N).fill(NaN);   // 各粒子の着地高さ（NaN=未着地）

const chainMesh = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
chainMesh.castShadow = true; chainMesh.receiveShadow = true;
scene.add(chainMesh);

// ---------- 巻取状態 ----------
const state = {
  bobbinSpinTheta: 0,
  restGrow: 0,
  L_supplied: initChord,
  wrappedPrev: 0,
  initChord,
  turns: 0,
  bobbinCount: 0,
  paused: false,
  traverse_y: TRAV_LO,
  traverse_dir: +1,
};

// ---------- Helpers ----------
function tubeFromPoints(mesh, pts, radius, segs, radial) {
  if (pts.length < 4) { mesh.visible = false; return; }
  mesh.visible = true;
  const vs = pts.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const curve = new THREE.CatmullRomCurve3(vs, false, "centripetal", 0.5);
  mesh.geometry.dispose();
  mesh.geometry = new THREE.TubeGeometry(curve, segs, radius, radial, false);
}

// ---------- 接触判定（core 集約：接触 + co-rotation 摩擦 + flange + 自己接触 + eye + guide）----------
const contactCtx = {
  cfg,
  legacyStick: 0.9,   // co-rotation 食いつき（GUI 更新）
  frictionDtheta: 0,
  eye: { y: TRAV_LO, z: EYE_WORLD_Z, rInner: EYE_R_INNER },
  guide: GUIDE,
};
function contactProject(pos, N, fixed, prev) {
  bobbinContact(pos, N, fixed, prev, contactCtx);
}

function resetWind() {
  state.bobbinSpinTheta = 0;
  state.restGrow = 0;
  state.L_supplied = initChord;
  state.wrappedPrev = 0;
  state.turns = 0;
  state.traverse_y = TRAV_LO;
  state.traverse_dir = +1;
  laidY.fill(NaN);
  chain.restLength = initChord / (CHAIN_N - 1);
  chain.initLine(anchorAt(0, TRAV_LO), sourcePin);
}

// ---------- 動的更新 ----------
function updateScene(dt) {
  if (state.paused || dt <= 0) return;

  // (a) 自転 + 強いガイド①: アンカーがヘリックスをなぞる（angle=spin, y=traverse）
  state.bobbinSpinTheta += cfg.omega * dt;
  bobbinSpin.rotation.y = state.bobbinSpinTheta;
  state.turns = Math.abs(state.bobbinSpinTheta) / (2 * Math.PI);
  const pin0World = anchorAt(state.bobbinSpinTheta, state.traverse_y);
  chain.restLength = state.L_supplied / (CHAIN_N - 1);

  // (b) PBD substeps（接触 + co-rotation 摩擦）
  const SUBSTEPS = 4, subDt = dt / SUBSTEPS;
  contactCtx.legacyStick = params.stickK;
  contactCtx.frictionDtheta = cfg.omega * subDt;   // 1 substep の表面回転角
  contactCtx.eye.y = state.traverse_y;
  for (let s = 0; s < SUBSTEPS; s++) {
    chain.step(subDt, pin0World, sourcePin, { contactProject, bendIters: 3, maxStep: 8 });
  }

  // (c) 強いガイド②: laid-height pin。接触帯に入った瞬間の traverse 高さを刻み、
  //     以降その高さへ params.layPin で強拘束 → 各巻きが着地高さに固定され隣接する。
  const ps = params.layPin;
  for (let i = 1; i < CHAIN_N; i++) {
    const k = 3 * i;
    const r = Math.hypot(chain.pos[k], chain.pos[k + 2]);
    if (r <= bandR && Math.abs(chain.pos[k + 1]) < innerY) {
      if (Number.isNaN(laidY[i])) laidY[i] = state.traverse_y;
      chain.pos[k + 1] += (laidY[i] - chain.pos[k + 1]) * ps;
    }
  }

  // (d) demand-driven 供給（バックテンションで張る）+ 描画 + traverse
  const wrapped = wrappedArcLength(chain.pos, CHAIN_N, cfg);
  supplyStep(state, wrapped, dt, cfg, { feedScale: params.backTension });
  tubeFromPoints(chainMesh, chain.toPoints(), cfg.wire_r, Math.max(120, CHAIN_N * 3), 10);
  traverseStep(state, dt, cfg);
  traverseGroup.position.y = state.traverse_y;
  activeRing.position.y = state.traverse_y;
}

// ---------- HUD ----------
const statusEl = document.getElementById("status");
function updateHUD(t) {
  const turns = state.turns.toFixed(2);
  const phase = "ガイド拘束 物理巻取";
  const fillPct = (state.L_supplied / cfg.L_capacity * 100).toFixed(0);
  statusEl.innerHTML =
    `<span class="phase">${phase}</span> · 巻数 <b>${turns}</b> · ` +
    `巻線長 ${state.L_supplied.toFixed(0)} mm (${fillPct}%) · ω=${cfg.omega.toFixed(2)} rad/s<br>` +
    `<span class="count">完成本数 ${state.bobbinCount}</span> · chain N=${CHAIN_N} · t=${t.toFixed(1)}s`;
}

// ---------- GUI ----------
const params = {
  omega: cfg.omega,
  wire_r: cfg.wire_r,
  stickK: 0.9,        // co-rotation 食いつき（周方向 no-slip）
  backTension: 0.88,  // 供給/巻取 比（<1 で張る）
  layPin: 1.0,        // laid-height pin 強さ（強いガイド拘束。1=その高さに即固定）
  paused: false,
  speed: 1.0,
  ejectBobbin() {
    // 完成本数 +1、巻き取りをリセットして次のボビン開始
    state.bobbinCount += 1;
    resetWind();
  },
  resetAll() {
    state.bobbinCount = 0;
    resetWind();
  },
  resetCamera() {
    controls.target.set(0, 0, 0);
    camera.position.set(220, 110, 240);
    controls.update();
  },
  viewSide() {
    controls.target.set(0, 0, 0);
    camera.position.set(0, 30, 320);
    controls.update();
  },
  viewTop() {
    controls.target.set(0, 0, 0);
    camera.position.set(0, 360, 0.01);
    controls.update();
  },
  viewContact() {
    controls.target.set(0, 0, cfg.WIND_R_OUT);
    camera.position.set(80, 60, 180);
    controls.update();
  },
};

const gui = new GUI({ title: "winding studio" });
gui.add(params, "ejectBobbin").name("▶ 完成 → 次のボビン");
gui.add(params, "resetAll").name("⟲ 全リセット");

const fMachine = gui.addFolder("機械");
fMachine.add(params, "omega", -4, 6, 0.05).name("ω [rad/s]").onChange(v => { cfg.omega = v; });
fMachine.add(params, "wire_r", 0.4, 4, 0.05).name("素線半径 [mm]").onChange(v => {
  cfg.wire_r = v;
  cfg.L_capacity = estimateCapacity(cfg);
});
fMachine.open();

const fGuide = gui.addFolder("ガイド拘束 / 物性");
fGuide.add(params, "layPin", 0, 1, 0.05).name("巻き固定 (laid-pin)");
fGuide.add(params, "stickK", 0, 1, 0.02).name("食いつき (co-rotation)");
fGuide.add(params, "backTension", 0.6, 1.0, 0.01).name("バックテンション");
fGuide.add({ bendK: chain.bendK }, "bendK", 0, 0.6, 0.01)
  .name("EI (パチもん)").onChange(v => { chain.bendK = v; });
fGuide.add({ damping: chain.damping }, "damping", 0.9, 0.999, 0.001)
  .name("減衰").onChange(v => { chain.damping = v; });
fGuide.open();

const fPlay = gui.addFolder("再生");
fPlay.add(params, "paused").name("一時停止").onChange(v => { state.paused = v; });
fPlay.add(params, "speed", 0, 5, 0.1).name("速度");

const fCam = gui.addFolder("カメラ");
fCam.add(params, "resetCamera").name("初期位置");
fCam.add(params, "viewSide").name("側面 (接触面)");
fCam.add(params, "viewTop").name("俯瞰");
fCam.add(params, "viewContact").name("接触点ドリーイン");

// ---------- Main loop ----------
let last = performance.now();
let simT = 0;

// ?warmup=N : 初回描画前に固定 dt(1/60) で N step 物理を空回し、巻取りが進んだ状態へ
// 直リンク（headless スクショ / デバッグ用。rAF は headless で激しく throttle される）。
const _warmup = parseInt(new URL(location.href).searchParams.get("warmup") || "0", 10);
if (_warmup > 0) {
  for (let i = 0; i < _warmup; i++) { updateScene(1 / 60); simT += 1 / 60; }
}

// 固定タイムステップ累積。物理は **常に FIXED_DT=1/60** で進める。
// 理由: maxStep など一部の項が dt 非依存で、可変 dt（低 fps）だと素線が回転 pin に
// 追従できず**ドラムから離脱して全く巻かない**（30fps で gripped=1, wrapped=0 を再現）。
// 壁時計の経過を貯めて 1/60 刻みで N 回 step → ライブ = headless テスト = warmup が一致。
const FIXED_DT = 1 / 60;
let acc = 0;
function tick() {
  const now = performance.now();
  const wallDt = Math.min(0.1, (now - last) / 1000);   // 0.1s 上限で spiral-of-death 回避
  last = now;
  if (!state.paused) {
    acc += wallDt * params.speed;
    let n = 0;
    while (acc >= FIXED_DT && n < 8) {   // 1 フレーム最大 8 sub（取りこぼしは捨てる）
      updateScene(FIXED_DT);
      simT += FIXED_DT;
      acc -= FIXED_DT;
      n++;
    }
  }
  updateHUD(simT);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

// ---------- Resize ----------
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
