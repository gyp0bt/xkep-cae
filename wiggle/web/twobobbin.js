// wiggle — 2ボビン手動巻取り（焼き付け方式を破棄した刷新版）
// 問い: 電線を各ボビン表面の 1 点に留め、ユーザが両ボビンを手で回したとき、
//   PBD + 接触 + Coulomb 摩擦だけで素直に巻き取り／払い出しできるか？ → YES（twobobbin_probe.mjs で確定）。
//
// 物理は twobobbin_core.js（headless probe と共有）:
//   - 電線 = 1 本の連続 PBD 鎖（固定長 N）。焼き付け・凍結パック・座標変換は一切なし。
//   - 巻き済みは「摩擦で表面に貼り付いた粒子」が創発的にボビンと一緒に回る → 継ぎ目・ラグが出ない。
//   - 両端を各ボビン表面の 1 点にピン（co-rotation）。GUI スライダで θ_A / θ_B を手動回転。
//   - 重力で弛みループが垂れ、回すとボビンへ巻き取られる。
// 座標系: +y 上、ボビン軸 = +z（水平・横並び）、gravity = -y。単位 mm / rad / s。
//
// ── kinematic 規約（wiggle 共通）──────────────────────────────────────────
//   [KINEMATIC] = 実機の機械が駆動する箇所のみ。座標/角度を物理応答ではなく
//   処方（直接代入）で与える入力境界。grep "[KINEMATIC]" で全駆動点を列挙可。
//   ここでの kinematic は 3 つだけ: ①ボビン回転θ ②軸トラバース ③端クランプ。
//   それ以外（弛み・巻き保持・横並びパッキング）は全て twobobbin_core.js の
//   創発物理（PBD+接触+Coulomb 摩擦+自己接触）。新規追加時もこの分界を厳守。
// ─────────────────────────────────────────────────────────────────────────
//
// ⚠ 質量保存の構造的限界（供給リール無し）───────────────────────────────────
//   電線は固定長 N の鎖で両端をピン固定。**払い出し供給が無い**ため、初期弛み
//   slackTurns を巻き尽くすと「巻く長さ」を調達できず PBD が伸びて辻褄合わせ。
//   実測（ωB=1t/s, slack=1）: 0.5 周 1.12× → 2 周 1.58× → 3 周 2.20×（発散・振動）。
//   slack=3 でも 3 周超で発散。→ 質量保存は「slackTurns 周以内」でのみ近似成立。
//   伸びは HUD に常時表示し、閾値超で赤警告（破れを隠さない方針）。
//   恒久対策（未実装）= 供給リール払い出し or 巻取分の rest 長供給で N を増やす。
// ─────────────────────────────────────────────────────────────────────────

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import GUI from "lil-gui";

const _ver = new URL(import.meta.url).search;
const { StrandChain } = await import("./physics.js" + _ver);
const { defaultCfg2, surfacePoint, twoBobbinContact } = await import("./twobobbin_core.js" + _ver);

// ---------- 寸法 ----------
const COPPER = 0xb87333;
const cfg = defaultCfg2();
const Rpin = cfg.BODY_R + cfg.wire_r;                 // 41.8
const circ = 2 * Math.PI * Rpin;
const innerZ = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const innerZ_usable = innerZ - cfg.wire_r - 2;
const pitch = 2 * cfg.wire_r * 1.05;
const cxA = -cfg.centerDist / 2, cxB = +cfg.centerDist / 2;

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
scene.fog = new THREE.Fog(0x0b0d10, 600, 2200);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.035).texture;

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 6000);
camera.position.set(120, 150, 470);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -20, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

scene.add(new THREE.AmbientLight(0xffffff, 0.2));
const keyL = new THREE.SpotLight(0xfff2dd, 14000, 1400, Math.PI / 5, 0.35, 1.4);
keyL.position.set(-180, 360, 300); keyL.target.position.set(0, 0, 0);
keyL.castShadow = true; keyL.shadow.mapSize.set(1024, 1024); keyL.shadow.bias = -0.002;
scene.add(keyL, keyL.target);
const fillL = new THREE.SpotLight(0x88aacc, 5000, 1400, Math.PI / 4, 0.6, 1.2);
fillL.position.set(280, 180, -160); fillL.target.position.set(0, 0, 0);
scene.add(fillL, fillL.target);
const rimL = new THREE.DirectionalLight(0xffe7cf, 1.2);
rimL.position.set(40, 60, -260);
scene.add(rimL);

const floor = new THREE.Mesh(new THREE.PlaneGeometry(2400, 2400),
  new THREE.MeshPhysicalMaterial({ color: 0x12141a, metalness: 0.25, roughness: 0.42, clearcoat: 0.3, clearcoatRoughness: 0.6 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -300;
floor.receiveShadow = true;
scene.add(floor);

// ---------- Materials ----------
const copperMat = new THREE.MeshPhysicalMaterial({ color: COPPER, metalness: 0.95, roughness: 0.18 });
const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0x3b2a1d, metalness: 0.15, roughness: 0.65 });
const flangeMat = new THREE.MeshPhysicalMaterial({ color: 0x1f1610, metalness: 0.5, roughness: 0.4, transparent: true, opacity: 0.38, side: THREE.DoubleSide });
const frameMat = new THREE.MeshPhysicalMaterial({ color: 0x5a6068, metalness: 0.8, roughness: 0.28 });

// ---------- Bobbin（軸=z, group.rotation.z で手動回転） ----------
function makeBobbin(cx, markColor) {
  const g = new THREE.Group();
  g.position.set(cx, 0, 0);
  const body = new THREE.Mesh(new THREE.CylinderGeometry(cfg.BODY_R, cfg.BODY_R, cfg.BODY_H, 40), bodyMat);
  body.rotation.x = Math.PI / 2;                       // 軸 y→z
  body.castShadow = true; body.receiveShadow = true;
  g.add(body);
  for (const sgn of [-1, +1]) {
    const fl = new THREE.Mesh(new THREE.CylinderGeometry(cfg.FLANGE_R, cfg.FLANGE_R, cfg.FLANGE_H, 48), flangeMat);
    fl.rotation.x = Math.PI / 2;
    fl.position.z = sgn * (cfg.BODY_H - cfg.FLANGE_H) / 2;
    g.add(fl);
  }
  // 着地点マーカ（回転が目で分かる）
  const mark = new THREE.Mesh(new THREE.SphereGeometry(cfg.wire_r * 1.8, 12, 12),
    new THREE.MeshBasicMaterial({ color: markColor }));
  mark.position.set(cfg.BODY_R, 0, 0);
  g.add(mark);
  // スポーク（回転視認）
  const spoke = new THREE.Mesh(new THREE.BoxGeometry(cfg.BODY_R * 1.9, 2.5, 2.5), frameMat);
  g.add(spoke);
  scene.add(g);
  return g;
}
const bobbinA = makeBobbin(cxA, 0x55ff88);
const bobbinB = makeBobbin(cxB, 0xff5544);

// 支持スピンドル（軸=z）
for (const cx of [cxA, cxB]) {
  const sp = new THREE.Mesh(new THREE.CylinderGeometry(7, 7, cfg.BODY_H * 1.5, 16), frameMat);
  sp.rotation.x = Math.PI / 2; sp.position.set(cx, 0, 0); sp.castShadow = true;
  scene.add(sp);
}

// ---------- 素線 (連続 PBD 鎖) ----------
const params = {
  thetaA: 0, thetaB: Math.PI,            // [KINEMATIC] 機械駆動①ボビン回転（巻取モータ／手回しスライダ）— 角度を直接処方
  mu: 0.8, seatGrip: 0.4, bendK: 0.02, gravity: 1.0,
  slackTurns: 1.0,                       // 初期弛み（chord + slackTurns 周ぶん）= 質量保存できる巻数の上限。超過で伸び発散
  traverse: true, selfContact: true,
  paused: false,
  reset() { buildChain(); },
  viewFront() { controls.target.set(0, -20, 0); camera.position.set(0, 40, 520); controls.update(); },
  viewTop() { controls.target.set(0, -20, 0); camera.position.set(0, 520, 0.01); controls.update(); },
  viewReset() { controls.target.set(0, -20, 0); camera.position.set(120, 150, 470); controls.update(); },
};

let chain = null, N = 0, L_wire = 0, segRest = 0;
let prevThetaA = params.thetaA, prevThetaB = params.thetaB;

// [KINEMATIC] 機械駆動②軸トラバース（送りネジ／トラバースガイド）— 着地 z を θ の三角波で処方。
//   1 回転で線径×1.05 進み、両フランジ間で折返す。創発ではなく実機ガイドの運動学そのもの。
function travZ(theta) {
  if (!params.traverse) return 0;
  const dist = (Math.abs(theta) / (2 * Math.PI)) * pitch;
  const span = 2 * innerZ_usable;
  let m = dist % (2 * span); if (m < 0) m += 2 * span;
  return m <= span ? -innerZ_usable + m : -innerZ_usable + (2 * span - m);
}
// [KINEMATIC] 機械駆動③端クランプ（チャック／端末固定）— 素線端をボビン表面の 1 点に処方拘束。
//   位置は ①回転θ + ②travZ から決まる。chain.step に固定端として渡す（物理応答ではない）。
const pinAvec = () => surfacePoint(cxA, Rpin, params.thetaA, travZ(params.thetaA));
const pinBvec = () => surfacePoint(cxB, Rpin, params.thetaB, travZ(params.thetaB));

function buildChain() {
  params.thetaA = 0; params.thetaB = Math.PI;
  prevThetaA = params.thetaA; prevThetaB = params.thetaB;
  const a0 = pinAvec(), b0 = pinBvec();
  const chord = Math.hypot(b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]);
  L_wire = chord + params.slackTurns * circ;
  segRest = 4.0;
  N = Math.round(L_wire / segRest) + 1;
  const half = L_wire / 2;
  const sag = Math.sqrt(Math.max(0, half * half - (chord / 2) ** 2));
  const mid = [(a0[0] + b0[0]) / 2, -sag, 0];
  chain = new StrandChain(N, segRest, {
    gravity: [0, -9810 * params.gravity, 0],
    damping: 0.99, iters: 12,
    restLengths: new Float64Array(N - 1).fill(segRest),
  });
  chain.bendK = params.bendK;
  for (let i = 0; i < N; i++) {                          // V 字に初期化 → 重力で U に垂れる
    const f = i / (N - 1);
    let p;
    if (f < 0.5) { const g = f / 0.5; p = [a0[0] + (mid[0] - a0[0]) * g, a0[1] + (mid[1] - a0[1]) * g, 0]; }
    else { const g = (f - 0.5) / 0.5; p = [mid[0] + (b0[0] - mid[0]) * g, mid[1] + (b0[1] - mid[1]) * g, 0]; }
    const k = 3 * i;
    chain.pos[k] = p[0]; chain.pos[k + 1] = p[1]; chain.pos[k + 2] = p[2];
    chain.prev[k] = p[0]; chain.prev[k + 1] = p[1]; chain.prev[k + 2] = p[2];
  }
  buildTubeGeometry();
}

const chainMesh = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
chainMesh.castShadow = true; chainMesh.receiveShadow = true;
scene.add(chainMesh);

// ---------- 接触 ----------
const ctx = { cfg, mu: 0.8, seatGrip: 0.4, skipFriction: false, selfContact: true, bobbins: [{ cx: cxA, dTheta: 0 }, { cx: cxB, dTheta: 0 }] };
const frictionStep = (pos, n, fixed, prev) => { ctx.skipFriction = false; twoBobbinContact(pos, n, fixed, prev, ctx); };
const contactProject = (pos, n, fixed, prev) => { ctx.skipFriction = true; twoBobbinContact(pos, n, fixed, prev, ctx); };

// ---------- 軽量チューブ（capstan.js と同方式: topology 1 回確保, 毎フレ position/normal のみ） ----------
const _tube = { T: 0, R: 8, pos: null, nor: null };
let _pts = [];
function chainPoints() {
  if (_pts.length !== N) _pts = Array.from({ length: N }, () => new THREE.Vector3());
  for (let i = 0; i < N; i++) { const k = 3 * i; _pts[i].set(chain.pos[k], chain.pos[k + 1], chain.pos[k + 2]); }
  return _pts;
}
function buildTubeGeometry() {
  const T = Math.min(420, Math.max(180, N));
  const R = 8;
  const nv = (T + 1) * (R + 1);
  const positions = new Float32Array(nv * 3), normals = new Float32Array(nv * 3), uvs = new Float32Array(nv * 2);
  const indices = [];
  for (let j = 1; j <= T; j++) for (let i = 1; i <= R; i++) {
    const a = (R + 1) * (j - 1) + (i - 1), b = (R + 1) * j + (i - 1), c = (R + 1) * j + i, d = (R + 1) * (j - 1) + i;
    indices.push(a, b, d, b, c, d);
  }
  for (let j = 0; j <= T; j++) for (let i = 0; i <= R; i++) { const k = 2 * ((R + 1) * j + i); uvs[k] = j / T; uvs[k + 1] = i / R; }
  const geo = new THREE.BufferGeometry();
  geo.setIndex(indices);
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  chainMesh.geometry.dispose();
  chainMesh.geometry = geo;
  _tube.T = T; _tube.R = R; _tube.pos = positions; _tube.nor = normals;
}
function updateTube(radius) {
  if (N < 4 || !_tube.pos) { chainMesh.visible = false; return; }
  chainMesh.visible = true;
  const T = _tube.T, R = _tube.R, pos = _tube.pos, nor = _tube.nor;
  const curve = new THREE.CatmullRomCurve3(chainPoints(), false, "centripetal", 0.5);
  const frames = curve.computeFrenetFrames(T, false);
  const P = new THREE.Vector3();
  for (let j = 0; j <= T; j++) {
    curve.getPointAt(j / T, P);
    const Nrm = frames.normals[j], Bnm = frames.binormals[j];
    for (let i = 0; i <= R; i++) {
      const v = i / R * Math.PI * 2, s = Math.sin(v), c = -Math.cos(v);
      let nx = c * Nrm.x + s * Bnm.x, ny = c * Nrm.y + s * Bnm.y, nz = c * Nrm.z + s * Bnm.z;
      const inv = 1 / (Math.hypot(nx, ny, nz) || 1); nx *= inv; ny *= inv; nz *= inv;
      const idx = 3 * ((R + 1) * j + i);
      nor[idx] = nx; nor[idx + 1] = ny; nor[idx + 2] = nz;
      pos[idx] = P.x + radius * nx; pos[idx + 1] = P.y + radius * ny; pos[idx + 2] = P.z + radius * nz;
    }
  }
  chainMesh.geometry.attributes.position.needsUpdate = true;
  chainMesh.geometry.attributes.normal.needsUpdate = true;
  chainMesh.geometry.computeBoundingSphere();
}

// ---------- wrapped 弧長（HUD） ----------
function wrapped(cx) {
  const wr = cfg.wire_r, bandR = cfg.BODY_R + wr + wr * 3;
  let w = 0;
  for (let i = 0; i < N - 1; i++) {
    const ka = 3 * i, kb = 3 * (i + 1);
    const ra = Math.hypot(chain.pos[ka] - cx, chain.pos[ka + 1]);
    const rb = Math.hypot(chain.pos[kb] - cx, chain.pos[kb + 1]);
    if (ra <= bandR && Math.abs(chain.pos[ka + 2]) < innerZ && rb <= bandR && Math.abs(chain.pos[kb + 2]) < innerZ)
      w += Math.hypot(chain.pos[kb] - chain.pos[ka], chain.pos[kb + 1] - chain.pos[ka + 1], chain.pos[kb + 2] - chain.pos[ka + 2]);
  }
  return w;
}

// ---------- 更新 ----------
function updateScene(dt) {
  if (params.paused || dt <= 0 || !chain) return;
  ctx.mu = params.mu; ctx.seatGrip = params.seatGrip; ctx.selfContact = params.selfContact;
  chain.bendK = params.bendK;
  chain.gravity[1] = -9810 * params.gravity;

  const SUBSTEPS = 4;
  const subDt = dt / SUBSTEPS;
  const dThA = (params.thetaA - prevThetaA) / SUBSTEPS;
  const dThB = (params.thetaB - prevThetaB) / SUBSTEPS;
  let tA = prevThetaA, tB = prevThetaB;
  for (let s = 0; s < SUBSTEPS; s++) {
    tA += dThA; tB += dThB;
    ctx.bobbins[0].dTheta = dThA; ctx.bobbins[1].dTheta = dThB;
    const pa = surfacePoint(cxA, Rpin, tA, travZ(tA));   // [KINEMATIC] ①②③ 合成の端クランプ目標
    const pb = surfacePoint(cxB, Rpin, tB, travZ(tB));   // [KINEMATIC] ①②③ 合成の端クランプ目標
    chain.step(subDt, pa, pb, { preStep: frictionStep, contactProject, bendIters: 3, maxStep: 8 });
  }
  prevThetaA = params.thetaA; prevThetaB = params.thetaB;

  bobbinA.rotation.z = params.thetaA;   // [KINEMATIC] ①回転を見た目に反映（駆動の可視化）
  bobbinB.rotation.z = params.thetaB;   // [KINEMATIC] ①回転を見た目に反映（駆動の可視化）
  updateTube(cfg.wire_r);
}

// ---------- HUD ----------
const statusEl = document.getElementById("status");
function updateHUD() {
  if (!chain) return;
  const wA = wrapped(cxA), wB = wrapped(cxB);
  const stretch = chain.totalArcLength() / L_wire;
  // 質量保存インジケータ（破れを隠さない: 緑=保存 / 橙=微伸び / 赤=非保存）。
  // 固定長・払出無しのため slackTurns を超えて巻くと伸び発散 → 赤警告で明示。
  const cons = stretch >= 1.10 ? { c: "#ff4d4d", t: "⚠ 質量非保存（供給超過）" }
    : stretch >= 1.02 ? { c: "#d8b24a", t: "微伸び" }
    : { c: "#55ff88", t: "保存OK" };
  // [KINEMATIC] 機械駆動の凡例（travZ は traverse トグル連動で表示）
  const kin = ["①回転θ", params.traverse ? "②軸トラバース" : null, "③端クランプ"].filter(Boolean).join(" · ");
  statusEl.innerHTML =
    `<span class="phase">2ボビン手動巻取り</span> · ` +
    `<span style="color:#55ff88">A 回転 ${(params.thetaA / (2 * Math.PI)).toFixed(2)}周（巻 ${(wA / circ).toFixed(2)}）</span> · ` +
    `<span style="color:#ff7766">B 回転 ${(params.thetaB / (2 * Math.PI)).toFixed(2)}周（巻 ${(wB / circ).toFixed(2)}）</span><br>` +
    `<span class="count">電線 ${N}粒子</span> · ` +
    `<span style="color:${cons.c}">質量保存: 伸び ${stretch.toFixed(3)}× ${cons.t}</span> · ` +
    `μ=${params.mu.toFixed(2)} · seat=${params.seatGrip.toFixed(2)} · 手回しスライダ<br>` +
    `<span style="color:#d8b24a">⚙ 機械駆動(KINEMATIC): ${kin}</span> ` +
    `<span style="color:#7a8088">〜 他は全て創発物理(PBD/接触/摩擦) 〜</span>`;
}

// ---------- GUI ----------
buildChain();
const gui = new GUI({ title: "2ボビン手動巻取り" });
gui.add(params, "reset").name("⟲ リセット");
const fRot = gui.addFolder("手動回転（スライダを回す）");
fRot.add(params, "thetaA", -6 * Math.PI, 6 * Math.PI, 0.02).name("🟢 ボビンA [rad]").listen();
fRot.add(params, "thetaB", -6 * Math.PI, 6 * Math.PI, 0.02).name("🔴 ボビンB [rad]").listen();
fRot.open();
const fPhys = gui.addFolder("物理");
fPhys.add(params, "mu", 0, 1.5, 0.02).name("摩擦 μ");
fPhys.add(params, "seatGrip", 0, 1, 0.05).name("seating");
fPhys.add(params, "bendK", 0, 0.2, 0.01).name("EI (パチもん)");
fPhys.add(params, "gravity", 0, 2, 0.05).name("重力 (×g)");
fPhys.add(params, "traverse").name("軸トラバース");
fPhys.add(params, "selfContact").name("自己接触（巻き重なり防止）");
fPhys.add(params, "slackTurns", 0.2, 3, 0.1).name("初期弛み [周]").onFinishChange(() => buildChain());
const fView = gui.addFolder("視点");
fView.add(params, "viewReset").name("初期");
fView.add(params, "viewFront").name("正面");
fView.add(params, "viewTop").name("俯瞰");
fView.add(params, "paused").name("一時停止");

// ---------- Main loop ----------
let last = performance.now();
const FIXED_DT = 1 / 60;
let acc = 0;
function tick() {
  const now = performance.now();
  const wallDt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!params.paused) {
    acc += wallDt;
    let n = 0;
    while (acc >= FIXED_DT && n < 6) { updateScene(FIXED_DT); acc -= FIXED_DT; n++; }
  }
  updateHUD();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
