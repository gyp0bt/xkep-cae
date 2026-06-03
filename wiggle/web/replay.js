// wiggle — 巻き替えリプレイ + 録画。twobobbin_sim.mjs が吐いた frames.bin/meta.json を
// 既存の three.js 見た目で再生し、MediaRecorder で webm に録画する（物理計算は一切しない）。
// 座標系は sim と同一: +y 上, ボビン軸=+z, x=±centerDist/2。

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import GUI from "lil-gui";

const _ver = new URL(import.meta.url).search;
const DIR = new URL(location.href).searchParams.get("dir") || "rewind";

// ---------- データ読み込み ----------
const meta = await (await fetch(`./${DIR}/meta.json${_ver}`)).json();
const buf = await (await fetch(`./${DIR}/frames.bin${_ver}`)).arrayBuffer();
const frames = new Float32Array(buf);
const N = meta.N, F = meta.frames, fps = meta.fps;
const wireR = meta.wireR, Rpin = meta.Rpin, cxA = meta.cxA, cxB = meta.cxB;
const L_rest = (N - 1) * meta.segRest;   // rest 全長 = 質量保存の基準長

// 当該フレームの arc 全長（質量保存チェック用, 体積 ∝ 全長 / 断面一定）
function frameArcLen(f) {
  const base = f * N * 3; let acc = 0;
  for (let i = 0; i < N - 1; i++) {
    const a = base + 3 * i, b = base + 3 * (i + 1);
    acc += Math.hypot(frames[b] - frames[a], frames[b + 1] - frames[a + 1], frames[b + 2] - frames[a + 2]);
  }
  return acc;
}

// ---------- Renderer / Scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
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
camera.position.set(120, 150, 520);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -20, 0);
controls.enableDamping = true; controls.dampingFactor = 0.06;

scene.add(new THREE.AmbientLight(0xffffff, 0.2));
const keyL = new THREE.SpotLight(0xfff2dd, 14000, 1600, Math.PI / 5, 0.35, 1.4);
keyL.position.set(-180, 360, 320); keyL.target.position.set(0, 0, 0);
keyL.castShadow = true; keyL.shadow.mapSize.set(2048, 2048); keyL.shadow.bias = -0.002;
scene.add(keyL, keyL.target);
const fillL = new THREE.SpotLight(0x88aacc, 5000, 1600, Math.PI / 4, 0.6, 1.2);
fillL.position.set(280, 180, -160); fillL.target.position.set(0, 0, 0);
scene.add(fillL, fillL.target);
const rimL = new THREE.DirectionalLight(0xffe7cf, 1.2);
rimL.position.set(40, 60, -260);
scene.add(rimL);

const floor = new THREE.Mesh(new THREE.PlaneGeometry(2400, 2400),
  new THREE.MeshPhysicalMaterial({ color: 0x12141a, metalness: 0.25, roughness: 0.42, clearcoat: 0.3, clearcoatRoughness: 0.6 }));
floor.rotation.x = -Math.PI / 2; floor.position.y = -300; floor.receiveShadow = true;
scene.add(floor);

// ---------- Materials / Bobbins ----------
const copperMat = new THREE.MeshPhysicalMaterial({ color: 0xb87333, metalness: 0.95, roughness: 0.18 });
const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0x3b2a1d, metalness: 0.15, roughness: 0.65 });
const flangeMat = new THREE.MeshPhysicalMaterial({ color: 0x1f1610, metalness: 0.5, roughness: 0.4, transparent: true, opacity: 0.34, side: THREE.DoubleSide });
const frameMat = new THREE.MeshPhysicalMaterial({ color: 0x5a6068, metalness: 0.8, roughness: 0.28 });

function makeBobbin(cx, markColor) {
  const g = new THREE.Group(); g.position.set(cx, 0, 0);
  const body = new THREE.Mesh(new THREE.CylinderGeometry(meta.BODY_R, meta.BODY_R, meta.BODY_H, 48), bodyMat);
  body.rotation.x = Math.PI / 2; body.castShadow = true; body.receiveShadow = true; g.add(body);
  for (const sgn of [-1, +1]) {
    const fl = new THREE.Mesh(new THREE.CylinderGeometry(meta.FLANGE_R, meta.FLANGE_R, meta.FLANGE_H, 56), flangeMat);
    fl.rotation.x = Math.PI / 2; fl.position.z = sgn * (meta.BODY_H - meta.FLANGE_H) / 2; g.add(fl);
  }
  const spoke = new THREE.Mesh(new THREE.BoxGeometry(meta.BODY_R * 1.9, 2.5, 2.5), frameMat); g.add(spoke);
  const mark = new THREE.Mesh(new THREE.SphereGeometry(wireR * 1.8, 12, 12), new THREE.MeshBasicMaterial({ color: markColor }));
  mark.position.set(meta.BODY_R, 0, 0); g.add(mark);
  scene.add(g); return g;
}
const bobbinA = makeBobbin(cxA, 0x55ff88), bobbinB = makeBobbin(cxB, 0xff5544);
for (const cx of [cxA, cxB]) {
  const sp = new THREE.Mesh(new THREE.CylinderGeometry(7, 7, meta.BODY_H * 1.5, 16), frameMat);
  sp.rotation.x = Math.PI / 2; sp.position.set(cx, 0, 0); sp.castShadow = true; scene.add(sp);
}

// ---------- 銅線チューブ（topology 1 回, 毎フレ position/normal 更新） ----------
const chainMesh = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
chainMesh.castShadow = true; chainMesh.receiveShadow = true; scene.add(chainMesh);
const _tube = { T: Math.min(1600, Math.max(360, N)), R: 8, pos: null, nor: null };
let _pts = Array.from({ length: N }, () => new THREE.Vector3());
function setFrame(f) {
  const base = f * N * 3;
  for (let i = 0; i < N; i++) { const k = base + 3 * i; _pts[i].set(frames[k], frames[k + 1], frames[k + 2]); }
}
function buildTube() {
  const T = _tube.T, R = _tube.R, nv = (T + 1) * (R + 1);
  const positions = new Float32Array(nv * 3), normals = new Float32Array(nv * 3), uvs = new Float32Array(nv * 2), indices = [];
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
  chainMesh.geometry.dispose(); chainMesh.geometry = geo;
  _tube.pos = positions; _tube.nor = normals;
}
function updateTube() {
  const T = _tube.T, R = _tube.R, pos = _tube.pos, nor = _tube.nor;
  const curve = new THREE.CatmullRomCurve3(_pts, false, "centripetal", 0.5);
  const frm = curve.computeFrenetFrames(T, false);
  const P = new THREE.Vector3();
  for (let j = 0; j <= T; j++) {
    curve.getPointAt(j / T, P);
    const Nrm = frm.normals[j], Bnm = frm.binormals[j];
    for (let i = 0; i <= R; i++) {
      const v = i / R * Math.PI * 2, s = Math.sin(v), c = -Math.cos(v);
      let nx = c * Nrm.x + s * Bnm.x, ny = c * Nrm.y + s * Bnm.y, nz = c * Nrm.z + s * Bnm.z;
      const inv = 1 / (Math.hypot(nx, ny, nz) || 1); nx *= inv; ny *= inv; nz *= inv;
      const idx = 3 * ((R + 1) * j + i);
      nor[idx] = nx; nor[idx + 1] = ny; nor[idx + 2] = nz;
      pos[idx] = P.x + wireR * nx; pos[idx + 1] = P.y + wireR * ny; pos[idx + 2] = P.z + wireR * nz;
    }
  }
  chainMesh.geometry.attributes.position.needsUpdate = true;
  chainMesh.geometry.attributes.normal.needsUpdate = true;
  chainMesh.geometry.computeBoundingSphere();
}
buildTube();

// ---------- 再生状態 ----------
const params = {
  frame: 0, playing: true, speed: 1.0, loop: true,
  recording: false,
  viewReset() { controls.target.set(0, -20, 0); camera.position.set(120, 150, 520); controls.update(); },
  viewFront() { controls.target.set(0, -20, 0); camera.position.set(0, 30, 560); controls.update(); },
  viewTop() { controls.target.set(0, -20, 0); camera.position.set(0, 560, 0.01); controls.update(); },
  record() { params.recording ? stopRec() : startRec(); },
};

// ---------- 録画（MediaRecorder） ----------
let recorder = null, chunks = [];
function startRec() {
  chunks = [];
  const stream = renderer.domElement.captureStream(fps);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `rewind_${Date.now()}.webm`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  };
  recorder.start();
  params.recording = true; params.frame = 0; params.playing = true; recCtl.name("■ 録画停止＆保存");
}
function stopRec() { if (recorder) recorder.stop(); recorder = null; params.recording = false; recCtl.name("● 録画開始"); }

// ---------- HUD ----------
const statusEl = document.getElementById("status");
function updateHUD() {
  const fi = params.frame | 0;
  const tA = meta.thetaA[fi] / (2 * Math.PI), tB = meta.thetaB[fi] / (2 * Math.PI);
  const stretch = frameArcLen(fi) / L_rest;   // 質量保存（破れを隠さず明示）
  const cons = stretch >= 1.10 ? { c: "#ff4d4d", t: "⚠ 質量非保存" }
    : stretch >= 1.02 ? { c: "#d8b24a", t: "微伸び" }
    : { c: "#55ff88", t: "保存OK" };
  statusEl.innerHTML =
    `<span class="phase">巻き替えリプレイ</span> · frame ${fi}/${F - 1}（${(fi / fps).toFixed(1)}s）` +
    `${params.recording ? ' · <span style="color:#ff5544">● REC</span>' : ""}<br>` +
    `<span class="count">単線 ${N}粒子 · ${meta.segRest}mm/seg</span> · ` +
    `<span style="color:#55ff88">A ${tA.toFixed(1)}回転</span> · <span style="color:#ff7766">B ${tB.toFixed(1)}回転</span> · ` +
    `<span style="color:${cons.c}">質量保存: 伸び ${stretch.toFixed(3)}× ${cons.t}</span><br>` +
    `<span style="color:#d8b24a">⚙ 機械駆動(KINEMATIC, 焼込済): ①回転θ · ②軸トラバース · ③端クランプ</span> ` +
    `<span style="color:#7a8088">〜 他は全て創発物理 〜</span>`;
}

// ---------- GUI ----------
const gui = new GUI({ title: "巻き替えリプレイ" });
const recCtl = gui.add(params, "record").name("● 録画開始");
gui.add(params, "playing").name("再生/停止").listen();
gui.add(params, "frame", 0, F - 1, 1).name("シーク").listen();
gui.add(params, "speed", 0.1, 4, 0.1).name("速度");
gui.add(params, "loop").name("ループ");
const fView = gui.addFolder("視点");
fView.add(params, "viewReset").name("初期"); fView.add(params, "viewFront").name("正面"); fView.add(params, "viewTop").name("俯瞰");

// ---------- Main loop ----------
let fAcc = 0, last = performance.now();
function tick() {
  const now = performance.now(), wallDt = Math.min(0.1, (now - last) / 1000); last = now;
  if (params.playing) {
    fAcc += wallDt * fps * params.speed;
    if (fAcc >= 1) {
      params.frame += Math.floor(fAcc); fAcc -= Math.floor(fAcc);
      if (params.frame >= F) {
        if (params.recording) { stopRec(); params.frame = F - 1; params.playing = false; }
        else if (params.loop) params.frame = 0;
        else { params.frame = F - 1; params.playing = false; }
      }
    }
  }
  setFrame(params.frame | 0);
  updateTube();
  bobbinA.rotation.z = meta.thetaA[params.frame | 0];
  bobbinB.rotation.z = meta.thetaB[params.frame | 0];
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
