// 2ボビン巻き替え headless シミュレータ（単線・大量巻き）。
//   ボビン A に単線を TURNS 周プリワインド → A 払い出し / B 巻取りで全長を B へ巻き替え。
//   自己接触は空間ハッシュ O(N)。毎フレームの粒子位置を frames.bin にダンプ → replay.html で再生・録画。
//
// 実行: node wiggle/web/twobobbin_sim.mjs 2>&1 | tee /tmp/rewind-$(date +%s).log
//   env: TURNS(30) OMEGA(turns/s,1.0) FPS(60) FRAMES(自動=TURNS/OMEGA*FPS) OUT(./rewind) SELF(1)

import { writeFileSync, mkdirSync } from "node:fs";
import { StrandChain } from "./physics.js";
import { defaultCfg2, surfacePoint, twoBobbinContact, wrappedArcLength2 } from "./twobobbin_core.js";

const cfg = defaultCfg2();
const Rpin = cfg.BODY_R + cfg.wire_r;
const innerZ = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const innerZ_usable = innerZ - cfg.wire_r - 2;
const circ = 2 * Math.PI * Rpin;
const cxA = -cfg.centerDist / 2, cxB = +cfg.centerDist / 2;
const segRest = 4.0;

const TURNS = parseFloat(process.env.TURNS ?? "30");
const OMEGA = parseFloat(process.env.OMEGA ?? "1.0");           // turns/s（払出=巻取で共通）
const FPS = parseInt(process.env.FPS ?? "60", 10);
const FRAMES = parseInt(process.env.FRAMES ?? String(Math.round(TURNS / OMEGA * FPS)), 10);
const OUT = process.env.OUT ?? "./rewind";
const SELF = (process.env.SELF ?? "1") !== "0";

// --- 構成: A プリワインド helix + A→B 渡りスパン ---
const woundLen = TURNS * circ;
const woundN = Math.round(woundLen / segRest);
const a0 = surfacePoint(cxA, Rpin, 0, +innerZ_usable);          // A のリード端（巻きの外端, 高 index 側）
const b0 = surfacePoint(cxB, Rpin, Math.PI, +innerZ_usable);    // B 着地点（A 側を向く）
const chord = Math.hypot(b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]);
const spanN = Math.max(20, Math.round(chord / segRest));
const N = woundN + spanN;
const L_wire = (N - 1) * segRest;

const chain = new StrandChain(N, segRest, {
  gravity: [0, -9810 * parseFloat(process.env.GRAV ?? "1"), 0],
  damping: 0.99, iters: 12,
  restLengths: new Float64Array(N - 1).fill(segRest),
});
chain.bendK = parseFloat(process.env.BENDK ?? "0.02");

// [KINEMATIC] 初期条件: A プリワインド螺旋を処方配置（機械で巻いた既存巻きの初期姿勢）。
// index 0 = A 内端アンカー（z=-innerZ_usable, 角度 0）。index 増で外へ螺旋（z を +へ traverse）。
// woundN-1 = リード端（z=+innerZ_usable, 角度 0）。以降 spanN で B へ直線。
for (let i = 0; i < N; i++) {
  let p;
  if (i < woundN) {
    const turn = (i * segRest) / circ;                            // 0..TURNS
    const ang = turn * 2 * Math.PI;                               // 巻き角（+方向）
    const z = -innerZ_usable + (turn / TURNS) * 2 * innerZ_usable; // 片フランジ→他方へ
    p = surfacePoint(cxA, Rpin, ang, z);
  } else {
    const f = (i - (woundN - 1)) / spanN;                         // 0..1
    p = [a0[0] + (b0[0] - a0[0]) * f, a0[1] + (b0[1] - a0[1]) * f, a0[2] + (b0[2] - a0[2]) * f];
  }
  const k = 3 * i;
  chain.pos[k] = p[0]; chain.pos[k + 1] = p[1]; chain.pos[k + 2] = p[2];
  chain.prev[k] = p[0]; chain.prev[k + 1] = p[1]; chain.prev[k + 2] = p[2];
}

const ctx = {
  cfg, mu: parseFloat(process.env.MU ?? "0.8"), seatGrip: parseFloat(process.env.SEATGRIP ?? "0.4"),
  skipFriction: false, selfContact: SELF, spatialHash: true,
  bobbins: [{ cx: cxA, dTheta: 0 }, { cx: cxB, dTheta: 0 }],
};
const frictionStep = (pos, n, fixed, prev) => { ctx.skipFriction = false; twoBobbinContact(pos, n, fixed, prev, ctx); };
const contactProject = (pos, n, fixed, prev) => { ctx.skipFriction = true; twoBobbinContact(pos, n, fixed, prev, ctx); };

// 巻き替え駆動: A 払い出し / B 巻取り。実走で確定 → **両ボビン同符号**（A も B も +）。
// 供給 A は外端リードが B の張力で剥がれる向きに回り、巻取 B が取り込む。
// ⚠ 質量保存は厳密には破れる: 実測 stretch は序盤 1.004× → 30 周終端 1.064×（+6%）へ
//   単調ドリフト（振動でなく非可逆）。原因 = 距離拘束 Gauss-Seidel iters=12 が N≈2009 の
//   長鎖に対し不足（1 反復 1 粒子伝播 → 12/2009 しか締まらない）+ 巻済みが摩擦ロックで
//   rest 長へ戻れず伸びが焼き付く。材料は移送で足りるので **発散はしない**（.html とは別問題）。
// 単層・等半径なので |ωA|=|ωB|。OMEGA_A/OMEGA_B（turns/s）で上書き可。
// [KINEMATIC] 機械駆動①ボビン回転（巻取／払出モータ）— 角速度を直接処方。
const omA = parseFloat(process.env.OMEGA_A ?? String(+OMEGA)) * 2 * Math.PI;   // rad/s
const omB = parseFloat(process.env.OMEGA_B ?? String(+OMEGA)) * 2 * Math.PI;
const ANCHOR_Z = -innerZ_usable;
// [KINEMATIC] 機械駆動②軸トラバース（送りネジ）— B 着地 z を θ_B の三角波で処方（+端から折返し）。
function travB(thB) {                                              // B の軸トラバース（+端から折返し）
  const dist = (Math.abs(thB) / (2 * Math.PI)) * (2 * cfg.wire_r * 1.05);
  const span = 2 * innerZ_usable;
  let m = dist % (2 * span); if (m < 0) m += 2 * span;
  return m <= span ? innerZ_usable - m : innerZ_usable - (2 * span - m);
}

const dt = 1 / FPS, SUBSTEPS = 8, subDt = dt / SUBSTEPS;
let thA = 0, thB = 0;
const posBuf = new Float32Array(FRAMES * N * 3);
const thetaA = new Float32Array(FRAMES), thetaB = new Float32Array(FRAMES);

console.log(`# 巻き替え sim  N=${N}（巻${woundN}+span${spanN}）  TURNS=${TURNS}  ω=${OMEGA}t/s  FRAMES=${FRAMES}  self=${SELF}`);
console.log(`# 出力 ${OUT}/  ｜ μ=${ctx.mu} seat=${ctx.seatGrip}  centerDist=${cfg.centerDist}`);
console.log(`# frame | 巻付A[周] | 巻付B[周] | 最小クリア | 伸び | ms/frame`);

const t0 = Date.now();
for (let f = 0; f < FRAMES; f++) {
  const dA = omA * dt / SUBSTEPS, dB = omB * dt / SUBSTEPS;
  for (let s = 0; s < SUBSTEPS; s++) {
    thA += dA; thB += dB;
    ctx.bobbins[0].dTheta = dA; ctx.bobbins[1].dTheta = dB;
    // [KINEMATIC] 機械駆動③端クランプ: 両端を ①回転θ + ②travB から決まる表面点へ処方拘束。
    const pa = surfacePoint(cxA, Rpin, thA, ANCHOR_Z);
    const pb = surfacePoint(cxB, Rpin, Math.PI + thB, travB(thB));
    chain.step(subDt, pa, pb, { preStep: frictionStep, contactProject, bendIters: 3, maxStep: 8 });
  }
  posBuf.set(chain.pos, f * N * 3);
  thetaA[f] = thA; thetaB[f] = thB;
  if (f % 60 === 0 || f === FRAMES - 1) {
    const wA = wrappedArcLength2(chain.pos, N, cfg, cxA);
    const wB = wrappedArcLength2(chain.pos, N, cfg, cxB);
    let mc = Infinity;
    for (let i = 0; i < N; i++) { const k = 3 * i; if (Math.abs(chain.pos[k + 2]) >= innerZ) continue; for (const cx of [cxA, cxB]) { const r = Math.hypot(chain.pos[k] - cx, chain.pos[k + 1]); if (r - Rpin < mc) mc = r - Rpin; } }
    const stretch = chain.totalArcLength() / L_wire;
    const mspf = (Date.now() - t0) / (f + 1);
    console.log(`# ${String(f).padStart(5)} | ${(wA / circ).toFixed(2).padStart(6)} | ${(wB / circ).toFixed(2).padStart(6)} | ${mc.toFixed(2).padStart(7)} | ${stretch.toFixed(2)} | ${mspf.toFixed(1)}`);
  }
}
const wall = (Date.now() - t0) / 1000;

// --- 書き出し ---
mkdirSync(OUT, { recursive: true });
writeFileSync(`${OUT}/frames.bin`, Buffer.from(posBuf.buffer));
writeFileSync(`${OUT}/meta.json`, JSON.stringify({
  N, frames: FRAMES, fps: FPS, segRest, wireR: cfg.wire_r,
  cxA, cxB, Rpin, BODY_R: cfg.BODY_R, BODY_H: cfg.BODY_H, FLANGE_R: cfg.FLANGE_R, FLANGE_H: cfg.FLANGE_H,
  thetaA: Array.from(thetaA), thetaB: Array.from(thetaB),
}));

const wA = wrappedArcLength2(chain.pos, N, cfg, cxA);
const wB = wrappedArcLength2(chain.pos, N, cfg, cxB);
console.log(`#\n# === 結果 ===  実時間 ${wall.toFixed(1)}s（${(wall / FRAMES * 1000).toFixed(1)} ms/frame, ${(posBuf.byteLength / 1e6).toFixed(1)}MB）`);
console.log(`# 巻付 A=${(wA / circ).toFixed(2)}周  B=${(wB / circ).toFixed(2)}周（B が増え A が減れば巻き替え成立）`);
console.log(`# ${OUT}/meta.json + frames.bin を replay.html で再生`);
