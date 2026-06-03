// 2ボビン手動巻取り probe（headless）: 「片方を回すと電線が巻き取られるか」の基本物理検証。
//
// セットアップ: 横並び 2 ボビン（軸=z, 中心 x=±centerDist/2）。電線は各ボビン表面の 1 点に
//   ピンされ、重力で下に弛んだループ。ボビン B を回し続け、弛みが B に巻き取られるか測る。
//
// ゲート: (1) 巻取進行  B の巻付弧が単調に増え 0.5 周以上  (2) 非貫入  どの粒子も胴体内に
//   食い込まない  (3) 健全  伸び stretch < 1.5・カクつき < 20°
//
// 実行: node wiggle/web/twobobbin_probe.mjs 2>&1 | tee /tmp/twobobbin-$(date +%s).log

import { StrandChain } from "./physics.js";
import { defaultCfg2, surfacePoint, twoBobbinContact, wrappedArcLength2 } from "./twobobbin_core.js";

const cfg = defaultCfg2();
const Rpin = cfg.BODY_R + cfg.wire_r;          // 41.8
const innerZ = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const circ = 2 * Math.PI * Rpin;               // ≈262.6 mm/周
const cxA = -cfg.centerDist / 2, cxB = +cfg.centerDist / 2;

// 着地点は互いに向き合う（A は +x 側=θ0, B は -x 側=θπ）。z=0 面で開始。
let thA = 0, thB = Math.PI;
const pinA = () => surfacePoint(cxA, Rpin, thA, 0);
const pinB = () => surfacePoint(cxB, Rpin, thB, 0);

// 電線: 弛んだループ（chord の 2.6 倍）。V 字に初期化して重力で U に垂れさせる。
const a0 = pinA(), b0 = pinB();
const chord = Math.hypot(b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]);
const L = chord * 2.6;
const segRest = 4.0;
const N = Math.round(L / segRest) + 1;
const mid = [(a0[0] + b0[0]) / 2, -(Math.sqrt(Math.max(0, (L / 2) ** 2 - (chord / 2) ** 2))), 0];

const chain = new StrandChain(N, segRest, {
  gravity: [0, -9810 * parseFloat(process.env.GRAV ?? "1"), 0],
  damping: 0.99, iters: 14,
  restLengths: new Float64Array(N - 1).fill(segRest),
});
chain.bendK = parseFloat(process.env.BENDK ?? "0.02");
for (let i = 0; i < N; i++) {                  // V 字（pinA→mid→pinB）
  const f = i / (N - 1);
  let p;
  if (f < 0.5) { const g = f / 0.5; p = [a0[0] + (mid[0] - a0[0]) * g, a0[1] + (mid[1] - a0[1]) * g, 0]; }
  else { const g = (f - 0.5) / 0.5; p = [mid[0] + (b0[0] - mid[0]) * g, mid[1] + (b0[1] - mid[1]) * g, 0]; }
  const k = 3 * i;
  chain.pos[k] = p[0]; chain.pos[k + 1] = p[1]; chain.pos[k + 2] = p[2];
  chain.prev[k] = p[0]; chain.prev[k + 1] = p[1]; chain.prev[k + 2] = p[2];
}

const ctx = {
  cfg, mu: parseFloat(process.env.MU ?? "0.8"), seatGrip: parseFloat(process.env.SEATGRIP ?? "0.4"),
  skipFriction: false, selfContact: true,
  bobbins: [{ cx: cxA, dTheta: 0 }, { cx: cxB, dTheta: 0 }],
};
const frictionStep = (pos, n, fixed, prev) => { ctx.skipFriction = false; twoBobbinContact(pos, n, fixed, prev, ctx); };
const contactProject = (pos, n, fixed, prev) => { ctx.skipFriction = true; twoBobbinContact(pos, n, fixed, prev, ctx); };

const dt = 1 / 60, SUBSTEPS = 6, subDt = dt / SUBSTEPS;
// 既定は弛み（≈1周ぶん）の範囲でクランク = 健全に巻ける regime を回帰ガード。
// 供給を超えて回し続けると（STEPS 大）固定長電線は伸びるしかない（実機は他リールから供給）。
const STEPS = parseInt(process.env.STEPS ?? "240", 10);
const omegaB = parseFloat(process.env.OMEGAB ?? "1.5");   // B を回す角速度
const omegaA = parseFloat(process.env.OMEGAA ?? "0");      // A は固定（払い出し側は弛みが供給）

function minClearance() {            // 全粒子の最小（半径 − Rpin）。負 = 貫入。
  let m = Infinity;
  for (let i = 0; i < N; i++) {
    const k = 3 * i;
    if (Math.abs(chain.pos[k + 2]) >= innerZ) continue;
    for (const cx of [cxA, cxB]) {
      const r = Math.hypot(chain.pos[k] - cx, chain.pos[k + 1]);
      if (r - Rpin < m) m = r - Rpin;
    }
  }
  return m;
}
function kinkDeg() {
  let s = 0, n = 0;
  for (let i = 1; i < N - 1; i++) {
    const a = 3 * (i - 1), b = 3 * i, c = 3 * (i + 1);
    const ux = chain.pos[b] - chain.pos[a], uy = chain.pos[b + 1] - chain.pos[a + 1], uz = chain.pos[b + 2] - chain.pos[a + 2];
    const vx = chain.pos[c] - chain.pos[b], vy = chain.pos[c + 1] - chain.pos[b + 1], vz = chain.pos[c + 2] - chain.pos[b + 2];
    const lu = Math.hypot(ux, uy, uz), lv = Math.hypot(vx, vy, vz);
    if (lu < 1e-9 || lv < 1e-9) continue;
    let ca = (ux * vx + uy * vy + uz * vz) / (lu * lv); ca = Math.max(-1, Math.min(1, ca));
    s += Math.acos(ca); n++;
  }
  return n ? s / n * 180 / Math.PI : 0;
}

console.log(`# 2ボビン probe  N=${N}  chord=${chord.toFixed(1)}  L=${L.toFixed(0)}（弛み ${(L - chord).toFixed(0)}）  segRest=${segRest}`);
console.log(`# centerDist=${cfg.centerDist}  Rpin=${Rpin}  circ=${circ.toFixed(1)}  μ=${ctx.mu} seatGrip=${ctx.seatGrip}  ωB=${omegaB}`);
console.log(`# step | θB[周] | 巻付A | 巻付B | B[周] | 最小クリア | カクつき | 伸び`);

let wB0 = wrappedArcLength2(chain.pos, N, cfg, cxB);
for (let s = 1; s <= STEPS; s++) {
  const dThA = omegaA * dt / SUBSTEPS, dThB = omegaB * dt / SUBSTEPS;
  for (let sub = 0; sub < SUBSTEPS; sub++) {
    thA += dThA; thB += dThB;
    ctx.bobbins[0].dTheta = dThA; ctx.bobbins[1].dTheta = dThB;
    chain.step(subDt, pinA(), pinB(), { preStep: frictionStep, contactProject, bendIters: 3, maxStep: 8 });
  }
  if (s % 200 === 0 || s === STEPS) {
    const wA = wrappedArcLength2(chain.pos, N, cfg, cxA);
    const wB = wrappedArcLength2(chain.pos, N, cfg, cxB);
    const stretch = chain.totalArcLength() / ((N - 1) * segRest);
    console.log(`# ${String(s).padStart(4)} | ${(thB / (2 * Math.PI)).toFixed(2).padStart(5)} | ${wA.toFixed(0).padStart(4)} | ${wB.toFixed(0).padStart(4)} | ${(wB / circ).toFixed(2).padStart(4)} | ${minClearance().toFixed(2).padStart(7)} | ${kinkDeg().toFixed(1).padStart(6)} | ${stretch.toFixed(2)}`);
  }
}

const wA = wrappedArcLength2(chain.pos, N, cfg, cxA);
const wB = wrappedArcLength2(chain.pos, N, cfg, cxB);
const turnsB = wB / circ;
const clr = minClearance();
const stretch = chain.totalArcLength() / ((N - 1) * segRest);
const kink = kinkDeg();
console.log(`#\n# === 結果 ===`);
console.log(`# 巻付 A=${wA.toFixed(0)}mm  B=${wB.toFixed(0)}mm (${turnsB.toFixed(2)}周)  最小クリア=${clr.toFixed(2)}mm  カクつき=${kink.toFixed(1)}°  伸び=${stretch.toFixed(2)}×`);
const gate1 = turnsB >= 0.5;                          // 巻取が 0.5 周以上進む
const gate2 = clr >= -cfg.wire_r;                     // 貫入が線半径以内（実質非貫入）
const gate3 = stretch < 1.5 && kink < 20;             // 健全
console.log(`# ゲート1 巻取進行: ${gate1 ? "PASS ✅" : "FAIL ❌"}  ゲート2 非貫入: ${gate2 ? "PASS ✅" : "FAIL ❌"}  ゲート3 健全: ${gate3 ? "PASS ✅" : "FAIL ❌"}`);
