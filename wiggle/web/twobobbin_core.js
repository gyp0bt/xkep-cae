// wiggle — 2ボビン手動巻取りの純物理コア（THREE 非依存）。
// twobobbin.js（描画+GUI）と twobobbin_probe.mjs（headless）が同一の真実を import する。
//
// 方針（status: 焼き付け方式を破棄して全面刷新）:
//   - 電線は **1 本の連続した PBD 鎖**（固定長 N）。焼き付け・コンベア・自動 pay-out は無し。
//   - 両端を各ボビン表面の 1 点にピン（co-rotating: ボビンが回ると着地点も回る）。
//   - 巻き済みの巻きは「摩擦で表面に貼り付いた粒子」が **創発的に** ボビンと一緒に回る
//     ＝ 別管理の凍結パックも座標変換も不要。継ぎ目・ラグが構造的に出ない。
//   - ユーザが両ボビンの角度 θ_A / θ_B を手で回す → 片方に巻き取り・片方から払い出し（転送巻き）。
//   - N は増えない（転送なので）→ コストが巻数で増えない。
//
// 座標系: +y 上、ボビン軸 = +z（水平・横並び 2 台）、gravity = -y。単位 mm / rad / s。
//
// ── kinematic 規約 ─────────────────────────────────────────────────────────
//   このファイルの物理（twoBobbinContact）は 100% 創発: 非貫入投影 + Coulomb 摩擦
//   + 自己接触のみで kinematic は含まない。機械駆動 [KINEMATIC] は全て呼び出し側
//   （twobobbin.js / twobobbin_sim.mjs）の端クランプ目標 = ①回転θ ②軸トラバース
//   ③端クランプに集約する。surfacePoint はその目標座標を作る純粋ヘルパ。
// ───────────────────────────────────────────────────────────────────────────

export function defaultCfg2() {
  return {
    BODY_R: 40,        // mm — 胴体半径（巻き付け面）
    BODY_H: 140,       // mm — 胴体長（軸=z 方向の幅）
    FLANGE_R: 78,      // mm — フランジ外径
    FLANGE_H: 8,       // mm — フランジ厚
    wire_r: 1.8,       // mm — 素線半径
    centerDist: 240,   // mm — 2 ボビン中心間距離（x 方向, 横並び）
  };
}

// ボビン（軸=+z, 中心 (cx,0,0)）の表面着地点。theta=回転角, zOff=軸方向位置。
// [KINEMATIC] の端クランプ目標座標を作る純粋ヘルパ（これ自体は driver ではなく座標関数）。
export function surfacePoint(cx, Rpin, theta, zOff) {
  return [cx + Rpin * Math.cos(theta), Rpin * Math.sin(theta), zOff];
}

// PBD iter ごとに呼ばれる接触投影: 2 円柱（軸=z）非貫入 + Coulomb 摩擦 + flange + 自己接触。
//   ctx = {
//     cfg, mu, seatGrip, skipFriction,
//     bobbins: [{cx, dTheta}, ...]   // dTheta = この substep でのボビン回転角増分（摩擦の表面追従）
//     selfContact: bool              // 巻き同士の重なり防止（O(N²)。重ければ off）
//   }
// 摩擦は winding_core.bobbinContact と同じ Coulomb（滑り限界 μ·dN）。貫入深さ dN +
// 接触帯 seating を法線力 proxy に、表面追従ずれ slipArc を μ·dN でクランプ → capstan 式
// 巻き保持が emergent に出る。bobbinContact との違いは「軸=z・中心 cx オフセット・2 台」だけ。
export function twoBobbinContact(pos, N, fixed, prev, ctx) {
  const cfg = ctx.cfg, wr = cfg.wire_r;
  const Rpin = cfg.BODY_R + wr;
  const Rpin2 = Rpin * Rpin;
  const innerZ = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;   // 胴体半幅（軸=z）
  const innerZ_clip = innerZ - wr;
  const Rflange = cfg.FLANGE_R + wr, Rflange2 = Rflange * Rflange;
  const mu = ctx.mu ?? 0;
  const seatGrip = ctx.seatGrip ?? 0;
  const bandR = Rpin + wr * 4;   // seating を効かせる接触帯

  // (1) 各ボビン胴体: 法線非貫入 + Coulomb 摩擦（軸=z, 半径面=xy）
  for (const bob of ctx.bobbins) {
    const cx = bob.cx, dTheta = bob.dTheta ?? 0;
    for (let i = 0; i < N; i++) {
      if (fixed[i]) continue;
      const k = 3 * i;
      if (Math.abs(pos[k + 2]) >= innerZ) continue;        // 軸方向に胴体外
      let lx = pos[k] - cx, ly = pos[k + 1];
      const r2 = lx * lx + ly * ly;
      let dN = 0;
      if (r2 < Rpin2 && r2 > 1e-12) {                      // 法線投影 → 表面へ押し出し
        const r = Math.sqrt(r2);
        dN = Rpin - r;
        const s = Rpin / r;
        lx *= s; ly *= s;
        pos[k] = cx + lx; pos[k + 1] = ly;
      }
      if (!ctx.skipFriction && mu > 0 && prev) {
        const rr2 = lx * lx + ly * ly;
        if (rr2 > 1e-9) {
          const rr = Math.sqrt(rr2);
          let dNf = dN;
          if (rr < bandR) dNf += (bandR - rr) * seatGrip;  // 接触帯 seating
          if (dNf > 0) {
            const aCur = Math.atan2(ly, lx);
            const aPrev = Math.atan2(prev[k + 1], prev[k] - cx);
            let dA = (aPrev + dTheta) - aCur;              // 表面が dTheta 回った先へ追従
            while (dA > Math.PI) dA -= 2 * Math.PI;
            while (dA < -Math.PI) dA += 2 * Math.PI;
            const slipArc = rr * dA;
            const maxFric = mu * dNf;
            const corr = slipArc > maxFric ? maxFric : (slipArc < -maxFric ? -maxFric : slipArc);
            const aNew = aCur + corr / rr;
            pos[k] = cx + rr * Math.cos(aNew);
            pos[k + 1] = rr * Math.sin(aNew);
            const dz = pos[k + 2] - prev[k + 2];           // 軸方向(z) no-slip クランプ
            if (dz > maxFric) pos[k + 2] = prev[k + 2] + maxFric;
            else if (dz < -maxFric) pos[k + 2] = prev[k + 2] - maxFric;
          }
        }
      }
    }
  }

  // (2) フランジ面（軸=z の薄い円盤, z=±innerZ）— 胴体近傍だけブロック（起動バースト回避）
  for (const bob of ctx.bobbins) {
    const cx = bob.cx;
    for (let i = 0; i < N; i++) {
      if (fixed[i]) continue;
      const k = 3 * i;
      const lz = pos[k + 2];
      if (Math.abs(lz) >= innerZ + wr) continue;
      const lx = pos[k] - cx, ly = pos[k + 1];
      if (lx * lx + ly * ly >= Rflange2) continue;
      if (lz > innerZ_clip) pos[k + 2] = innerZ_clip;
      else if (lz < -innerZ_clip) pos[k + 2] = -innerZ_clip;
    }
  }

  // (3) 自己接触（素線-素線, skipAdj=4）— 巻き同士が重ならないように。
  //   N 小（ブラウザ ~100 粒子）は素朴 O(N²)。N 大（大量巻き headless）は空間ハッシュ O(N)。
  if (ctx.selfContact) {
    if (ctx.spatialHash) selfContactHashed(pos, N, fixed, wr);
    else selfContactBrute(pos, N, fixed, wr);
  }
}

function pushPair(pos, i, j, fixed, minD) {
  const ki = 3 * i, kj = 3 * j;
  const dx = pos[ki] - pos[kj], dy = pos[ki + 1] - pos[kj + 1], dz = pos[ki + 2] - pos[kj + 2];
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 >= minD * minD || d2 < 1e-12) return;
  const d = Math.sqrt(d2), corr = (minD - d) / d * 0.5;
  const cx2 = dx * corr, cy2 = dy * corr, cz2 = dz * corr;
  const fi = fixed[i], fj = fixed[j];
  if (!fi && !fj) { pos[ki] += cx2; pos[ki + 1] += cy2; pos[ki + 2] += cz2; pos[kj] -= cx2; pos[kj + 1] -= cy2; pos[kj + 2] -= cz2; }
  else if (!fi) { pos[ki] += 2 * cx2; pos[ki + 1] += 2 * cy2; pos[ki + 2] += 2 * cz2; }
  else if (!fj) { pos[kj] -= 2 * cx2; pos[kj + 1] -= 2 * cy2; pos[kj + 2] -= 2 * cz2; }
}

function selfContactBrute(pos, N, fixed, wr) {
  const minD = 2 * wr;
  for (let i = 0; i < N; i++) for (let j = i + 4; j < N; j++) pushPair(pos, i, j, fixed, minD);
}

// 空間ハッシュ自己接触: セル幅 = 接触距離。各粒子を格子セルに登録し、27 近傍セルだけ走査して
// O(N) で衝突対を見つける。N が数千〜数十万でも素朴 O(N²) を回避。ハッシュ衝突は実距離で弾く。
export function selfContactHashed(pos, N, fixed, wr) {
  const minD = 2 * wr, cs = minD, inv = 1 / cs;
  const grid = new Map();
  const cellOf = (x) => Math.floor(x * inv);
  const key = (cx, cy, cz) => (((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) >>> 0);
  for (let i = 0; i < N; i++) {
    const k = 3 * i;
    const h = key(cellOf(pos[k]), cellOf(pos[k + 1]), cellOf(pos[k + 2]));
    let arr = grid.get(h); if (!arr) { arr = []; grid.set(h, arr); } arr.push(i);
  }
  for (let i = 0; i < N; i++) {
    const k = 3 * i;
    const cx = cellOf(pos[k]), cy = cellOf(pos[k + 1]), cz = cellOf(pos[k + 2]);
    for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) for (let oz = -1; oz <= 1; oz++) {
      const arr = grid.get(key(cx + ox, cy + oy, cz + oz));
      if (!arr) continue;
      for (let a = 0; a < arr.length; a++) {
        const j = arr[a];
        if (j > i && j - i >= 4) pushPair(pos, i, j, fixed, minD);   // 各対 1 回 + skipAdj
      }
    }
  }
}

// ボビン（中心 cx）の胴体に巻き付いている弧長 [mm]。接触帯に両端が入る seg の長さ合計。
export function wrappedArcLength2(pos, N, cfg, cx) {
  const wr = cfg.wire_r;
  const bandR = cfg.BODY_R + wr + wr * 3;
  const innerZ = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
  let w = 0;
  for (let i = 0; i < N - 1; i++) {
    const ka = 3 * i, kb = 3 * (i + 1);
    const ra = Math.hypot(pos[ka] - cx, pos[ka + 1]);
    const rb = Math.hypot(pos[kb] - cx, pos[kb + 1]);
    if (ra <= bandR && Math.abs(pos[ka + 2]) < innerZ && rb <= bandR && Math.abs(pos[kb + 2]) < innerZ) {
      w += Math.hypot(pos[kb] - pos[ka], pos[kb + 1] - pos[ka + 1], pos[kb + 2] - pos[ka + 2]);
    }
  }
  return w;
}
