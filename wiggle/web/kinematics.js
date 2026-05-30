// 撚線機の運動学。Python 版 wiggle/kinematics.py の JS ポート。
// 単位: mm / rad / s
// 物理整合: helix の imprint 位相は -ω·t_pass（撚り上げ方向 = 材料は +z 流れ）

export class StranderConfig {
  constructor(overrides = {}) {
    const defaults = {
      n_outer: 6,
      R_layer: 5.5,         // mm — 1+6 構成の center-to-outer 距離
      R_bobbin: 120.0,      // mm — cradle 半径
      L_pitch: 45.0,        // mm — ~12× 素線径
      z_lay: 250.0,         // mm — レイプレート位置
      L_tail_in: 450.0,     // mm — ボビン → レイプレート
      L_tail_out: 400.0,    // mm — レイプレート → 巻取り
      omega: (2 * Math.PI) / 2.0,
      n_segments: 400,
      core_radius: 2.0,     // mm
      outer_radius: 1.8,    // mm
      L_total: 8000.0,      // mm — サプライ 1 本あたりの初期素線長（多層巻きが見えるよう余裕）
    };
    Object.assign(this, defaults, overrides);
  }
  get z0() { return this.z_lay - this.L_tail_in; }
  get z1() { return this.z_lay + this.L_tail_out; }
  get helix_angle_deg() {
    return Math.atan2(2 * Math.PI * this.R_layer, this.L_pitch) * 180 / Math.PI;
  }
}

// path 全長 (bobbin 出口 → lay plate → takeup 入口) の弧長近似 [mm]
export function strandPathArcLength(cfg) {
  const tailIn = Math.hypot(cfg.R_bobbin - cfg.R_layer, cfg.L_tail_in);
  const alpha = Math.hypot(1, 2 * Math.PI * cfg.R_layer / cfg.L_pitch);
  const tailOut = cfg.L_tail_out * alpha;
  return tailIn + tailOut;
}

// 素線消費速度（lay 後の helix 弧長/時間）[mm/s]
export function strandFeedRate(cfg) {
  const v_feed = Math.abs(cfg.omega) * cfg.L_pitch / (2 * Math.PI);
  const alpha = Math.hypot(1, 2 * Math.PI * cfg.R_layer / cfg.L_pitch);
  return v_feed * alpha;
}

// 時刻 t の素線残量分配（material conservation）
// 返り値: { L_path, L_supply_init, L_supply, L_transit, L_takeup, s_trail, v_strand, phase, t_total }
export function strandState(cfg, t) {
  const L_path = strandPathArcLength(cfg);
  const v_strand = strandFeedRate(cfg);
  const L_total = cfg.L_total;
  const L_supply_init = Math.max(L_total - L_path, 0);
  const t_total = v_strand > 1e-9 ? L_total / v_strand : Infinity;
  const L_consumed = Math.min(Math.max(v_strand * Math.max(t, 0), 0), L_total);
  const L_transit = Math.max(0, Math.min(L_path, L_total - L_consumed));
  const L_supply = Math.max(0, L_supply_init - L_consumed);
  const L_takeup = L_total - L_supply - L_transit;
  const s_trail = L_path - L_transit;   // 描画開始点の弧長（bobbin 側）
  let phase;
  if (L_consumed >= L_total - 1e-6) phase = 3;
  else if (L_supply > 0) phase = 1;
  else phase = 2;
  return { L_path, L_supply_init, L_supply, L_transit, L_takeup,
           s_trail, v_strand, phase, t_total, L_consumed };
}

function _zGrid(cfg) {
  const out = new Float32Array(cfg.n_segments);
  const dz = (cfg.z1 - cfg.z0) / (cfg.n_segments - 1);
  for (let k = 0; k < cfg.n_segments; k++) out[k] = cfg.z0 + k * dz;
  return out;
}

export function outerStrandCenterline(cfg, i, t) {
  const z = _zGrid(cfg);
  const phi0 = (2 * Math.PI * i) / cfg.n_outer;
  const theta_lay = phi0 - cfg.omega * t;
  const pts = new Array(cfg.n_segments);
  for (let k = 0; k < cfg.n_segments; k++) {
    const zk = z[k];
    let radius, theta;
    if (zk < cfg.z_lay) {
      const ramp = Math.min(1, Math.max(0, (zk - cfg.z0) / cfg.L_tail_in));
      radius = cfg.R_bobbin + (cfg.R_layer - cfg.R_bobbin) * ramp;
      theta = theta_lay;
    } else {
      radius = cfg.R_layer;
      theta = theta_lay + (2 * Math.PI * (zk - cfg.z_lay)) / cfg.L_pitch;
    }
    pts[k] = { x: radius * Math.cos(theta), y: radius * Math.sin(theta), z: zk };
  }
  return pts;
}

// 弧長 s_trail [mm] より bobbin 側を切り捨てた centerline を返す。
// s_trail <= 0: そのまま全長返す。フェーズ 2 で後端が path 沿いに前進する用。
export function trimCenterlineByArc(pts, s_trail) {
  if (s_trail <= 0) return pts;
  let accum = 0;
  for (let k = 1; k < pts.length; k++) {
    const dx = pts[k].x - pts[k - 1].x;
    const dy = pts[k].y - pts[k - 1].y;
    const dz = pts[k].z - pts[k - 1].z;
    const dl = Math.hypot(dx, dy, dz);
    if (accum + dl >= s_trail) {
      const frac = dl > 1e-9 ? (s_trail - accum) / dl : 0;
      const cut = {
        x: pts[k - 1].x + frac * dx,
        y: pts[k - 1].y + frac * dy,
        z: pts[k - 1].z + frac * dz,
      };
      return [cut, ...pts.slice(k)];
    }
    accum += dl;
  }
  return [];
}

// 多層レベルワインド巻きの素線ヘリックスを構築。
// L 長の素線を R_core から R_outer 方向へ、軸 +y、ピッチ pitch で交互往復しながら巻く。
// 終端 (= 最新点 = 払い出し/取り付け側) は pts[pts.length-1]。
// 戻り値: bobbin local frame (cylinder axis = +y、中心 y=0) の点列。
export function buildLevelWindHelix(L, R_core, R_outer, H_wind, wire_d, opts = {}) {
  const pitch = (opts.pitch ?? wire_d) * 1.05;
  const layer_step = (opts.layer_step ?? wire_d) * 0.96;
  const turns_per_layer = Math.max(2, Math.floor((H_wind - pitch) / pitch));
  const axial_span = turns_per_layer * pitch;
  const y_min = -axial_span / 2;
  const y_max = +axial_span / 2;
  const segs_per_turn = opts.segs_per_turn ?? 28;
  const pts = [];
  let remaining = Math.max(L, 0);
  let layer = 0;
  while (remaining > 1e-3) {
    const r = R_core + (layer + 0.5) * layer_step;
    if (r > R_outer) break;
    const turn_len = Math.hypot(2 * Math.PI * r, pitch);
    const turns_this = Math.min(turns_per_layer, remaining / turn_len);
    if (turns_this < 1e-5) break;
    const dir = (layer % 2 === 0) ? +1 : -1;
    const y_start = (dir > 0) ? y_min : y_max;
    const phi_offset = layer * Math.PI;
    const N = Math.max(2, Math.ceil(turns_this * segs_per_turn));
    for (let k = 0; k <= N; k++) {
      const f = k / N;
      const angle = phi_offset + dir * 2 * Math.PI * turns_this * f;
      const y = y_start + dir * pitch * turns_this * f;
      pts.push({ x: r * Math.cos(angle), y, z: r * Math.sin(angle) });
    }
    remaining -= turns_this * turn_len;
    layer += 1;
  }
  return pts;
}

// 外周素線の wrap 由来 pretension で chain の余剰長を縮める。
// 物理: 各外周素線 1 mm あたりに遠心力 ρω²R が働き、helix 張力 T = ρω²R²/sin²α で支える。
// 軸方向成分は T cos α、n_outer 本ぶん合算で bundle 軸の pretension。
// catenary: sag ∝ 1/T → 余剰長 ∝ 1/T²。slack_eff = 1 + (slack_base - 1) · (T_base/T_eff)²。
// k_omega は T_base 由来の単一フィットパラメータ。
export function wrapPretensionSlack(baseSlack, cfg, k_omega = 3e-4) {
  const tanA2 = (2 * Math.PI * cfg.R_layer / cfg.L_pitch) ** 2;
  const sin2 = tanA2 / (1 + tanA2);
  const cosA = 1 / Math.sqrt(1 + tanA2);
  if (sin2 < 1e-6) return baseSlack;
  const wrap = cfg.n_outer * cfg.omega * cfg.omega *
               cfg.R_layer * cfg.R_layer * cosA / sin2;
  const scale = 1 / (1 + k_omega * wrap);
  return 1 + (baseSlack - 1) * scale * scale;
}

// 点列に沿った Bishop（parallel-transport）フレーム。
// 各点で接線 t_k に直交する e1_k, e2_k を返す。
// 始点は (1,0,0) を t_0 直交平面に射影して初期化、以降は前フレームを transport。
export function computeBishopFrames(pts) {
  const N = pts.length;
  if (N < 2) return null;
  const tangents = new Array(N);
  const e1s = new Array(N);
  const e2s = new Array(N);
  for (let k = 0; k < N; k++) {
    let dx, dy, dz;
    if (k === 0) {
      dx = pts[1].x - pts[0].x; dy = pts[1].y - pts[0].y; dz = pts[1].z - pts[0].z;
    } else if (k === N - 1) {
      dx = pts[N - 1].x - pts[N - 2].x;
      dy = pts[N - 1].y - pts[N - 2].y;
      dz = pts[N - 1].z - pts[N - 2].z;
    } else {
      dx = pts[k + 1].x - pts[k - 1].x;
      dy = pts[k + 1].y - pts[k - 1].y;
      dz = pts[k + 1].z - pts[k - 1].z;
    }
    const m = Math.hypot(dx, dy, dz) || 1;
    tangents[k] = [dx / m, dy / m, dz / m];
  }
  // 始点 e1: (1,0,0) を t_0 直交平面に射影。退化なら (0,1,0) で再試行。
  const t0 = tangents[0];
  let e1 = [1 - t0[0] * t0[0], -t0[0] * t0[1], -t0[0] * t0[2]];
  let m1 = Math.hypot(e1[0], e1[1], e1[2]);
  if (m1 < 0.2) {
    e1 = [-t0[1] * t0[0], 1 - t0[1] * t0[1], -t0[1] * t0[2]];
    m1 = Math.hypot(e1[0], e1[1], e1[2]) || 1;
  }
  e1s[0] = [e1[0] / m1, e1[1] / m1, e1[2] / m1];
  e2s[0] = [
    t0[1] * e1s[0][2] - t0[2] * e1s[0][1],
    t0[2] * e1s[0][0] - t0[0] * e1s[0][2],
    t0[0] * e1s[0][1] - t0[1] * e1s[0][0],
  ];
  for (let k = 1; k < N; k++) {
    const tk = tangents[k];
    const prev = e1s[k - 1];
    const dot = prev[0] * tk[0] + prev[1] * tk[1] + prev[2] * tk[2];
    let nx = prev[0] - dot * tk[0];
    let ny = prev[1] - dot * tk[1];
    let nz = prev[2] - dot * tk[2];
    const mn = Math.hypot(nx, ny, nz) || 1;
    e1s[k] = [nx / mn, ny / mn, nz / mn];
    e2s[k] = [
      tk[1] * e1s[k][2] - tk[2] * e1s[k][1],
      tk[2] * e1s[k][0] - tk[0] * e1s[k][2],
      tk[0] * e1s[k][1] - tk[1] * e1s[k][0],
    ];
  }
  return { tangents, e1s, e2s };
}

export function cumulativeArcLength(pts) {
  const N = pts.length;
  const s = new Float32Array(N);
  for (let k = 1; k < N; k++) {
    s[k] = s[k - 1] + Math.hypot(
      pts[k].x - pts[k - 1].x,
      pts[k].y - pts[k - 1].y,
      pts[k].z - pts[k - 1].z,
    );
  }
  return s;
}

// downstream helix のみ（z_lay 〜 z1）。chain physics と接続するため独立関数化。
// 始点 (z=z_lay) は (R_layer·cos θ_lay, R_layer·sin θ_lay, z_lay)。
export function outerStrandHelixDownstream(cfg, i, t) {
  const n = Math.max(40, Math.floor(cfg.n_segments / 2));
  const dz = (cfg.z1 - cfg.z_lay) / (n - 1);
  const phi0 = (2 * Math.PI * i) / cfg.n_outer;
  const theta_lay = phi0 - cfg.omega * t;
  const pts = new Array(n);
  for (let k = 0; k < n; k++) {
    const zk = cfg.z_lay + k * dz;
    const theta = theta_lay + (2 * Math.PI * (zk - cfg.z_lay)) / cfg.L_pitch;
    pts[k] = { x: cfg.R_layer * Math.cos(theta), y: cfg.R_layer * Math.sin(theta), z: zk };
  }
  return pts;
}

// 物理 chain の始点（サプライボビン body 中央）と終点（lay 入口）を計算
export function strandPinEndpoints(cfg, i, t) {
  const phi = (2 * Math.PI * i) / cfg.n_outer - cfg.omega * t;
  // bobbin body 中央: cradle ring から軸方向に BOBBIN_BODY_H/2 内側
  const R_body_mid = cfg.R_bobbin - 35;  // = R_bobbin - BOBBIN_BODY_H/2 (main.js 側定数と整合)
  const start = [
    R_body_mid * Math.cos(phi),
    R_body_mid * Math.sin(phi),
    cfg.z0 - 40,
  ];
  const end = [
    cfg.R_layer * Math.cos(phi),
    cfg.R_layer * Math.sin(phi),
    cfg.z_lay,
  ];
  return { start, end };
}

export function coreStrandCenterline(cfg) {
  const z = _zGrid(cfg);
  const pts = new Array(cfg.n_segments);
  for (let k = 0; k < cfg.n_segments; k++) pts[k] = { x: 0, y: 0, z: z[k] };
  return pts;
}

export function bobbinTransform(cfg, i, t) {
  const phi0 = (2 * Math.PI * i) / cfg.n_outer - cfg.omega * t;
  return {
    position: {
      x: cfg.R_bobbin * Math.cos(phi0),
      y: cfg.R_bobbin * Math.sin(phi0),
      z: cfg.z0 - 40,
    },
    axis: { x: -Math.cos(phi0), y: -Math.sin(phi0), z: 0 },
  };
}
