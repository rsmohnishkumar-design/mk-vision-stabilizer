/*!
 * MK Vision Stabilizer — browser module
 *
 *   const stab = new MKStabilizer();
 *   await stab.load(fileOrUrlOrVideoElement);
 *   await stab.analyze({ onProgress: p => ... });            // one-time motion analysis (in a Web Worker)
 *   await stab.configure({ mode: 'lock', crop: 0.2, edgeFill: 'blur', strength: 1 });   // instant, re-run any time
 *     edgeFill: 'blur' | 'previous' | 'black' (borders allowed)  or  'zoom'/'none' (crop only, never shows borders)
 *     strength: 0.1..1.5 scales smoothing in cinematic/natural modes
 *   stab.drawFrame(ctx, video, mediaTime, width, height);    // use inside your own render loop
 *   const blob = await stab.export({ onProgress });         // stabilized video file (mp4 or webm) with audio
 *
 * Modes:  'lock'      tripod shot — holds the framing still for as long as the footage allows
 *         'cinematic' very smooth glide (3 s smoothing), follows deliberate moves
 *         'natural'   light smoothing (1 s), keeps more of the frame
 */
(function (global) {
  'use strict';

  const CORE_FACTORY = function () { /*
 * MK Vision — Stabilizer core (pure JS, no dependencies).
 * Works on grayscale Uint8 frames at a small analysis width (default 320).
 *
 *  estimateMotion(prev, cur, w, h)  -> {a,b,tx,ty,ok,method}  similarity in CENTERED coords
 *      x' = a*x - b*y + tx ;  y' = b*x + a*y + ty   (maps prev -> cur)
 *  solvePath(motions, opts)         -> per-frame correction transforms
 */

// ---------- image helpers ----------
function toFloat(g) { const f = new Float32Array(g.length); for (let i = 0; i < g.length; i++) f[i] = g[i]; return f; }

function blurDown(src, w, h) {
  // 5-tap binomial blur + 2x decimation
  const k = [1, 4, 6, 4, 1];
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -2; i <= 2; i++) { let xx = x + i; xx = xx < 0 ? -xx : xx >= w ? 2 * w - xx - 2 : xx; s += k[i + 2] * src[y * w + xx]; }
    tmp[y * w + x] = s / 16;
  }
  const w2 = w >> 1, h2 = h >> 1, out = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    let s = 0; const sx = x * 2;
    for (let i = -2; i <= 2; i++) { let yy = y * 2 + i; yy = yy < 0 ? -yy : yy >= h ? 2 * h - yy - 2 : yy; s += k[i + 2] * tmp[yy * w + sx]; }
    out[y * w2 + x] = s / 16;
  }
  return { d: out, w: w2, h: h2 };
}

function pyramid(gray, w, h, levels) {
  const p = [{ d: toFloat(gray), w, h }];
  for (let l = 1; l < levels; l++) { const q = p[l - 1]; p.push(blurDown(q.d, q.w, q.h)); }
  return p;
}

function grad(img) {
  const { d, w, h } = img, gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    gx[i] = (d[i + 1] - d[i - 1]) * 0.5; gy[i] = (d[i + w] - d[i - w]) * 0.5;
  }
  return { gx, gy };
}

function sample(d, w, h, x, y) {
  if (x < 0) x = 0; else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0; else if (y > h - 1.001) y = h - 1.001;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return (d[i] * (1 - fx) + d[i + 1] * fx) * (1 - fy) + (d[i + w] * (1 - fx) + d[i + w + 1] * fx) * fy;
}

// ---------- corners (Shi-Tomasi, grid-distributed) ----------
function corners(img, g, gridX = 16, gridY = 10, r = 3) {
  const { w, h } = img, { gx, gy } = g, pts = [];
  const cw = w / gridX, ch = h / gridY, m = 8;
  let globalMax = 0; const cand = [];
  for (let cy = 0; cy < gridY; cy++) for (let cx = 0; cx < gridX; cx++) {
    let best = 0, bx = 0, by = 0;
    const x0 = Math.max(m, Math.floor(cx * cw)), x1 = Math.min(w - m, Math.floor((cx + 1) * cw));
    const y0 = Math.max(m, Math.floor(cy * ch)), y1 = Math.min(h - m, Math.floor((cy + 1) * ch));
    for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) {
      let sxx = 0, syy = 0, sxy = 0;
      for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
        const k = (y + j) * w + x + i, a = gx[k], b = gy[k];
        sxx += a * a; syy += b * b; sxy += a * b;
      }
      const tr = sxx + syy, det = sxx * syy - sxy * sxy;
      const lmin = tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det));
      if (lmin > best) { best = lmin; bx = x; by = y; }
    }
    if (best > 0) { cand.push([bx, by, best]); if (best > globalMax) globalMax = best; }
  }
  for (const c of cand) if (c[2] > globalMax * 0.01) pts.push(c);
  return pts;
}

// ---------- coarse global translation (brute force SAD at top level) ----------
function coarseShift(A, B, R) {
  const { w, h } = A; let best = Infinity, bx = 0, by = 0;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    let s = 0, n = 0;
    for (let y = Math.max(0, -dy); y < Math.min(h, h - dy); y++) {
      const ra = y * w, rb = (y + dy) * w;
      for (let x = Math.max(0, -dx); x < Math.min(w, w - dx); x++) { s += Math.abs(A.d[ra + x] - B.d[rb + x + dx]); n++; }
    }
    if (n < w * h * 0.3) continue;
    s /= n; if (s < best) { best = s; bx = dx; by = dy; }
  }
  return [bx, by];
}

// ---------- pyramidal Lucas-Kanade ----------
function trackLK(P, Q, GP, pts, init, half = 6, iters = 12) {
  const L = P.length, res = [];
  for (const p of pts) {
    let ux = init[0], uy = init[1];  // displacement at level 0
    let ok = true;
    for (let l = L - 1; l >= 0; l--) {
      const s = 1 / (1 << l), A = P[l], B = Q[l], g = GP[l];
      const px = p[0] * s, py = p[1] * s;
      let dx = ux * s, dy = uy * s;
      if (px < half + 1 || py < half + 1 || px > A.w - half - 2 || py > A.h - half - 2) continue;
      let sxx = 0, syy = 0, sxy = 0;
      const n = (2 * half + 1) ** 2, ix = new Float32Array(n), iy = new Float32Array(n), iv = new Float32Array(n);
      let k = 0;
      for (let j = -half; j <= half; j++) for (let i = -half; i <= half; i++) {
        const a = sample(g.gx, A.w, A.h, px + i, py + j), b = sample(g.gy, A.w, A.h, px + i, py + j);
        ix[k] = a; iy[k] = b; iv[k] = sample(A.d, A.w, A.h, px + i, py + j);
        sxx += a * a; syy += b * b; sxy += a * b; k++;
      }
      const det = sxx * syy - sxy * sxy;
      if (det < 1e-3) { ok = false; break; }
      for (let it = 0; it < iters; it++) {
        let bx = 0, by = 0; k = 0;
        for (let j = -half; j <= half; j++) for (let i = -half; i <= half; i++) {
          const e = sample(B.d, B.w, B.h, px + i + dx, py + j + dy) - iv[k];
          bx += e * ix[k]; by += e * iy[k]; k++;
        }
        const vx = -(syy * bx - sxy * by) / det, vy = -(-sxy * bx + sxx * by) / det;
        dx += vx; dy += vy;
        if (vx * vx + vy * vy < 0.0009) break;
      }
      ux = dx / s; uy = dy / s;
    }
    if (ok) res.push([p[0], p[1], p[0] + ux, p[1] + uy]);
  }
  return res;
}

// ---------- similarity fit + RANSAC ----------
function fitSim(m, idx) {
  let sx = 0, sy = 0, su = 0, sv = 0, n = idx.length;
  for (const i of idx) { sx += m[i][0]; sy += m[i][1]; su += m[i][2]; sv += m[i][3]; }
  const mx = sx / n, my = sy / n, mu = su / n, mv = sv / n;
  let num_a = 0, num_b = 0, den = 0;
  for (const i of idx) {
    const x = m[i][0] - mx, y = m[i][1] - my, u = m[i][2] - mu, v = m[i][3] - mv;
    num_a += x * u + y * v; num_b += x * v - y * u; den += x * x + y * y;
  }
  if (den < 1e-9) return null;
  const a = num_a / den, b = num_b / den;
  return { a, b, tx: mu - (a * mx - b * my), ty: mv - (b * mx + a * my) };
}

function ransacSim(m, thr, iters = 400) {
  const n = m.length; if (n < 6) return null;
  let bestIn = [], rnd = 12345;
  const rand = () => (rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let t = 0; t < iters; t++) {
    const i = (rand() * n) | 0, j = (rand() * n) | 0; if (i === j) continue;
    const M = fitSim(m, [i, j]); if (!M) continue;
    const sc = Math.hypot(M.a, M.b); if (sc < 0.8 || sc > 1.25) continue;
    const inl = [];
    for (let k = 0; k < n; k++) {
      const [x, y, u, v] = m[k];
      const ex = M.a * x - M.b * y + M.tx - u, ey = M.b * x + M.a * y + M.ty - v;
      if (ex * ex + ey * ey < thr * thr) inl.push(k);
    }
    if (inl.length > bestIn.length) bestIn = inl;
  }
  if (bestIn.length < Math.max(6, n * 0.25)) return null;
  const M = fitSim(m, bestIn); M.inliers = bestIn.length; return M;
}

/**
 * prevPyr/curPyr can be passed to reuse work (see makeFrame()).
 */
function makeFrame(gray, w, h, levels = 4) {
  const P = pyramid(gray, w, h, levels);
  return { P, G: P.map(grad), w, h };
}

function estimateMotion(F0, F1) {
  const { w, h } = F0, L = F0.P.length;
  const top = L - 1, s = 1 << top;
  const R = Math.ceil(Math.max(F0.P[top].w, F0.P[top].h) * 0.25);
  const [cx, cy] = coarseShift(F0.P[top], F1.P[top], R);
  const init = [cx * s, cy * s];
  const cands = [];
  // A) feature tracking + RANSAC
  const pts = corners(F0.P[0], F0.G[0]);
  if (pts.length >= 8) {
    const tr = trackLK(F0.P, F1.P, F0.G, pts, init);
    const back = trackLK(F1.P, F0.P, F1.G, tr.map(t => [t[2], t[3]]), [-init[0], -init[1]]);
    const good = [];
    for (let i = 0; i < Math.min(tr.length, back.length); i++) {
      const e = Math.hypot(back[i][2] - tr[i][0], back[i][3] - tr[i][1]);
      if (e < 1.0) good.push(tr[i]);
    }
    const hw = w / 2, hh = h / 2;
    const M = ransacSim(good.map(([x, y, u, v]) => [x - hw, y - hh, u - hw, v - hh]), Math.max(1.0, w / 320 * 1.5));
    if (M) cands.push(['features', M]);
  }
  let best = null;
  const tryCand = (name, M0) => {
    const r = directAlign(F0, F1, M0, Math.min(2, top), 1, 12);
    const sc = Math.hypot(r.M.a, r.M.b);
    if (!(sc > 0.85 && sc < 1.18)) return;
    if (!best || r.score < best.score) best = { ...r, name };
  };
  if (cands.length && cands[0][1].inliers >= 25) {
    tryCand('features', cands[0][1]);
    if (best && best.score < 0.12) return { ...best.M, method: 'features', score: best.score };
  }
  // B) coarse translation, C) coarse rotation+translation hypotheses
  cands.push(['global', { a: 1, b: 0, tx: init[0], ty: init[1] }]);
  for (const M of coarseRotShift(F0, F1)) cands.push(['global', M]);
  cands.push(['global', { a: 1, b: 0, tx: 0, ty: 0 }]);
  for (const [name, M0] of cands) {
    const r = directAlign(F0, F1, M0, Math.min(2, top), 1, 12);
    const sc = Math.hypot(r.M.a, r.M.b);
    if (!(sc > 0.85 && sc < 1.18)) continue;
    if (!best || r.score < best.score) best = { ...r, name };
  }
  if (!best) return { a: 1, b: 0, tx: 0, ty: 0, method: 'none', score: 1 };
  return { ...best.M, method: best.name, score: best.score };
}

// ---------- direct (photometric) similarity alignment, robust to blur ----------
// M in centered coords at level 0. Works coarse->fine from level `fromL` to `toL`.
function directAlign(F0, F1, M, fromL, toL = 0, iters = 15) {
  M = { ...M };
  let score = Infinity;
  for (let l = fromL; l >= toL; l--) {
    const A = F0.P[l], B = F1.P[l], G = F1.G[l], s = 1 / (1 << l);
    const hw = A.w / 2, hh = A.h / 2, step = A.w > 200 ? 2 : 1, bd = 2;
    // normalize intensities (gain/bias) — handles exposure changes
    let tx = M.tx * s, ty = M.ty * s, a = M.a, b = M.b;
    for (let it = 0; it < iters; it++) {
      const H = new Float64Array(16), g = new Float64Array(4);
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, cnt = 0;
      // pass 1: stats for gain/bias
      const samp = [];
      for (let y = bd; y < A.h - bd; y += step) for (let x = bd; x < A.w - bd; x += step) {
        const xc = x - hw, yc = y - hh;
        const u = a * xc - b * yc + tx + hw, v = b * xc + a * yc + ty + hh;
        if (u < 1 || v < 1 || u > B.w - 2 || v > B.h - 2) continue;
        const i0 = A.d[y * A.w + x], i1 = sample(B.d, B.w, B.h, u, v);
        samp.push(xc, yc, u, v, i0, i1);
        sa += i0; sb += i1; saa += i0 * i0; sbb += i1 * i1; sab += i0 * i1; cnt++;
      }
      if (cnt < 50) { score = Infinity; break; }
      const ma = sa / cnt, mb = sb / cnt;
      const va = Math.max(1e-6, saa / cnt - ma * ma), vb = Math.max(1e-6, sbb / cnt - mb * mb);
      const gain = Math.sqrt(vb / va);
      score = 1 - (sab / cnt - ma * mb) / Math.sqrt(va * vb); // 1 - NCC
      const coverage = cnt / (((A.w - 2 * bd) / step) * ((A.h - 2 * bd) / step));
      score += Math.max(0, 0.5 - coverage) * 0.5; // penalize tiny overlaps
      let res2 = 0;
      for (let k = 0; k < samp.length; k += 6) { const r = samp[k + 5] - mb - gain * (samp[k + 4] - ma); res2 += r * r; }
      const sigma = Math.sqrt(res2 / cnt) * 1.5 + 1e-3;
      for (let k = 0; k < samp.length; k += 6) {
        const xc = samp[k], yc = samp[k + 1], u = samp[k + 2], v = samp[k + 3];
        const r = samp[k + 5] - mb - gain * (samp[k + 4] - ma);
        const w = Math.abs(r) < sigma ? 1 : sigma / Math.abs(r); // Huber
        const gx = sample(G.gx, B.w, B.h, u, v), gy = sample(G.gy, B.w, B.h, u, v);
        const J = [gx * xc + gy * yc, -gx * yc + gy * xc, gx, gy];
        for (let p = 0; p < 4; p++) { g[p] += w * J[p] * r; for (let q = 0; q < 4; q++) H[p * 4 + q] += w * J[p] * J[q]; }
      }
      // damping
      for (let p = 0; p < 4; p++) H[p * 5] *= 1.001; H[0] += 1e-3 * cnt * hw * hw * 0.01; H[5] += 1e-3 * cnt * hw * hw * 0.01;
      const d = solve4(H, g); if (!d) break;
      a -= d[0]; b -= d[1]; tx -= d[2]; ty -= d[3];
      if (Math.abs(d[2]) + Math.abs(d[3]) < 0.01 && Math.abs(d[0]) + Math.abs(d[1]) < 1e-5) break;
    }
    M = { a, b, tx: tx / s, ty: ty / s };
  }
  return { M, score };
}

function solve4(H, g) {
  const A = [0, 1, 2, 3].map(i => [H[i * 4], H[i * 4 + 1], H[i * 4 + 2], H[i * 4 + 3], g[i]]);
  for (let c = 0; c < 4; c++) {
    let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 4; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k < 5; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((r, i) => r[4] / r[i]);
}

// rotation+translation brute force at the top level (NCC)
function coarseRotShift(F0, F1) {
  const L = F0.P.length - 1, A = F0.P[L], B = F1.P[L], s = 1 << L;
  const hw = A.w / 2, hh = A.h / 2, R = Math.ceil(A.w * 0.3);
  const out = [];
  for (let deg = -16; deg <= 16; deg += 4) {
    const th = deg * Math.PI / 180, c = Math.cos(th), sn = Math.sin(th);
    for (let dy = -R; dy <= R; dy += 1) for (let dx = -R; dx <= R; dx += 1) {
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
      for (let y = 1; y < A.h - 1; y++) for (let x = 1; x < A.w - 1; x++) {
        const xc = x - hw, yc = y - hh, u = c * xc - sn * yc + dx + hw, v = sn * xc + c * yc + dy + hh;
        const ui = Math.round(u), vi = Math.round(v);
        if (ui < 0 || vi < 0 || ui >= B.w || vi >= B.h) continue;
        const i0 = A.d[y * A.w + x], i1 = B.d[vi * B.w + ui];
        sa += i0; sb += i1; saa += i0 * i0; sbb += i1 * i1; sab += i0 * i1; n++;
      }
      if (n < A.w * A.h * 0.35) continue;
      const ma = sa / n, mb = sb / n, va = saa / n - ma * ma, vb = sbb / n - mb * mb;
      if (va < 1e-6 || vb < 1e-6) continue;
      const ncc = (sab / n - ma * mb) / Math.sqrt(va * vb) - 0.15 * (1 - n / (A.w * A.h));
      out.push([ncc, { a: c, b: sn, tx: dx * s, ty: dy * s }]);
    }
  }
  out.sort((p, q) => q[0] - p[0]);
  return out.slice(0, 3).map(o => o[1]);
}

// ---------- 2x3 similarity matrix helpers (centered coords) ----------
const S = {
  mul(A, B) { // A∘B
    return { a: A.a * B.a - A.b * B.b, b: A.b * B.a + A.a * B.b,
             tx: A.a * B.tx - A.b * B.ty + A.tx, ty: A.b * B.tx + A.a * B.ty + A.ty };
  },
  inv(A) {
    const d = A.a * A.a + A.b * A.b, ia = A.a / d, ib = -A.b / d;
    return { a: ia, b: ib, tx: -(ia * A.tx - ib * A.ty), ty: -(ib * A.tx + ia * A.ty) };
  },
  fromP(tx, ty, ang, ls) { const s = Math.exp(ls); return { a: s * Math.cos(ang), b: s * Math.sin(ang), tx, ty }; },
  toP(A) { return [A.tx, A.ty, Math.atan2(A.b, A.a), Math.log(Math.hypot(A.a, A.b))]; },
  apply(A, x, y) { return [A.a * x - A.b * y + A.tx, A.b * x + A.a * y + A.ty]; },
};

function gauss(arr, sigma) {
  const n = arr.length; if (sigma <= 0.01) return arr.slice();
  if (sigma > n * 4) { // effectively infinite: linear-fit-free constant (mean)
    const m = arr.reduce((a, b) => a + b, 0) / n; return arr.map(() => m);
  }
  const r = Math.ceil(sigma * 3), k = [];
  let ks = 0; for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); ks += v; }
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = -r; j <= r; j++) {
      let t = i + j;
      // reflect around ends (odd reflection keeps trend at clip ends)
      let v;
      if (t < 0) v = 2 * arr[0] - arr[Math.min(n - 1, -t)];
      else if (t >= n) v = 2 * arr[n - 1] - arr[Math.max(0, 2 * n - 2 - t)];
      else v = arr[t];
      s += k[j + r] * v;
    }
    out[i] = s / ks;
  }
  return out;
}

// Box-constrained smooth path: min  w1|D1 p|^2 + w2|D2 p|^2 + w3|D3 p|^2 + eps|p-t|^2,  lo<=p<=hi
// solved with FISTA (accelerated projected gradient). Convex -> no jitter at the limits.
function boxSmooth(t, lo, hi, smoothFrames, iters = 3000) {
  const n = t.length; if (n < 4) return t.slice();
  const L1 = 1, L2 = smoothFrames ** 2, L3 = smoothFrames ** 3 * 0.5, eps = 1e-4;
  const Lip = 2 * (4 * L1 + 16 * L2 + 64 * L3 + eps);
  const step = 1 / Lip;
  let x = t.map((v, i) => Math.min(hi[i], Math.max(lo[i], v))), y = x.slice(), tk = 1;
  const g = new Float64Array(n), d1 = new Float64Array(n), d2 = new Float64Array(n), d3 = new Float64Array(n);
  for (let it = 0; it < iters; it++) {
    g.fill(0);
    // gradient of sum (D p)^2 = 2 D^T D p, for D1, D2, D3
    for (let i = 0; i < n - 1; i++) { const r = y[i + 1] - y[i]; g[i] -= 2 * L1 * r; g[i + 1] += 2 * L1 * r; }
    for (let i = 0; i < n - 2; i++) { const r = y[i] - 2 * y[i + 1] + y[i + 2]; g[i] += 2 * L2 * r; g[i + 1] -= 4 * L2 * r; g[i + 2] += 2 * L2 * r; }
    for (let i = 0; i < n - 3; i++) { const r = -y[i] + 3 * y[i + 1] - 3 * y[i + 2] + y[i + 3]; g[i] -= 2 * L3 * r; g[i + 1] += 6 * L3 * r; g[i + 2] -= 6 * L3 * r; g[i + 3] += 2 * L3 * r; }
    const xn = new Array(n);
    for (let i = 0; i < n; i++) { const v = y[i] - step * (g[i] + 2 * eps * (y[i] - t[i])); xn[i] = v < lo[i] ? lo[i] : v > hi[i] ? hi[i] : v; }
    const tn = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2, f = (tk - 1) / tn;
    for (let i = 0; i < n; i++) y[i] = xn[i] + f * (xn[i] - x[i]);
    x = xn; tk = tn;
  }
  return x;
}

/**
 * motions[i] maps frame i-1 -> frame i (centered similarity; motions[0] ignored).
 * opts: { w, h, fps, crop (per side 0.05..0.35), smoothSec, lock, fillEdges }
 * returns { corrections: [{a,b,tx,ty}] (centered, analysis px), zoom, stats }
 *   output_pixel = zoom * K_i(frame_pixel)
 */
function solvePath(motions, opts) {
  const n = motions.length, { w, h } = opts, fps = opts.fps ?? 30;
  const crop = opts.crop ?? 0.2, zoom = 1 / (1 - 2 * crop);
  const hw = w / 2, hh = h / 2;
  // camera trajectory (accumulated per-frame motion; camera moves opposite to image)
  const T = [[0], [0], [0], [0]];
  for (let i = 1; i < n; i++) {
    const m = motions[i];
    T[0].push(T[0][i - 1] + m.tx); T[1].push(T[1][i - 1] + m.ty);
    T[2].push(T[2][i - 1] + Math.atan2(m.b, m.a)); T[3].push(T[3][i - 1] + Math.log(Math.hypot(m.a, m.b)));
  }
  const smoothF = opts.lock ? n * 3 : Math.max(2, (opts.smoothSec ?? 1.5) * fps);
  const extra = opts.fillEdges ? 1.6 : 1.0;   // edge filling lets us move further than the crop
  // 1) rotation: keep horizon steady, bounded by what the crop can hide
  const rotMax = Math.min(Math.asin(Math.min(1, crop * 1.2)), 12 * Math.PI / 180) * extra;
  const rs = boxSmooth(T[2], T[2].map(v => v - rotMax), T[2].map(v => v + rotMax), smoothF);
  // 2) zoom drift (very slow)
  const zs = boxSmooth(T[3], T[3].map(v => v - 0.05), T[3].map(v => v + 0.05), smoothF * 2);
  // 3) translation bounds depend on the rotation correction actually used
  const bx = [], by = [];
  for (let i = 0; i < n; i++) {
    const th = Math.abs(rs[i] - T[2][i]), c = Math.cos(th), sn = Math.sin(th);
    const sc = Math.exp(zs[i] - T[3][i]);
    bx.push(Math.max(0, (hw * extra - (hw * c + hh * sn) / (zoom * sc)) * 0.98));
    by.push(Math.max(0, (hh * extra - (hw * sn + hh * c) / (zoom * sc)) * 0.98));
  }
  const xs = boxSmooth(T[0], T[0].map((v, i) => v - bx[i]), T[0].map((v, i) => v + bx[i]), smoothF);
  const ys = boxSmooth(T[1], T[1].map((v, i) => v - by[i]), T[1].map((v, i) => v + by[i]), smoothF);
  // corrections: rotate by (smooth - raw) about centre, scale, then shift by (smooth - raw)
  const corrections = [];
  for (let i = 0; i < n; i++) {
    const th = rs[i] - T[2][i], sc = Math.exp(zs[i] - T[3][i]);
    // a world point sits at x0 + T_i in frame i; we want it at x0 + S_i  =>  shift by S_i - T_i
    corrections.push({ a: sc * Math.cos(th), b: sc * Math.sin(th), tx: xs[i] - T[0][i], ty: ys[i] - T[1][i] });
  }
  const jerk = a => { let s = 0; for (let i = 2; i < a.length; i++) s += Math.abs(a[i] - 2 * a[i - 1] + a[i - 2]); return s / Math.max(1, a.length - 2); };
  const before = jerk(T[0]) + jerk(T[1]), after = jerk(xs) + jerk(ys);
  return { corrections, zoom, stats: { before, after, reduction: before ? 1 - after / before : 0 } };
}

return { makeFrame, estimateMotion, solvePath, S };
 };

  const ANALYSIS_W = 320;

  function makeWorker() {
    const src = `const MKStabCore = (${CORE_FACTORY.toString()})();
      let prev = null, motions = [], W = 0, H = 0;
      onmessage = (e) => {
        const m = e.data;
        if (m.type === 'reset') { prev = null; motions = []; W = m.w; H = m.h; return; }
        if (m.type === 'frame') {
          const cur = MKStabCore.makeFrame(new Uint8Array(m.gray), W, H);
          if (!prev) motions.push({ a: 1, b: 0, tx: 0, ty: 0, method: 'first' });
          else motions.push(MKStabCore.estimateMotion(prev, cur));
          prev = cur;
          postMessage({ type: 'frameDone', index: m.index });
          return;
        }
        if (m.type === 'setMotions') { motions = m.motions; return; }
        if (m.type === 'solve') {
          const r = MKStabCore.solvePath(motions, m.opts);
          postMessage({ type: 'solved', result: r, motions });
        }
      };`;
    return new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  }

  const MODES = {
    lock: { lock: true },
    cinematic: { smoothSec: 3 },
    natural: { smoothSec: 1 },
  };

  class MKStabilizer {
    constructor() {
      this.video = null; this.fps = 30; this.duration = 0;
      this.times = []; this.motions = null; this.result = null;
      this.opts = { mode: 'lock', crop: 0.2, edgeFill: 'blur' };
      this.worker = makeWorker();
      this._blurCanvas = document.createElement('canvas');
      this._blurCanvas.width = 48; this._blurCanvas.height = 27;
      this._prevCanvas = null;
    }

    /** Accepts File/Blob, URL string or an existing <video>. */
    async load(src) {
      let v;
      if (src instanceof HTMLVideoElement) v = src;
      else {
        v = document.createElement('video');
        v.src = (src instanceof Blob) ? URL.createObjectURL(src) : src;
      }
      v.muted = true; v.playsInline = true; v.preload = 'auto'; v.crossOrigin = 'anonymous';
      if (v.readyState < 1) await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error('Cannot load video')); });
      this.video = v; this.duration = v.duration;
      this.fps = await this._detectFps(v);
      return { width: v.videoWidth, height: v.videoHeight, duration: v.duration, fps: this.fps };
    }

    async _detectFps(v) {
      if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) return 30;
      const ts = [];
      await new Promise((resolve) => {
        const done = () => { v.pause(); resolve(); };
        const cb = (_, meta) => { ts.push(meta.mediaTime); if (ts.length >= 12) done(); else v.requestVideoFrameCallback(cb); };
        v.requestVideoFrameCallback(cb);
        v.currentTime = 0; v.play().catch(done);
        setTimeout(done, 1500);
      });
      v.pause(); v.currentTime = 0;
      const d = []; for (let i = 1; i < ts.length; i++) if (ts[i] > ts[i - 1]) d.push(ts[i] - ts[i - 1]);
      if (!d.length) return 30;
      d.sort((a, b) => a - b);
      const fps = 1 / d[Math.floor(d.length / 2)];
      return Math.min(120, Math.max(10, Math.round(fps * 100) / 100));
    }

    _seek(t) {
      const v = this.video;
      return new Promise((res) => {
        const on = () => { v.removeEventListener('seeked', on); res(); };
        v.addEventListener('seeked', on); v.currentTime = t;
      });
    }

    /** Motion analysis. Seeks frame by frame (frame-accurate), estimation runs in a worker. */
    async analyze({ onProgress, signal } = {}) {
      const v = this.video; if (!v) throw new Error('load() first');
      const aw = ANALYSIS_W, ah = Math.round(aw * v.videoHeight / v.videoWidth / 2) * 2;
      this.aw = aw; this.ah = ah;
      const c = document.createElement('canvas'); c.width = aw; c.height = ah;
      const cx = c.getContext('2d', { willReadFrequently: true });
      const n = Math.max(2, Math.floor(v.duration * this.fps));
      this.times = [];
      this.worker.postMessage({ type: 'reset', w: aw, h: ah });
      let inflight = 0, doneCount = 0, wake = null;
      this.worker.onmessage = (e) => {
        if (e.data.type === 'frameDone') { inflight--; doneCount++; onProgress && onProgress(doneCount / n); wake && wake(); }
      };
      v.pause();
      for (let i = 0; i < n; i++) {
        if (signal && signal.aborted) throw new Error('aborted');
        const t = Math.min(v.duration - 0.001, (i + 0.5) / this.fps);
        await this._seek(t);
        cx.drawImage(v, 0, 0, aw, ah);
        const rgba = cx.getImageData(0, 0, aw, ah).data, g = new Uint8Array(aw * ah);
        for (let p = 0, q = 0; p < g.length; p++, q += 4) g[p] = (rgba[q] * 77 + rgba[q + 1] * 150 + rgba[q + 2] * 29) >> 8;
        this.times.push(t);
        while (inflight >= 3) await new Promise(r => (wake = r));
        inflight++;
        this.worker.postMessage({ type: 'frame', gray: g.buffer, index: i }, [g.buffer]);
      }
      while (inflight > 0) await new Promise(r => (wake = r));
      await this.configure(this.opts);
      return this.result.stats;
    }

    /** Change mode / crop / edge fill. Fast (no re-analysis). */
    configure(o = {}) {
      Object.assign(this.opts, o);
      const m = MODES[this.opts.mode] || MODES.lock;
      const st = this.opts.strength ?? 1;
      const opts = { w: this.aw, h: this.ah, fps: this.fps, crop: this.opts.crop,
        fillEdges: !['none', 'zoom'].includes(this.opts.edgeFill), ...m };
      if (opts.smoothSec) opts.smoothSec *= st;
      return new Promise((res) => {
        this.worker.onmessage = (e) => {
          if (e.data.type === 'solved') { this.result = e.data.result; this.motions = e.data.motions; this._prevCanvas = null; res(this.result.stats); }
        };
        this.worker.postMessage({ type: 'solve', opts });
      });
    }

    /** Correction for a media time (interpolated between analysed frames). */
    transformAt(t) {
      const C = this.result.corrections, n = C.length;
      let f = t * this.fps - 0.5; if (f < 0) f = 0; if (f > n - 1) f = n - 1;
      const i = Math.floor(f), j = Math.min(n - 1, i + 1), u = f - i, A = C[i], B = C[j];
      return { a: A.a + (B.a - A.a) * u, b: A.b + (B.b - A.b) * u, tx: A.tx + (B.tx - A.tx) * u, ty: A.ty + (B.ty - A.ty) * u };
    }

    /** Draw the stabilized frame of `video` at `mediaTime` into ctx (w x h). */
    drawFrame(ctx, video, mediaTime, w, h) {
      if (!this.result) { ctx.drawImage(video, 0, 0, w, h); return; }
      const K = this.transformAt(mediaTime), z = this.result.zoom;
      const Wv = video.videoWidth, Hv = video.videoHeight, k = Wv / this.aw, r = w / Wv;
      const A = K.a * z, B = K.b * z, TX = K.tx * k * z, TY = K.ty * k * z;
      const cx = Wv / 2, cy = Hv / 2;
      const e = TX + cx - (A * cx - B * cy), f = TY + cy - (B * cx + A * cy);
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const fill = this.opts.edgeFill;
      if (fill === 'blur') {
        const bc = this._blurCanvas, bx = bc.getContext('2d');
        bx.drawImage(video, 0, 0, bc.width, bc.height);
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(bc, -w * 0.1, -h * 0.1, w * 1.2, h * 1.2);
        ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(0, 0, w, h);
      } else if (fill === 'previous' && this._prevCanvas) {
        ctx.drawImage(this._prevCanvas, 0, 0, w, h);
      } else {
        ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
      }
      ctx.setTransform(A * r, B * r, -B * r, A * r, e * r, f * r);
      ctx.drawImage(video, 0, 0);
      ctx.restore();
      if (fill === 'previous') {
        if (!this._prevCanvas) { this._prevCanvas = document.createElement('canvas'); }
        if (this._prevCanvas.width !== w) { this._prevCanvas.width = w; this._prevCanvas.height = h; }
        this._prevCanvas.getContext('2d').drawImage(ctx.canvas, 0, 0);
      }
    }

    /** Render the whole clip to a video Blob (real-time speed). */
    async export({ onProgress, maxWidth = 1920, bitrate } = {}) {
      const v = this.video;
      const s = Math.min(1, maxWidth / v.videoWidth);
      const w = Math.round(v.videoWidth * s / 2) * 2, h = Math.round(v.videoHeight * s / 2) * 2;
      const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      const stream = canvas.captureStream(this.fps);
      // audio
      let ac = null;
      try {
        if (!this._audioCtx) {
          this._audioCtx = new AudioContext();
          this._audioSrc = this._audioCtx.createMediaElementSource(v);
          this._audioDest = this._audioCtx.createMediaStreamDestination();
          this._audioSrc.connect(this._audioDest);
        }
        ac = this._audioCtx; await ac.resume();
        this._audioDest.stream.getAudioTracks().forEach(t => stream.addTrack(t));
      } catch (e) { /* no audio */ }
      const types = ['video/mp4;codecs=avc1.640028,mp4a.40.2', 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4;codecs=avc1,opus', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm'];
      const mimeType = types.find(t => MediaRecorder.isTypeSupported(t)) || '';
      const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: bitrate || Math.round(w * h * this.fps * 0.15) });
      const chunks = []; rec.ondataavailable = e => e.data.size && chunks.push(e.data);
      const finished = new Promise(r => (rec.onstop = r));
      this._prevCanvas = null;
      await this._seek(0);
      v.muted = false; v.volume = 1;
      const draw = (now, meta) => {
        this.drawFrame(ctx, v, meta ? meta.mediaTime : v.currentTime, w, h);
        onProgress && onProgress(v.currentTime / v.duration);
        if (!v.ended && !v.paused) v.requestVideoFrameCallback(draw);
      };
      this.drawFrame(ctx, v, 0, w, h);
      rec.start(250);
      v.requestVideoFrameCallback(draw);
      await v.play();
      await new Promise(r => (v.onended = r));
      rec.stop(); await finished;
      v.muted = true; v.onended = null;
      onProgress && onProgress(1);
      return new Blob(chunks, { type: (mimeType || 'video/webm').split(';')[0] });
    }

    /** Peak per-frame shake in source pixels / degrees (for telemetry). */
    peakShake() {
      if (!this.motions || !this.video) return { px: 0, deg: 0 };
      const k = this.video.videoWidth / this.aw; let px = 0, deg = 0;
      for (const m of this.motions) { px = Math.max(px, Math.hypot(m.tx, m.ty) * k); deg = Math.max(deg, Math.abs(Math.atan2(m.b, m.a)) * 180 / Math.PI); }
      return { px, deg };
    }

    /** Save/restore analysis so MK Vision projects don't need to re-analyse. */
    toJSON() { return { v: 1, fps: this.fps, aw: this.aw, ah: this.ah, motions: this.motions, opts: this.opts }; }
    async fromJSON(d) {
      this.fps = d.fps; this.aw = d.aw; this.ah = d.ah; this.opts = d.opts;
      this.worker.postMessage({ type: 'setMotions', motions: d.motions });
      return this.configure(this.opts);
    }
    dispose() { this.worker.terminate(); }
  }

  global.MKStabilizer = MKStabilizer;
})(typeof window !== 'undefined' ? window : globalThis);
