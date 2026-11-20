// GTA VI trophy rarity forecast (Node 22, standard library only).
//
// Model: forecast each trophy's Exophase earn rate at the case target.
//
// A rate is earners over players tracked, and the two move differently:
// Exophase refreshes its counts on its own schedule, hours apart, so a
// read shows the rates as of the last refresh, not as of the read. Between
// refreshes the page is frozen; at a refresh the player count jumps and
// every rate drops (dilution) unless earners grew as fast. So the history
// is first reduced to refresh snapshots (consecutive identical reads are
// one state, timed at its first read), and each trophy's rate is then
// fitted with a saturating curve in time since release:
//
//   r_i(t) = A_i * (1 - exp(-lambda_i * t))
//
// the cumulative shape of players gradually earning a trophy: fast at
// first for easy trophies, slow for late-game ones, levelling off toward
// an asymptote A_i (the eventual share of players who ever earn it). The
// rate lambda_i is pooled by tier (platinum/gold/silver/bronze) -- a
// trophy with little history borrows its tier's typical speed -- and A_i
// is then fitted per trophy at that pooled speed. A power law was tried
// first and rejected: on a synthetic week-one case it extrapolated the
// early steep climb and overshot common trophies by ~40 points, while the
// saturating fit tracks the levelling-off.
//
// A flat history (a single snapshot, as on launch night) carries no trend,
// so the forecast falls back to a refresh prior taken from the one refresh
// the rules document: launch night the page sat at 638 players, then
// refreshed to 3,868 (x6.06) while earners grew only ~x1.2, so every rate
// but the 100% one fell to roughly a fifth. The fallback therefore expects
// one refresh before the target with P x6.06 and E x1.2, with a wide
// spread, because the next refresh's size is genuinely uncertain.
//
// Quantiles are the median rate in logit space plus/minus z_tau * s, where
// s combines a random-walk term that grows with the horizon (0.5 logit
// units per sqrt day, as the house baseline), the per-trophy fit residual
// in logit space, and an extra term when the history is flat. A forecast
// that is too narrow pays for every trophy outside it, one too wide pays
// at every trophy, so the spread is meant to be honest, not minimal.
//
// Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
// (else /output), one <id>.out per <id>.in. Deterministic: no randomness,
// no clock, no network; the same input always yields the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The standard normal's quantile at each level (0.05, 0.25, 0.5, 0.75, 0.95). */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant; t is measured in hours from here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Random-walk spread of a rate's logit move, per square-root day. */
const SIGMA_RW = 0.5;
/** Extra logit spread when the history is a single flat snapshot. */
const SIGMA_FLAT = 0.6;
/** Documented first refresh: players 638 -> 3,868. */
const REFRESH_PLAYER_FACTOR = 3868 / 638;
/** Earners grew only ~x1.2 across that same refresh (implied by the house
 *  baseline's published practice loss ~1.51 and typical miss ~0.68). */
const REFRESH_EARNER_FACTOR = 1.2;
/** Candidate earning speeds, per hour, for the saturating curve. */
const LAMBDAS = [0.002, 0.004, 0.007, 0.012, 0.02, 0.035, 0.06, 0.1, 0.18, 0.32, 0.6];
/** Recency half-life for history weights, in hours. */
const HALF_LIFE_H = 36;

interface TrophyMeta {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: TrophyMeta[];
  history: Read[];
}

interface Snap {
  t: number;
  atMs: number;
  p: number;
  rates: number[];
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logit(pct: number): number {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  return 100 / (1 + Math.exp(-l));
}

/** Hours from release to an ISO timestamp (floored at 0 before release). */
function hoursSinceRelease(iso: string): number {
  return Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3600000);
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((p, q) => p - q);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Collapse consecutive identical reads into refresh snapshots. */
function snapshots(c: Case): Snap[] {
  const out: Snap[] = [];
  for (const h of c.history) {
    const rates = h.rates.map((r) => (r === null ? NaN : r));
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.p === h.players_tracked &&
      prev.rates.length === rates.length &&
      prev.rates.every((v, i) => (Number.isNaN(v) && Number.isNaN(rates[i])) || v === rates[i])
    ) {
      continue;
    }
    out.push({ t: hoursSinceRelease(h.at), atMs: Date.parse(h.at), p: h.players_tracked, rates });
  }
  return out;
}

/**
 * Fit r(t) = A * (1 - exp(-lambda * t)) to a trophy's snapshot rates by
 * weighted least squares over the lambda grid. Returns the best lambda, A
 * and weighted SSE. With lambda fixed, A has a closed form.
 */
function fitSaturating(
  ts: number[],
  rs: number[],
  ws: number[],
  lambdaFixed?: number,
): { lambda: number; a: number; sse: number } {
  const grid = lambdaFixed === undefined ? LAMBDAS : [lambdaFixed];
  let best = { lambda: grid[0], a: 0, sse: Infinity };
  for (const lambda of grid) {
    let num = 0;
    let den = 0;
    for (let k = 0; k < ts.length; k++) {
      const x = 1 - Math.exp(-lambda * ts[k]);
      num += ws[k] * x * rs[k];
      den += ws[k] * x * x;
    }
    const a = den > 1e-12 ? clamp(num / den, 0, 100) : 0;
    let sse = 0;
    for (let k = 0; k < ts.length; k++) {
      const x = 1 - Math.exp(-lambda * ts[k]);
      const d = rs[k] - a * x;
      sse += ws[k] * d * d;
    }
    if (sse < best.sse - 1e-12) best = { lambda, a, sse };
  }
  return best;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const n = c.trophies.length;
  const snaps = snapshots(c);
  const last = snaps[snaps.length - 1];
  const tTarget = hoursSinceRelease(c.target);
  const days = Math.max(0, c.horizon_minutes / 1440);
  const flat = snaps.length < 2;

  // --- Flat history: refresh prior (see header). ---
  let pTarget = last.p * REFRESH_PLAYER_FACTOR;
  if (!flat) {
    // Players P(t): power law in log(1 + t), recency-weighted.
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    for (const s of snaps) {
      xs.push(Math.log1p(s.t));
      ys.push(Math.log(Math.max(1, s.p)));
      ws.push(Math.pow(0.5, Math.max(0, (last.atMs - s.atMs) / 3600000 / HALF_LIFE_H)));
    }
    let sw = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (let k = 0; k < xs.length; k++) {
      sw += ws[k];
      sx += ws[k] * xs[k];
      sy += ws[k] * ys[k];
      sxx += ws[k] * xs[k] * xs[k];
      sxy += ws[k] * xs[k] * ys[k];
    }
    const denom = sw * sxx - sx * sx;
    let b = 0;
    let a = sy / Math.max(sw, 1e-9);
    if (Math.abs(denom) > 1e-12) {
      b = (sw * sxy - sx * sy) / denom;
      a = (sy - b * sx) / sw;
    }
    pTarget = clamp(Math.exp(a + b * Math.log1p(tTarget)), last.p, last.p * 1e4);
  }

  // --- Per-trophy saturating fits; pool lambda by tier. ---
  interface Fit {
    lambda: number;
    a: number;
    residVar: number;
    nPts: number;
    rLast: number;
    eLast: number;
  }
  const fits: Fit[] = [];
  const byTier = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const ts: number[] = [];
    const rs: number[] = [];
    const ws: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (!Number.isNaN(r)) {
        ts.push(s.t);
        rs.push(r);
        ws.push(Math.pow(0.5, Math.max(0, (last.atMs - s.atMs) / 3600000 / HALF_LIFE_H)));
      }
    }
    const rLast = last.rates[i];
    const eLast = Number.isNaN(rLast) ? NaN : Math.round((rLast * last.p) / 100);
    if (ts.length >= 2) {
      const f = fitSaturating(ts, rs, ws);
      fits.push({ lambda: f.lambda, a: f.a, residVar: 0, nPts: ts.length, rLast, eLast });
      const tier = c.trophies[i].tier;
      if (!byTier.has(tier)) byTier.set(tier, []);
      byTier.get(tier)!.push(f.lambda);
    } else {
      fits.push({ lambda: NaN, a: NaN, residVar: 0, nPts: ts.length, rLast, eLast });
    }
  }
  const allLambdas = fits.filter((f) => !Number.isNaN(f.lambda)).map((f) => f.lambda);
  const globalLambda = allLambdas.length > 0 ? median(allLambdas) : 0.02;
  // Refit A at the pooled tier lambda, and measure logit residuals.
  for (let i = 0; i < n; i++) {
    const f = fits[i];
    if (f.nPts < 2) continue;
    const tierLambdas = byTier.get(c.trophies[i].tier) ?? [];
    const lambda = tierLambdas.length > 0 ? median(tierLambdas) : globalLambda;
    const ts: number[] = [];
    const rs: number[] = [];
    const ws: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (!Number.isNaN(r)) {
        ts.push(s.t);
        rs.push(r);
        ws.push(Math.pow(0.5, Math.max(0, (last.atMs - s.atMs) / 3600000 / HALF_LIFE_H)));
      }
    }
    const refit = fitSaturating(ts, rs, ws, lambda);
    f.lambda = lambda;
    f.a = refit.a;
    let sw = 0;
    let se = 0;
    for (let k = 0; k < ts.length; k++) {
      const pred = refit.a * (1 - Math.exp(-lambda * ts[k]));
      const d = logit(rs[k]) - logit(pred);
      se += ws[k] * d * d;
      sw += ws[k];
    }
    f.residVar = sw > 0 ? se / sw : 0;
  }

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < n; i++) {
    const f = fits[i];
    let rMed: number;
    let residVar = f.residVar;
    if (!Number.isNaN(f.rLast) && f.rLast >= 99.99) {
      // The trophy everyone has: it stays at the top, refresh or not.
      rMed = 99.99;
    } else if (flat) {
      // Refresh prior: earners x1.2, players x6.06.
      const eTarget = Number.isNaN(f.eLast) ? 0 : f.eLast * REFRESH_EARNER_FACTOR;
      rMed = clamp((100 * eTarget) / pTarget, 0.01, 99.99);
      residVar = 0;
    } else if (f.nPts < 2 || Number.isNaN(f.a)) {
      // No usable trend: hold the last rate, wide spread.
      rMed = Number.isNaN(f.rLast) ? 0.01 : clamp(f.rLast, 0.01, 99.99);
      residVar = Math.max(residVar, 0.25);
    } else {
      rMed = clamp(f.a * (1 - Math.exp(-f.lambda * tTarget)), 0.01, 99.99);
    }

    let s2 = SIGMA_RW * SIGMA_RW * days + Math.min(residVar, 0.25);
    if (flat) s2 += SIGMA_FLAT * SIGMA_FLAT;
    const s = Math.sqrt(Math.max(s2, 0.0025));

    const l = logit(rMed);
    const q: number[] = [];
    for (let k = 0; k < 5; k++) {
      q.push(Math.round(invLogit(l + Z[k] * s) * 10000) / 10000);
    }
    for (let k = 1; k < 5; k++) q[k] = Math.max(q[k], q[k - 1]);
    for (let k = 0; k < 5; k++) q[k] = clamp(q[k], 0.01, 99.99);
    rows.push({ id: c.trophies[i].id, q });
  }
  return rows;
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR)
    .filter((f) => f.endsWith('.in'))
    .sort();
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, f), 'utf8')) as Case;
    const rows = forecastCase(c);
    const id = f.replace(/\.in$/, '');
    const out = JSON.stringify({ trophies: rows });
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), out, 'utf8');
  }
}

main();
