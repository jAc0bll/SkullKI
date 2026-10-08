// Players: a name (no password yet), an avatar and statistics, kept in one
// JSON file (server/data/users.json). Written atomically, debounced.
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const FILE = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data', 'users.json');
const STYLES = ['adventurer', 'avataaars', 'lorelei', 'notionists', 'pixel-art', 'fun-emoji', 'bottts'];

/** @type {Record<string, any>} key = lowercase name */
let users = {};
if (existsSync(FILE)) users = JSON.parse(readFileSync(FILE, 'utf8'));

let saveTimer = null;
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    mkdirSync(path.dirname(FILE), { recursive: true });
    writeFileSync(FILE + '.tmp', JSON.stringify(users, null, 1));
    renameSync(FILE + '.tmp', FILE);
  }, 300);
}

export function cleanName(name) {
  const n = String(name ?? '').trim().replace(/\s+/g, ' ');
  if (n.length < 2 || n.length > 20) return null;
  if (!/^[\p{L}\p{N} _.-]+$/u.test(n)) return null;
  return n;
}

export function cleanAvatar(a) {
  if (!a || typeof a !== 'object') return null;
  const style = STYLES.includes(a.style) ? a.style : null;
  const seed = String(a.seed ?? '').slice(0, 40);
  const hex = (x) => /^([0-9a-f]{6}|transparent)$/i.test(x ?? '');
  const out = { style, seed };
  if (hex(a.bg)) out.bg = a.bg.toLowerCase();
  // hand-picked parts and colours (names/values as DiceBear uses them)
  const parts = Object.entries(a.parts ?? {}).filter(
    ([k, v]) => /^[a-zA-Z]{1,24}$/.test(k) && (v === null || /^[a-zA-Z0-9]{1,40}$/.test(String(v))),
  );
  const colors = Object.entries(a.colors ?? {}).filter(([k, v]) => /^[a-zA-Z]{1,30}$/.test(k) && hex(v));
  if (parts.length) out.parts = Object.fromEntries(parts.slice(0, 24));
  if (colors.length) out.colors = Object.fromEntries(colors.slice(0, 16).map(([k, v]) => [k, v.toLowerCase()]));
  return style && seed ? out : null;
}

const emptyStats = () => ({
  games: 0, wins: 0, points: 0, best: null, rounds: 0, bidsHit: 0, gtoAgree: 0, gtoTotal: 0,
  bot: { games: 0, wins: 0 }, online: { games: 0, wins: 0 },
});

export const publicUser = (u) => u && { name: u.name, avatar: u.avatar, createdAt: u.createdAt, stats: u.stats, history: u.history };

export function getUser(name) {
  return users[String(name ?? '').trim().toLowerCase()] ?? null;
}

/** Log in by name; a new name creates the player. */
export function login(name, avatar) {
  const n = cleanName(name);
  if (!n) return { error: 'Name: 2–20 Zeichen, Buchstaben, Zahlen, Leerzeichen, _ . -' };
  const key = n.toLowerCase();
  if (!users[key]) {
    users[key] = {
      name: n,
      avatar: cleanAvatar(avatar) ?? { style: 'adventurer', seed: n },
      createdAt: new Date().toISOString(),
      stats: emptyStats(),
      history: [],
    };
    save();
  }
  return { user: users[key] };
}

export function setAvatar(name, avatar) {
  const u = getUser(name);
  const a = cleanAvatar(avatar);
  if (!u || !a) return null;
  u.avatar = a;
  save();
  return u;
}

/**
 * One finished game for one player.
 * g = {mode:'bot'|'online', score, rank (1-4), rounds, bidsHit, gtoAgree, gtoTotal, players:[names]}
 */
export function recordGame(name, g) {
  const u = getUser(name);
  if (!u) return null;
  const num = (x, max) => Math.max(0, Math.min(max, Math.round(Number(x) || 0)));
  const mode = g.mode === 'online' ? 'online' : 'bot';
  const score = Math.max(-2000, Math.min(2000, Math.round(Number(g.score) || 0)));
  const rank = num(g.rank, 4) || 4;
  const s = u.stats;
  s.games += 1;
  s.points += score;
  s.best = s.best === null ? score : Math.max(s.best, score);
  s.rounds += num(g.rounds, 10);
  s.bidsHit += num(g.bidsHit, 10);
  s.gtoAgree += num(g.gtoAgree, 500);
  s.gtoTotal += num(g.gtoTotal, 500);
  s[mode].games += 1;
  if (rank === 1) {
    s.wins += 1;
    s[mode].wins += 1;
  }
  u.history.unshift({
    at: new Date().toISOString(),
    mode,
    score,
    rank,
    players: (Array.isArray(g.players) ? g.players : []).slice(0, 4).map((p) => String(p).slice(0, 24)),
  });
  u.history = u.history.slice(0, 30);
  save();
  return u;
}

export function leaderboard() {
  return Object.values(users)
    .filter((u) => u.stats.games > 0)
    .map((u) => ({
      name: u.name,
      avatar: u.avatar,
      games: u.stats.games,
      wins: u.stats.wins,
      avg: Math.round(u.stats.points / u.stats.games),
    }))
    .sort((a, b) => b.wins - a.wins || b.avg - a.avg)
    .slice(0, 20);
}

/** Summary of a finished game for `seat`, from its final view (session.cpp). */
export function summarize(view, seat) {
  const order = [0, 1, 2, 3].sort((a, b) => view.scores[b] - view.scores[a]);
  const rank = 1 + order.filter((s) => view.scores[s] > view.scores[seat]).length;
  return {
    score: view.scores[seat],
    rank,
    rounds: view.results.length,
    bidsHit: view.results.filter((r) => r.bids[seat] === r.won[seat]).length,
    gtoAgree: view.review.filter((r) => r.chosen === r.best).length,
    gtoTotal: view.review.length,
  };
}
