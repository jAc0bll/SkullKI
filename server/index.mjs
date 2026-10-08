// SkullKI multiplayer server. Also serves the web app (app/dist, built with
// `npx expo export -p web`), so one address gives the app and multiplayer.
//
// Rooms with a 4-letter code; up to 4 people, free seats are taken by bots.
// The server is the only one that knows all cards: it runs the same C++ game
// engine as the app (compiled to WebAssembly, app/tools/sk-node.cjs) and
// sends every player only their own view.
//
//   PORT=8787 node index.mjs
//
// Protocol (JSON over WebSocket):
//   client -> server  {t:"hello", token?}            resume a seat after reconnect
//                     {t:"create", name}             new room, you are host
//                     {t:"join", code, name}
//                     {t:"start"}                    host only
//                     {t:"act", a}                   "bid:2" | "card:13" | "tig:1"
//                     {t:"next"}                     ready for the next round
//                     {t:"leave"}
//   server -> client  {t:"joined", code, token, seat}
//                     {t:"room", code, seat, host, started, players:[{name,bot,online}], ready:[seat]}
//                     {t:"state", view}              (see solver/src/session.cpp)
//                     {t:"error", msg}
import { createRequire } from 'node:module';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { randomBytes, randomInt } from 'node:crypto';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { WebSocketServer } from 'ws';
import * as games from './games.mjs';
import * as users from './users.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT ?? 8787);
const BOT_NAMES = ['Anne Bonny', 'Blackbeard', 'Calico Jack', 'Mary Read'];

// ---- engine -----------------------------------------------------------------
const createSk = createRequire(import.meta.url)(path.join(here, '..', 'app', 'tools', 'sk-node.cjs'));
const sk = await createSk();
for (let r = 1; r <= 10; r++) {
  sk.FS.writeFile(`/r${r}.bin`, readFileSync(path.join(here, '..', 'app', 'assets', 'models', `r${r}.bin`)));
  if (!sk.ccall('sk_load', 'number', ['number', 'string'], [r, `/r${r}.bin`])) throw new Error(`model ${r}`);
}
const game = (cmd) => JSON.parse(sk.ccall('sk_spot', 'string', ['string'], [`game ${cmd}`]));

// ---- rooms ------------------------------------------------------------------
/** @type {Map<string, Room>} */
const rooms = new Map();
/** token -> {room, seat} */
const tokens = new Map();

function newCode() {
  const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
  for (;;) {
    const c = Array.from({ length: 4 }, () => A[randomInt(A.length)]).join('');
    if (!rooms.has(c)) return c;
  }
}

class Room {
  constructor(code) {
    this.code = code;
    this.players = []; // {name, token, ws, bot, online}
    this.gameId = null;
    this.ready = new Set();
    this.timer = null;
    this.nextTimer = null;
    this.lastActive = Date.now();
  }
  get started() {
    return this.gameId !== null;
  }
  touch() {
    this.lastActive = Date.now();
  }
  send(seat, msg) {
    const ws = this.players[seat]?.ws;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg));
  }
  roomMsg(seat) {
    return {
      t: 'room',
      code: this.code,
      seat,
      host: seat === 0,
      started: this.started,
      players: this.players.map((p) => ({ name: p.name, bot: p.bot, online: p.bot || p.online, avatar: p.avatar ?? null })),
      ready: [...this.ready],
    };
  }
  broadcast() {
    this.players.forEach((p, seat) => {
      if (p.bot) return;
      this.send(seat, this.roomMsg(seat));
      if (this.started) this.send(seat, { t: 'state', view: game(`view id=${this.gameId} seat=${seat}`) });
    });
  }
  start() {
    while (this.players.length < 4) {
      const used = new Set(this.players.map((p) => p.name));
      const name = BOT_NAMES.find((n) => !used.has(n));
      this.players.push({ name, bot: true, online: true, avatar: { style: 'bottts', seed: name } });
    }
    this.gameId = game(`new seed=${randomInt(2 ** 31)} start=${randomInt(4)}`).id;
    this.afterChange();
  }
  /** Re-broadcast and let bots (or stand-ins for absent players) move. */
  afterChange() {
    this.touch();
    this.broadcast();
    clearTimeout(this.timer);
    const v = game(`view id=${this.gameId} seat=0`);
    if (v.phase === 'roundEnd') {
      clearTimeout(this.nextTimer);
      this.nextTimer = setTimeout(() => this.nextRound(), 45_000); // nobody waits forever
      return;
    }
    if (v.phase === 'gameOver') {
      this.recordResults();
      return;
    }
    const due = v.toAct.filter((s) => this.players[s].bot || !this.players[s].online);
    if (!due.length) return;
    const absentOnly = due.every((s) => !this.players[s].bot);
    const trickJustEnded = v.phase === 'playing' && v.trick.cards.length === 0 && v.lastTrick;
    const delay = absentOnly ? 25_000 : v.phase === 'bidding' ? 600 : trickJustEnded ? 1600 : 900;
    this.timer = setTimeout(() => {
      for (const s of v.phase === 'bidding' ? due : due.slice(0, 1)) game(`bot id=${this.gameId} seat=${s}`);
      this.afterChange();
    }, delay);
  }
  recordResults() {
    if (this.recorded) return;
    this.recorded = true;
    const names = this.players.map((p) => p.name);
    this.players.forEach((p, seat) => {
      if (p.bot || !p.account) return;
      const view = game(`view id=${this.gameId} seat=${seat}`);
      users.recordGame(p.account, { mode: 'online', ...users.summarize(view, seat), players: names });
    });
    const accounts = this.players.map((p) => (p.account ? users.getUser(p.account) : null));
    const optedOut = this.players.some((p, s) => !p.bot && accounts[s] && !users.sharesGames(accounts[s]));
    if (!optedOut) {
      games.saveGame({
        mode: 'online',
        source: 'server',
        players: this.players.map((p, s) => ({ pid: p.bot ? 'bot' : accounts[s] ? users.anonId(accounts[s]) : games.guestId(), human: !p.bot })),
        log: JSON.parse(sk.ccall('sk_spot', 'string', ['string'], [`game log id=${this.gameId}`])),
      });
    }
  }
  nextRound() {
    clearTimeout(this.nextTimer);
    const r = game(`next id=${this.gameId}`);
    if (!r.ok) return;
    this.ready.clear();
    this.afterChange();
  }
  close() {
    clearTimeout(this.timer);
    clearTimeout(this.nextTimer);
    if (this.gameId !== null) game(`drop id=${this.gameId}`);
    for (const p of this.players) if (p.token) tokens.delete(p.token);
    rooms.delete(this.code);
  }
}

function addPlayer(room, msg, ws) {
  const token = randomBytes(16).toString('hex');
  const seat = room.players.length;
  // a logged-in player (msg.account) keeps name and avatar from the profile
  const account = msg.account ? users.getUser(msg.account) : null;
  room.players.push({
    name: account?.name ?? String(msg.name || 'Spieler').slice(0, 20),
    account: account?.name ?? null,
    avatar: account?.avatar ?? users.cleanAvatar(msg.avatar),
    token,
    ws,
    bot: false,
    online: true,
  });
  tokens.set(token, { room, seat });
  ws.ctx = { room, seat };
  ws.send(JSON.stringify({ t: 'joined', code: room.code, token, seat }));
  room.broadcast();
}

function handle(ws, msg) {
  const ctx = ws.ctx;
  const err = (m) => ws.send(JSON.stringify({ t: 'error', msg: m }));
  switch (msg.t) {
    case 'hello': {
      const found = msg.token && tokens.get(msg.token);
      if (!found) return;
      const p = found.room.players[found.seat];
      if (p.ws && p.ws !== ws) p.ws.close();
      p.ws = ws;
      p.online = true;
      ws.ctx = found;
      ws.send(JSON.stringify({ t: 'joined', code: found.room.code, token: msg.token, seat: found.seat }));
      if (found.room.started) found.room.afterChange();   // cancels a stand-in move
      else found.room.broadcast();
      return;
    }
    case 'create': {
      const room = new Room(newCode());
      rooms.set(room.code, room);
      addPlayer(room, msg, ws);
      return;
    }
    case 'join': {
      const room = rooms.get(String(msg.code || '').toUpperCase().trim());
      if (!room) return err('Raum nicht gefunden');
      if (room.started) return err('Das Spiel in diesem Raum läuft schon');
      if (room.players.length >= 4) return err('Der Raum ist voll');
      addPlayer(room, msg, ws);
      return;
    }
  }
  if (!ctx) return err('Erst einem Raum beitreten');
  const { room, seat } = ctx;
  room.touch();
  switch (msg.t) {
    case 'start':
      if (seat !== 0) return err('Nur der Host kann starten');
      if (!room.started) room.start();
      return;
    case 'act': {
      if (!room.started) return;
      const r = game(`act id=${room.gameId} seat=${seat} a=${String(msg.a).replace(/\s/g, '')}`);
      if (!r.ok) return err(r.error);
      room.afterChange();
      return;
    }
    case 'next': {
      room.ready.add(seat);
      const humans = room.players.map((p, s) => (!p.bot && p.online ? s : -1)).filter((s) => s >= 0);
      if (humans.every((s) => room.ready.has(s))) room.nextRound();
      else room.broadcast();
      return;
    }
    case 'leave': {
      const p = room.players[seat];
      tokens.delete(p.token);
      ws.ctx = null;
      if (!room.started) {
        room.players.splice(seat, 1);
        room.players.forEach((q, s) => q.token && tokens.set(q.token, { room, seat: s }));
        room.players.forEach((q) => q.ws && q.ws.ctx && (q.ws.ctx = tokens.get(q.token)));
        if (!room.players.some((q) => !q.bot)) room.close();
        else room.broadcast();
      } else {
        // a bot takes over the seat
        Object.assign(p, { bot: true, ws: null, token: null, name: `${p.name} (KI)` });
        if (!room.players.some((q) => !q.bot)) room.close();
        else room.afterChange();
      }
      return;
    }
  }
}

// ---- server -----------------------------------------------------------------
const DIST = path.join(here, '..', 'app', 'dist');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon', '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf', '.woff2': 'font/woff2', '.wasm': 'application/wasm', '.bin': 'application/octet-stream',
};
// ---- accounts API (JSON) -----------------------------------------------------
//   POST /api/login {name, avatar?}        -> user (new names are created)
//   GET  /api/users/:name                  -> user
//   POST /api/users/:name/avatar {avatar}  -> user
//   POST /api/users/:name/games {score, rank, rounds, bidsHit, gtoAgree, gtoTotal, players, log?}
//   POST /api/users/:name/settings {shareGames}
//   GET  /api/leaderboard
function api(req, res, url) {
  const send = (code, obj) => {
    res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'access-control-allow-origin': '*' });
    res.end(JSON.stringify(obj));
  };
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'access-control-allow-origin': '*',
      'access-control-allow-methods': 'GET, POST, OPTIONS',
      'access-control-allow-headers': 'content-type',
    });
    return res.end();
  }
  let body = '';
  req.on('data', (c) => {
    body += c;
    if (body.length > 400_000) req.destroy();
  });
  req.on('end', () => {
    let data = {};
    try {
      data = body ? JSON.parse(body) : {};
    } catch {
      return send(400, { error: 'Ungültige Daten' });
    }
    const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent); // ['api', ...]
    if (parts[1] === 'login' && req.method === 'POST') {
      const r = users.login(data.name, data.avatar);
      return r.error ? send(400, { error: r.error }) : send(200, users.publicUser(r.user));
    }
    if (parts[1] === 'leaderboard') return send(200, users.leaderboard());
    if (parts[1] === 'users' && parts[2]) {
      if (!parts[3] && req.method === 'GET') {
        const u = users.getUser(parts[2]);
        return u ? send(200, users.publicUser(u)) : send(404, { error: 'Unbekannter Name' });
      }
      if (parts[3] === 'avatar' && req.method === 'POST') {
        const u = users.setAvatar(parts[2], data.avatar);
        return u ? send(200, users.publicUser(u)) : send(400, { error: 'Avatar ungültig' });
      }
      if (parts[3] === 'games' && req.method === 'POST') {
        const u = users.recordGame(parts[2], { ...data, mode: 'bot' });
        if (!u) return send(404, { error: 'Unbekannter Name' });
        if (data.log && users.sharesGames(u))
          games.saveGame({
            mode: 'bot',
            source: 'app',
            players: [0, 1, 2, 3].map((s) => ({ pid: s === 0 ? users.anonId(u) : 'bot', human: s === 0 })),
            log: data.log,
          });
        return send(200, users.publicUser(u));
      }
      if (parts[3] === 'settings' && req.method === 'POST') {
        const u = users.setShareGames(parts[2], data.shareGames);
        return u ? send(200, users.publicUser(u)) : send(404, { error: 'Unbekannter Name' });
      }
    }
    send(404, { error: 'Nicht gefunden' });
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://x');
  if (url.pathname.startsWith('/api/')) return api(req, res, url);
  if (url.pathname === '/health' || !existsSync(DIST)) {
    res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' });
    res.end(`SkullKI server ok · ${rooms.size} Räume\n`);
    return;
  }
  // static web app; unknown paths are app routes -> index.html
  let file = path.normalize(path.join(DIST, decodeURIComponent(url.pathname)));
  if (!file.startsWith(DIST) || !existsSync(file) || statSync(file).isDirectory()) file = path.join(DIST, 'index.html');
  const ext = path.extname(file);
  const hashed = url.pathname.startsWith('/_expo/') || url.pathname.startsWith('/assets/');
  res.writeHead(200, {
    'content-type': TYPES[ext] ?? 'application/octet-stream',
    'cache-control': hashed ? 'public, max-age=31536000, immutable' : 'no-cache',
  });
  res.end(readFileSync(file));
});
const wss = new WebSocketServer({ server });
wss.on('connection', (ws) => {
  ws.isAlive = true;
  ws.on('pong', () => (ws.isAlive = true));
  ws.on('message', (data) => {
    try {
      handle(ws, JSON.parse(String(data)));
    } catch (e) {
      ws.send(JSON.stringify({ t: 'error', msg: 'Ungültige Nachricht' }));
      console.error(e);
    }
  });
  ws.on('close', () => {
    const ctx = ws.ctx;
    if (!ctx) return;
    const p = ctx.room.players[ctx.seat];
    if (p && p.ws === ws) {
      p.online = false;
      p.ws = null;
      if (ctx.room.started) ctx.room.afterChange();
      else ctx.room.broadcast();
    }
  });
});

// keep connections alive behind proxies, drop dead ones, forget idle rooms
setInterval(() => {
  for (const ws of wss.clients) {
    if (!ws.isAlive) ws.terminate();
    ws.isAlive = false;
    ws.ping();
  }
  for (const room of rooms.values()) if (Date.now() - room.lastActive > 3 * 3600_000) room.close();
}, 30_000);

server.listen(PORT, () => console.log(`SkullKI server on :${PORT}`));
