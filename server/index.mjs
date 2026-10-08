// SkullKI multiplayer server.
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
import { readFileSync } from 'node:fs';
import { randomBytes, randomInt } from 'node:crypto';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { WebSocketServer } from 'ws';

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
      players: this.players.map((p) => ({ name: p.name, bot: p.bot, online: p.bot || p.online })),
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
      this.players.push({ name, bot: true, online: true });
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
    if (v.phase === 'gameOver') return;
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

function addPlayer(room, name, ws) {
  const token = randomBytes(16).toString('hex');
  const seat = room.players.length;
  room.players.push({ name: String(name || 'Spieler').slice(0, 20), token, ws, bot: false, online: true });
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
      addPlayer(room, msg.name, ws);
      return;
    }
    case 'join': {
      const room = rooms.get(String(msg.code || '').toUpperCase().trim());
      if (!room) return err('Raum nicht gefunden');
      if (room.started) return err('Das Spiel in diesem Raum läuft schon');
      if (room.players.length >= 4) return err('Der Raum ist voll');
      addPlayer(room, msg.name, ws);
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
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' });
  res.end(`SkullKI server ok · ${rooms.size} Räume\n`);
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
