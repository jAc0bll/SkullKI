// Two simulated players play a full game against two bots on a local server.
//   PORT=8799 node index.mjs &  node smoke.mjs ws://localhost:8799
import WebSocket from 'ws';
const url = process.argv[2] ?? 'ws://localhost:8787';
function client(name) {
  const ws = new WebSocket(url);
  const c = { ws, name, view: null, room: null, seat: -1, done: null };
  c.send = (m) => ws.send(JSON.stringify(m));
  c.finished = new Promise((res) => (c.done = res));
  ws.on('message', (d) => {
    const m = JSON.parse(String(d));
    if (m.t === 'joined') c.seat = m.seat, c.code = m.code;
    if (m.t === 'room') c.room = m;
    if (m.t === 'error') console.log(name, 'error:', m.msg);
    if (m.t === 'state') {
      const v = (c.view = m.view);
      if (v.phase === 'gameOver') return c.done(v);
      if (v.phase === 'roundEnd' && !c.room.ready.includes(c.seat)) return c.send({ t: 'next' });
      if (v.toAct.includes(v.seat) && v.legal.length) {
        const a = v.legal[Math.floor(Math.random() * v.legal.length)].a;
        setTimeout(() => c.send({ t: 'act', a }), 5);
      }
    }
  });
  return new Promise((res) => ws.on('open', () => res(c)));
}
const a = await client('Alice');
a.send({ t: 'create', name: 'Alice', account: process.env.ACCOUNT });
await new Promise((r) => setTimeout(r, 200));
const b = await client('Bob');
b.send({ t: 'join', code: a.code, name: 'Bob' });
await new Promise((r) => setTimeout(r, 200));
console.log('room', a.code, a.room.players.map((p) => p.name));
a.send({ t: 'start' });
const t0 = Date.now();
const [va, vb] = await Promise.all([a.finished, b.finished]);
console.log('game over after', ((Date.now() - t0) / 1000).toFixed(0), 's; scores', va.scores, 'rounds', va.results.length,
  '| same scores for both:', JSON.stringify(va.scores) === JSON.stringify(vb.scores), '| Bob review items', vb.review.length);
process.exit(0);
