// Plays random spots through the WebAssembly build and compares every answer
// with the native sk_deep (same code, so probabilities must agree).
//   node web/wasm/check.mjs <models dir with r1.bin..r10.bin> <path to sk_deep> [models dir for sk_deep]
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import createSk from "../src/wasm/sk.js";

const [modelDir, native, nativeModelDir = modelDir] = process.argv.slice(2);
const sk = await createSk();
const spot = sk.cwrap("sk_spot", "string", ["string"]);
const load = sk.cwrap("sk_load", "number", ["number", "string"]);

function deal(round) {
  const deck = [];
  for (let k = 0; k < 61; k++) for (let i = 0; i < ([56, 58].includes(k) ? 5 : k === 57 ? 2 : 1); i++) deck.push(k);
  for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  return deck.slice(0, round);
}
const pick = (opts) => opts[Math.floor(Math.random() * opts.length)];

let compared = 0, worst = 0;
for (let round = 1; round <= 10; round++) {
  sk.FS.writeFile(`/r${round}.bin`, readFileSync(`${modelDir}/r${round}.bin`));
  if (!load(round, `/r${round}.bin`)) throw new Error("load failed " + round);
  for (let game = 0; game < 3; game++) {
    const me = Math.floor(Math.random() * 4);
    const q = { round, me, hand: deal(round), bids: [-1, -1, -1, -1], play: [] };
    for (;;) {
      const text = `round=${q.round} me=${q.me} hand=${q.hand} bids=${q.bids} play=${q.play}`;
      const a = JSON.parse(spot(text));
      if (!a.ok) throw new Error(text + " -> " + a.error);
      if (game === 0 || Math.random() < 0.25) {
        const b = JSON.parse(execFileSync(native, ["spot", "--policy", `${nativeModelDir}/r${round}.bin`, text]).toString());
        const pa = [...(a.bidAdvice ?? []), ...a.options].map((o) => o.p ?? 0);
        const pb = [...(b.bidAdvice ?? []), ...b.options].map((o) => o.p ?? 0);
        if (pa.length !== pb.length) throw new Error("option count differs: " + text);
        pa.forEach((p, i) => { worst = Math.max(worst, Math.abs(p - pb[i])); });
        compared++;
      }
      if (a.phase === "done") break;
      if (a.phase === "bidding") { q.bids = q.bids.map(() => Math.floor(Math.random() * (round + 1))); continue; }
      const o = pick(a.options);
      q.play.push(o.type === "tigress" ? (o.value === "pirate" ? 241 : 240) : o.value);
    }
  }
}
console.log(`compared ${compared} spots, max probability difference ${worst.toExponential(2)}`);
if (worst > (nativeModelDir === modelDir ? 1e-3 : 2e-2)) process.exit(1);
