// The C++ spot solver, compiled to WebAssembly (scripts/build_wasm.sh).
import createSk from "./wasm/sk.js";

export type OptionType = "bid" | "card" | "tigress";
export interface Option {
  type: OptionType;
  value: number | "pirate" | "escape";
  /** my strategy's probability (only when I am to act) */
  p?: number;
  /** copies still unseen (opponent card options) */
  n?: number;
}
export interface Trick {
  leader: number;
  cards: [number, number][];
  tigress: null | "pirate" | "escape";
  winner: number;
}
export interface SpotAnswer {
  ok: true;
  round: number;
  me: number;
  phase: "bidding" | "playing" | "done";
  toAct: number;
  pendingTigress: boolean;
  bids: number[];
  won?: number[];
  hand?: number[];
  unseen?: number[];
  voids?: number[];
  tricks?: Trick[];
  bidAdvice?: { bid: number; p: number }[];
  options: Option[];
  points?: number[];
}
export type SpotResult = SpotAnswer | { ok: false; error: string };

export interface SpotQuery {
  round: number;
  me: number;
  hand: number[];
  bids: number[];
  play: number[];
}

type Module = Awaited<ReturnType<typeof createSk>>;
let mod: Promise<Module> | null = null;
let spotFn: ((text: string) => string) | null = null;
const loaded = new Map<number, Promise<void>>();

function module() {
  mod ??= createSk().then((m: Module) => {
    spotFn = m.cwrap("sk_spot", "string", ["string"]);
    return m;
  });
  return mod;
}

/** Download (once) and load the strategy net of a round. */
export function loadRound(round: number): Promise<void> {
  let p = loaded.get(round);
  if (!p) {
    p = (async () => {
      const m = await module();
      const res = await fetch(`${import.meta.env.BASE_URL}models/r${round}.bin`);
      if (!res.ok) throw new Error(`Strategie für Runde ${round} nicht gefunden`);
      const bytes = new Uint8Array(await res.arrayBuffer());
      m.FS.writeFile(`/r${round}.bin`, bytes);
      if (!m.ccall("sk_load", "number", ["number", "string"], [round, `/r${round}.bin`]))
        throw new Error(`Strategie für Runde ${round} ist beschädigt`);
      m.FS.unlink(`/r${round}.bin`);
    })();
    p.catch(() => loaded.delete(round));
    loaded.set(round, p);
  }
  return p;
}

export function isLoaded(round: number) {
  return spotFn !== null && loaded.has(round);
}

/** Synchronous once the round is loaded. */
export function spot(q: SpotQuery): SpotResult {
  if (!spotFn) return { ok: false, error: "Solver lädt noch…" };
  const text = `round=${q.round} me=${q.me} hand=${q.hand.join(",")} bids=${q.bids.join(",")} play=${q.play.join(",")}`;
  return JSON.parse(spotFn(text)) as SpotResult;
}
