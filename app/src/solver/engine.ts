// Talks to the C++ spot solver: native on iPhone (modules/sk-solver),
// WebAssembly on the web. Nets ship inside the app (assets/models).
import { Asset } from 'expo-asset';
import { Platform } from 'react-native';
import SkSolver from '../../modules/sk-solver';

/* eslint-disable @typescript-eslint/no-require-imports */
const MODELS: Record<number, number> = {
  1: require('../../assets/models/r1.bin'),
  2: require('../../assets/models/r2.bin'),
  3: require('../../assets/models/r3.bin'),
  4: require('../../assets/models/r4.bin'),
  5: require('../../assets/models/r5.bin'),
  6: require('../../assets/models/r6.bin'),
  7: require('../../assets/models/r7.bin'),
  8: require('../../assets/models/r8.bin'),
  9: require('../../assets/models/r9.bin'),
  10: require('../../assets/models/r10.bin'),
};
/* eslint-enable @typescript-eslint/no-require-imports */

export interface Option {
  type: 'bid' | 'card' | 'tigress';
  value: number | 'pirate' | 'escape';
  p?: number;
  n?: number;
}
export interface Trick {
  leader: number;
  cards: [number, number][];
  tigress: null | 'pirate' | 'escape';
  winner: number;
}
export interface Answer {
  ok: true;
  round: number;
  me: number;
  phase: 'bidding' | 'playing' | 'done';
  toAct: number;
  pendingTigress?: boolean;
  bids?: number[];
  won?: number[];
  hand?: number[];
  unseen?: number[];
  voids?: number[];
  tricks?: Trick[];
  bidAdvice?: { bid: number; p: number }[];
  options: Option[];
  points?: number[];
  leader?: number;
}
export type Result = Answer | { ok: false; error: string };

const loaded = new Map<number, Promise<void>>();
const ready = new Set<number>();

export function loadRound(round: number): Promise<void> {
  let p = loaded.get(round);
  if (!p) {
    p = (async () => {
      if (!SkSolver) throw new Error('Der Solver ist auf diesem Gerät nicht verfügbar');
      const asset = Asset.fromModule(MODELS[round]);
      await asset.downloadAsync();
      const uri = asset.localUri ?? asset.uri;
      const path = Platform.OS === 'web' ? uri : decodeURIComponent(uri.replace(/^file:\/\//, ''));
      if (!(await SkSolver.load(round, path))) throw new Error(`Strategie für Runde ${round} konnte nicht geladen werden`);
      ready.add(round);
    })();
    p.catch(() => loaded.delete(round));
    loaded.set(round, p);
  }
  return p;
}

export const isReady = (round: number) => ready.has(round);

export function query(text: string): Result {
  if (!SkSolver) return { ok: false, error: 'Solver nicht verfügbar' };
  return JSON.parse(SkSolver.query(text)) as Result;
}

export interface FullSpot {
  round: number;
  me: number;
  hand: number[];
  bids: number[];
  play: number[];
}
export interface DirectSpot {
  round: number;
  me: number;
  hand: number[];
  bids: number[];
  won: number[];
  played: number[];
  trick: number[];
  tigress: number;
  voids: number[];
}

export const fullText = (q: FullSpot) =>
  `round=${q.round} me=${q.me} hand=${q.hand.join(',')} bids=${q.bids.join(',')} play=${q.play.join(',')}`;

export const directText = (q: DirectSpot) =>
  `mode=direct round=${q.round} me=${q.me} hand=${q.hand.join(',')} bids=${q.bids.join(',')} won=${q.won.join(',')} ` +
  `played=${q.played.join(',')} trick=${q.trick.join(',')} tigress=${q.tigress} voids=${q.voids.join(',')}`;
