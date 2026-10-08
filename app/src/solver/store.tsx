// Solver state shared by the solver screen and the card picker sheet,
// with undo.
import { createContext, use, useCallback, useMemo, useReducer, type ReactNode } from 'react';
import type { DirectSpot, FullSpot } from './engine';

export type Mode = 'full' | 'direct';

export interface State {
  mode: Mode;
  round: number;
  me: number;
  // full round
  hand: number[];
  bids: number[];
  play: number[];
  // mid-round (direct)
  dHand: number[];
  dBids: number[];
  won: number[];
  played: number[];
  trick: number[];
  tigress: number;
  voids: number[];
}

const initial: State = {
  mode: 'full',
  round: 1,
  me: 0,
  hand: [],
  bids: [-1, -1, -1, -1],
  play: [],
  dHand: [],
  dBids: [0, 0, 0, 0],
  won: [0, 0, 0, 0],
  played: [],
  trick: [],
  tigress: -1,
  voids: [0, 0, 0, 0],
};

type Action = { type: 'set'; patch: Partial<State> } | { type: 'undo' } | { type: 'reset'; keep?: Partial<State> };
interface Hist {
  now: State;
  past: State[];
}

function reducer(h: Hist, a: Action): Hist {
  switch (a.type) {
    case 'set':
      return { now: { ...h.now, ...a.patch }, past: [...h.past.slice(-150), h.now] };
    case 'undo':
      return h.past.length ? { now: h.past[h.past.length - 1], past: h.past.slice(0, -1) } : h;
    case 'reset': {
      const fresh =
        h.now.mode === 'full'
          ? { hand: [], bids: initial.bids, play: [] }
          : { dHand: [], dBids: initial.dBids, won: initial.won, played: [], trick: [], tigress: -1, voids: initial.voids };
      return { now: { ...h.now, ...fresh, ...a.keep }, past: [...h.past, h.now] };
    }
  }
}

interface Store {
  s: State;
  set: (patch: Partial<State>) => void;
  undo: () => void;
  reset: (keep?: Partial<State>) => void;
  canUndo: boolean;
  full: FullSpot;
  direct: DirectSpot;
}

const Ctx = createContext<Store | null>(null);

export function SolverProvider({ children }: { children: ReactNode }) {
  const [h, dispatch] = useReducer(reducer, { now: initial, past: [] });
  const set = useCallback((patch: Partial<State>) => dispatch({ type: 'set', patch }), []);
  const undo = useCallback(() => dispatch({ type: 'undo' }), []);
  const reset = useCallback((keep?: Partial<State>) => dispatch({ type: 'reset', keep }), []);
  const s = h.now;
  const value = useMemo<Store>(
    () => ({
      s,
      set,
      undo,
      reset,
      canUndo: h.past.length > 0,
      full: { round: s.round, me: s.me, hand: s.hand, bids: s.bids, play: s.play },
      direct: {
        round: s.round,
        me: s.me,
        hand: s.dHand,
        bids: s.dBids,
        won: s.won,
        played: s.played,
        trick: s.trick,
        tigress: s.tigress,
        voids: s.voids,
      },
    }),
    [s, set, undo, reset, h.past.length],
  );
  return <Ctx value={value}>{children}</Ctx>;
}

export function useSolver() {
  const v = use(Ctx);
  if (!v) throw new Error('useSolver outside SolverProvider');
  return v;
}

export const countOf = (list: number[], k: number) => list.reduce((n, x) => n + (x === k ? 1 : 0), 0);
export const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
