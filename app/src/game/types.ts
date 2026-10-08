import type { AvatarSpec } from '@/account/avatars';

// What one seat sees of a game (solver/src/session.cpp, "game view").
export interface GameTrick {
  leader: number;
  cards: [number, number][];
  tigress: null | 'pirate' | 'escape';
  winner: number;
}
export interface GameAction {
  type: 'bid' | 'card' | 'tigress';
  value: number;
  a: string;
}
export interface RoundResult {
  round: number;
  bids: number[];
  won: number[];
  points: number[];
}
export interface ReviewItem {
  round: number;
  trick: number;
  what: 'bid' | 'card' | 'tigress';
  chosen: number;
  p: number;
  best: number;
  pBest: number;
}
export interface GameView {
  ok: true;
  id: number;
  seat: number;
  round: number;
  phase: 'bidding' | 'playing' | 'roundEnd' | 'gameOver';
  start: number;
  pendingTigress: boolean;
  toAct: number[];
  hand: number[];
  bids: number[];
  bidIn: boolean[];
  won: number[];
  scores: number[];
  trick: GameTrick;
  lastTrick: GameTrick | null;
  legal: GameAction[];
  results: RoundResult[];
  review: ReviewItem[];
}

export interface Player {
  name: string;
  bot: boolean;
  online?: boolean;
  avatar?: AvatarSpec | null;
}

/** The game table drives either a local bot game or an online room. */
export interface GameController {
  view: GameView | null;
  players: Player[];
  busy: boolean;
  error: string | null;
  act: (a: string) => void;
  next: () => void;
  /** GTO probabilities for my options (training aid); null if not offered */
  hint?: () => Promise<Record<string, number> | null>;
  /** waiting for other humans to continue after a round */
  waitingNext?: boolean;
}
