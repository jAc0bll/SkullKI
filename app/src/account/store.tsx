// The signed-in player: just a name (no password yet), an avatar and the
// statistics kept by the server (server/users.mjs). Remembered on the device.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, use, useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { SERVER_HTTP } from '@/config';
import type { GameView } from '@/game/types';
import type { AvatarSpec } from './avatars';

export interface Stats {
  games: number;
  wins: number;
  points: number;
  best: number | null;
  rounds: number;
  bidsHit: number;
  gtoAgree: number;
  gtoTotal: number;
  bot: { games: number; wins: number };
  online: { games: number; wins: number };
}
export interface HistoryEntry {
  at: string;
  mode: 'bot' | 'online';
  score: number;
  rank: number;
  players: string[];
}
export interface User {
  name: string;
  avatar: AvatarSpec;
  createdAt: string;
  stats: Stats;
  history: HistoryEntry[];
  /** games may be kept anonymously to improve the AI */
  shareGames: boolean;
}
export interface LeaderRow {
  name: string;
  avatar: AvatarSpec;
  games: number;
  wins: number;
  avg: number;
}

const KEY = 'skullki.user';

async function call<T>(path: string, body?: object): Promise<T> {
  const res = await fetch(`${SERVER_HTTP}${path}`, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? 'Server nicht erreichbar');
  return data as T;
}

/** What a finished game means for `seat` (same as server/users.mjs summarize). */
export function summarize(view: GameView, seat: number) {
  const rank = 1 + [0, 1, 2, 3].filter((s) => view.scores[s] > view.scores[seat]).length;
  return {
    score: view.scores[seat],
    rank,
    rounds: view.results.length,
    bidsHit: view.results.filter((r) => r.bids[seat] === r.won[seat]).length,
    gtoAgree: view.review.filter((r) => r.chosen === r.best).length,
    gtoTotal: view.review.length,
  };
}

interface Account {
  user: User | null;
  ready: boolean;
  error: string | null;
  login: (name: string, avatar: AvatarSpec) => Promise<boolean>;
  logout: () => void;
  setAvatar: (a: AvatarSpec) => Promise<void>;
  recordBotGame: (view: GameView, players: string[], log?: object) => Promise<void>;
  setShareGames: (on: boolean) => Promise<void>;
  refresh: () => Promise<void>;
  leaderboard: () => Promise<LeaderRow[]>;
}

const Ctx = createContext<Account | null>(null);

export function AccountProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const keep = useCallback((u: User | null) => {
    setUser(u);
    if (u) AsyncStorage.setItem(KEY, JSON.stringify(u));
    else AsyncStorage.removeItem(KEY);
  }, []);

  const refresh = useCallback(async () => {
    const saved = await AsyncStorage.getItem(KEY);
    if (!saved) return;
    const u = JSON.parse(saved) as User;
    setUser(u);
    try {
      keep(await call<User>(`/api/users/${encodeURIComponent(u.name)}`));
    } catch {
      /* offline: keep the saved copy */
    }
  }, [keep]);

  useEffect(() => {
    refresh().finally(() => setReady(true));
  }, [refresh]);

  const value = useMemo<Account>(
    () => ({
      user,
      ready,
      error,
      refresh,
      async login(name, avatar) {
        setError(null);
        try {
          keep(await call<User>('/api/login', { name, avatar }));
          return true;
        } catch (e) {
          setError((e as Error).message === 'Failed to fetch' ? 'Server nicht erreichbar' : (e as Error).message);
          return false;
        }
      },
      logout: () => keep(null),
      async setAvatar(avatar) {
        if (!user) return;
        keep({ ...user, avatar });
        try {
          keep(await call<User>(`/api/users/${encodeURIComponent(user.name)}/avatar`, { avatar }));
        } catch (e) {
          setError((e as Error).message);
        }
      },
      async recordBotGame(view, players, log) {
        if (!user) return;
        try {
          keep(
            await call<User>(`/api/users/${encodeURIComponent(user.name)}/games`, {
              ...summarize(view, view.seat),
              players,
              log: user.shareGames !== false ? log : undefined,
            }),
          );
        } catch {
          /* offline: this game is not counted */
        }
      },
      async setShareGames(on) {
        if (!user) return;
        keep({ ...user, shareGames: on });
        try {
          keep(await call<User>(`/api/users/${encodeURIComponent(user.name)}/settings`, { shareGames: on }));
        } catch (e) {
          setError((e as Error).message);
        }
      },
      leaderboard: () => call<LeaderRow[]>('/api/leaderboard'),
    }),
    [user, ready, error, refresh, keep],
  );
  return <Ctx value={value}>{children}</Ctx>;
}

export function useAccount() {
  const v = use(Ctx);
  if (!v) throw new Error('useAccount outside AccountProvider');
  return v;
}
