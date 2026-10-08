// Offline game against three bots that play the trained strategy. Runs the
// C++ game session on the device; bots move with a short delay so you can
// follow the table.
import { useCallback, useEffect, useRef, useState } from 'react';
import { loadRound, rawQuery } from '@/solver/engine';
import type { GameController, GameView, Player } from './types';

export const BOT_NAMES = ['Anne Bonny', 'Blackbeard', 'Calico Jack'];
const ME = 0;

const call = (cmd: string) => JSON.parse(rawQuery(cmd));

export function useLocalGame(): GameController & { restart: () => void } {
  const [id, setId] = useState<number | null>(null);
  const [view, setView] = useState<GameView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const players: Player[] = [{ name: 'Du', bot: false }, ...BOT_NAMES.map((name) => ({ name, bot: true }))];

  const refresh = useCallback((gid: number) => {
    const v = call(`game view id=${gid} seat=${ME}`);
    if (v.ok) setView(v as GameView);
    else setError(v.error);
    return v as GameView;
  }, []);

  const start = useCallback(async () => {
    if (timer.current) clearTimeout(timer.current);
    setBusy(true);
    setError(null);
    try {
      await loadRound(1);
      const seed = Math.floor(Math.random() * 2 ** 31);
      const created = call(`game new seed=${seed} start=${Math.floor(Math.random() * 4)}`);
      setId(created.id);
      refresh(created.id);
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  }, [refresh]);

  useEffect(() => {
    start();
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [start]);

  // Let the bots move whenever it is their turn.
  useEffect(() => {
    if (id === null || !view || (view.phase !== 'bidding' && view.phase !== 'playing')) return;
    const bots = view.toAct.filter((s) => s !== ME);
    if (!bots.length) return;
    const trickJustEnded = view.phase === 'playing' && view.trick.cards.length === 0 && view.lastTrick;
    const delay = view.phase === 'bidding' ? 450 : trickJustEnded ? 1500 : 750;
    timer.current = setTimeout(async () => {
      try {
        await loadRound(view.round);
        for (const s of view.phase === 'bidding' ? bots : bots.slice(0, 1)) call(`game bot id=${id} seat=${s}`);
        refresh(id);
      } catch (e) {
        setError((e as Error).message);
      }
    }, delay);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [id, view, refresh]);

  const act = useCallback(
    (a: string) => {
      if (id === null) return;
      const r = call(`game act id=${id} seat=${ME} a=${a}`);
      if (!r.ok) setError(r.error);
      refresh(id);
    },
    [id, refresh],
  );

  const next = useCallback(async () => {
    if (id === null || !view) return;
    setBusy(true);
    try {
      await loadRound(Math.min(10, view.round + 1));
      call(`game next id=${id}`);
      refresh(id);
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  }, [id, view, refresh]);

  const hint = useCallback(async () => {
    if (id === null || !view) return null;
    await loadRound(view.round);
    const r = call(`game hint id=${id} seat=${ME}`);
    if (!r.ok) return null;
    return Object.fromEntries((r.options as { a: string; p: number }[]).map((o) => [o.a, o.p]));
  }, [id, view]);

  return { view, players, busy, error, act, next, hint, restart: start };
}
