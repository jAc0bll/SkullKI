// Multiplayer client: one WebSocket to the SkullKI server (server/index.mjs).
// Reconnects by itself and resumes the seat with a saved token.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { GameController, GameView, Player } from './types';

export const DEFAULT_SERVER = 'wss://skullki.schwartihost.com';

export interface RoomInfo {
  code: string;
  seat: number;
  host: boolean;
  started: boolean;
  players: Player[];
  ready: number[];
}

export type Status = 'idle' | 'connecting' | 'online' | 'offline';

const KEY = { server: 'skullki.server', name: 'skullki.name', token: 'skullki.token' };

export async function loadPrefs() {
  const [server, name] = await Promise.all([AsyncStorage.getItem(KEY.server), AsyncStorage.getItem(KEY.name)]);
  return { server: server ?? DEFAULT_SERVER, name: name ?? '' };
}

export function useOnlineGame() {
  const [status, setStatus] = useState<Status>('idle');
  const [room, setRoom] = useState<RoomInfo | null>(null);
  const [view, setView] = useState<GameView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sentNext, setSentNext] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const url = useRef<string>('');
  const queue = useRef<object[]>([]);
  const retry = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closing = useRef(false);

  const send = useCallback((m: object) => {
    const s = ws.current;
    if (s && s.readyState === 1) s.send(JSON.stringify(m));
    else queue.current.push(m);
  }, []);

  const connect = useCallback((server: string) => {
    url.current = server;
    closing.current = false;
    setStatus('connecting');
    const s = new WebSocket(server);
    ws.current = s;
    s.onopen = async () => {
      setStatus('online');
      setError(null);
      const token = await AsyncStorage.getItem(KEY.token);
      if (token) s.send(JSON.stringify({ t: 'hello', token }));
      for (const m of queue.current.splice(0)) s.send(JSON.stringify(m));
    };
    s.onmessage = (e) => {
      const m = JSON.parse(String(e.data));
      if (m.t === 'joined') AsyncStorage.setItem(KEY.token, m.token);
      else if (m.t === 'room') setRoom(m as RoomInfo);
      else if (m.t === 'state') {
        setView(m.view as GameView);
        if (m.view.phase !== 'roundEnd') setSentNext(false);
      } else if (m.t === 'error') setError(m.msg);
    };
    s.onclose = () => {
      if (ws.current !== s) return;
      setStatus('offline');
      if (!closing.current) retry.current = setTimeout(() => connect(url.current), 2000);
    };
    s.onerror = () => setError('Server nicht erreichbar');
  }, []);

  useEffect(
    () => () => {
      closing.current = true;
      if (retry.current) clearTimeout(retry.current);
      ws.current?.close();
    },
    [],
  );

  const ensure = useCallback(
    async (server: string, name: string) => {
      await AsyncStorage.multiSet([
        [KEY.server, server],
        [KEY.name, name],
      ]);
      if (!ws.current || ws.current.readyState > 1 || url.current !== server) {
        ws.current?.close();
        connect(server);
      }
    },
    [connect],
  );

  const create = useCallback(
    async (server: string, name: string) => {
      await AsyncStorage.removeItem(KEY.token);
      await ensure(server, name);
      send({ t: 'create', name });
    },
    [ensure, send],
  );
  const join = useCallback(
    async (server: string, name: string, code: string) => {
      await AsyncStorage.removeItem(KEY.token);
      await ensure(server, name);
      send({ t: 'join', code, name });
    },
    [ensure, send],
  );
  /** Rejoin a room after the app was closed (saved token). */
  const resume = useCallback(
    async (server: string) => {
      if (await AsyncStorage.getItem(KEY.token)) connect(server);
    },
    [connect],
  );
  const leave = useCallback(async () => {
    send({ t: 'leave' });
    await AsyncStorage.removeItem(KEY.token);
    setRoom(null);
    setView(null);
  }, [send]);

  const controller: GameController = {
    view,
    players: room?.players ?? [],
    busy: status !== 'online',
    error: status === 'offline' ? 'Verbindung weg, verbinde neu…' : error,
    act: (a) => send({ t: 'act', a }),
    next: () => {
      setSentNext(true);
      send({ t: 'next' });
    },
    waitingNext: sentNext,
  };

  return { status, room, error, create, join, resume, leave, start: () => send({ t: 'start' }), controller };
}
