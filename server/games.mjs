// Finished games, kept for analysing human play (and later a model of how
// people play). One JSON object per line, one file per month:
//   server/data/games/2026-10.jsonl
// Players appear only as pseudonymous ids, never by name.
//
// Line: {id, at, mode:'online'|'bot', source:'server'|'app',
//        players:[{pid, human}], log:{v, rounds:[...], scores}}   (log: session.cpp)
import { appendFileSync, mkdirSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data', 'games');

export function saveGame({ mode, source, players, log }) {
  if (!log || typeof log !== 'object' || !Array.isArray(log.rounds)) return false;
  mkdirSync(DIR, { recursive: true });
  const at = new Date().toISOString();
  const line = JSON.stringify({ id: randomBytes(6).toString('hex'), at, mode, source, players, log });
  if (line.length > 300_000) return false;
  appendFileSync(path.join(DIR, `${at.slice(0, 7)}.jsonl`), line + '\n');
  return true;
}

/** A pseudonymous id for a guest without an account (new per game). */
export const guestId = () => 'g-' + randomBytes(4).toString('hex');
