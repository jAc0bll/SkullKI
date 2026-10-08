import { AnimatePresence, motion } from "motion/react";
import { SUITS, cardName } from "../cards";
import type { SpotAnswer, Trick } from "../engine";
import { Card } from "./Card";
import { Glyph } from "./Glyph";
import { POS, seatLabel } from "./Setup";

const PLACE = ["bottom", "left", "top", "right"];

// The table seen from my chair: me at the bottom, play goes clockwise
// (left, across, right).
export function Table({ a }: { a: SpotAnswer }) {
  const tricks = a.tricks ?? [];
  const last = tricks[tricks.length - 1];
  const current: Trick | undefined = last && last.winner < 0 ? last : undefined;
  // Right after a trick is complete, keep showing it until the next card.
  const shown = current ?? (a.phase === "done" ? last : last && last.cards.length === 4 ? last : undefined);
  const won = a.won ?? [0, 0, 0, 0];
  return (
    <div className="table" aria-label="Spieltisch">
      <div className="table-felt" />
      {[0, 1, 2, 3].map((s) => {
        const rel = (s - a.me + 4) % 4;
        const toAct = a.toAct === s && a.phase === "playing";
        const bid = a.bids[s];
        const hit = bid === won[s];
        return (
          <div key={s} className={`seat seat-${PLACE[rel]} ${toAct ? "is-turn" : ""} ${s === a.me ? "is-me" : ""}`}>
            <span className="seat-name">
              {seatLabel(s, a.me)}
              {rel > 0 && <em>{POS[rel]}</em>}
            </span>
            <span className={`seat-score ${hit ? "is-hit" : won[s] > bid ? "is-over" : ""}`}>
              <b>{won[s]}</b>
              <span>/{bid}</span>
            </span>
            {a.voids && a.voids[s] > 0 && (
              <span className="seat-voids" title="hat diese Farbe nicht mehr">
                {SUITS.map((su, i) =>
                  (a.voids![s] >> i) & 1 ? (
                    <span key={i} className={`void suit-${i}`} aria-label={`kein ${su.name}`}>
                      <Glyph name={su.glyph} size={12} />
                    </span>
                  ) : null,
                )}
              </span>
            )}
          </div>
        );
      })}
      <div className="trick">
        <AnimatePresence>
          {shown?.cards.map(([seat, kind]) => {
            const rel = (seat - a.me + 4) % 4;
            const win = shown.winner === seat;
            return (
              <motion.div
                key={`${tricks.indexOf(shown)}-${seat}`}
                className={`trick-card trick-${PLACE[rel]} ${win ? "is-winner" : ""}`}
                initial={{ opacity: 0, scale: 0.7, ...FROM[rel] }}
                animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ type: "spring", stiffness: 380, damping: 28 }}
              >
                <Card kind={kind} size="sm" />
                {kind === 59 && shown.tigress && (
                  <span className="tig-mode">{shown.tigress === "pirate" ? "Pirat" : "Flucht"}</span>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
        {!shown && a.phase === "playing" && <span className="trick-empty">Stich {tricks.length + 1}</span>}
      </div>
    </div>
  );
}

const FROM = [{ y: 60 }, { x: -60 }, { y: -60 }, { x: 60 }];

export function TrickLog({ a }: { a: SpotAnswer }) {
  const done = (a.tricks ?? []).filter((t) => t.winner >= 0);
  if (!done.length) return null;
  return (
    <details className="log">
      <summary>
        Gespielte Stiche <span className="muted">({done.length})</span>
      </summary>
      <ol>
        {done.map((t, i) => (
          <li key={i}>
            <span className="log-n">{i + 1}</span>
            <span className="log-cards">
              {t.cards.map(([s, k]) => (
                <span key={s} className={`log-card ${s === t.winner ? "is-winner" : ""}`} title={seatLabel(s, a.me)}>
                  {cardName(k)}
                  {k === 59 && t.tigress ? ` (${t.tigress === "pirate" ? "Pirat" : "Flucht"})` : ""}
                </span>
              ))}
            </span>
            <span className="log-win">→ {seatLabel(t.winner, a.me)}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}
