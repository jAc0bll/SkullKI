import { motion } from "motion/react";
import { pct } from "./Card";
import { POS, seatLabel } from "./Setup";

export function BidAdvice({ advice, chosen, onChoose }: {
  advice: { bid: number; p: number }[];
  chosen?: number;
  onChoose?: (b: number) => void;
}) {
  const best = advice.reduce((a, b) => (b.p > a.p ? b : a), advice[0]);
  const mixed = advice.filter((a) => a.p >= 0.05).length > 1;
  return (
    <div className="advice">
      <div className="advice-head">
        <span className="kicker">GTO-Ansage</span>
        <div className="advice-big">
          <span className="advice-num">{best.bid}</span>
          <span className="advice-sub">
            {best.bid === 1 ? "Stich" : "Stiche"} · <b>{pct(best.p)}%</b>
            {mixed && <em> gemischt</em>}
          </span>
        </div>
      </div>
      <ol className="bars" aria-label="Wahrscheinlichkeit je Ansage">
        {advice.map((a, i) => (
          <li key={a.bid}>
            <button
              type="button"
              className={`bar ${a.bid === best.bid ? "is-best" : ""} ${a.bid === chosen ? "is-chosen" : ""}`}
              onClick={onChoose ? () => onChoose(a.bid) : undefined}
              disabled={!onChoose}
              aria-label={`Ansage ${a.bid}: ${pct(a.p)} Prozent`}
            >
              <span className="bar-label">{a.bid}</span>
              <span className="bar-track">
                <motion.span
                  className="bar-fill"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: Math.max(a.p, 0.004) }}
                  transition={{ delay: i * 0.03, type: "spring", stiffness: 140, damping: 20 }}
                />
              </span>
              <span className="bar-pct">{pct(a.p)}%</span>
            </button>
          </li>
        ))}
      </ol>
      {mixed && (
        <p className="hint">
          Gemischte Strategie: GTO spielt hier mehrere Ansagen. Jede davon ist gleich gut, wenn du sie im richtigen
          Verhältnis wählst – so bist du nicht ausrechenbar.
        </p>
      )}
    </div>
  );
}

export function BidEntry({ round, me, bids, onBid }: {
  round: number;
  me: number;
  bids: number[];
  onBid: (seat: number, bid: number) => void;
}) {
  return (
    <div className="bid-entry">
      <span className="kicker">Ansagen am Tisch</span>
      {[0, 1, 2, 3].map((s) => {
        const rel = (s - me + 4) % 4;
        const v = bids[s];
        return (
          <div className={`bid-row ${s === me ? "is-me" : ""}`} key={s}>
            <span className="bid-who">
              {seatLabel(s, me)}
              {rel > 0 && <em>{POS[rel]}</em>}
            </span>
            <div className="stepper">
              <button type="button" aria-label="weniger" onClick={() => onBid(s, v <= 0 ? -1 : v - 1)} disabled={v < 0}>
                −
              </button>
              <output aria-live="polite">{v < 0 ? "?" : v}</output>
              <button type="button" aria-label="mehr" onClick={() => onBid(s, Math.min(round, v + 1))} disabled={v >= round}>
                +
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
