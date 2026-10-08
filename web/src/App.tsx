import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { TIGRESS_ESCAPE, TIGRESS_PIRATE, cardName, handOrder, multiplicity } from "./cards";
import { type Option, type SpotAnswer, type SpotQuery, isLoaded, loadRound, spot } from "./engine";
import { BidAdvice, BidEntry } from "./components/BidPanel";
import { Card, pct } from "./components/Card";
import { CardGrid } from "./components/CardGrid";
import { Glyph } from "./components/Glyph";
import { POS, Setup, seatLabel } from "./components/Setup";
import { Table, TrickLog } from "./components/Table";

const EMPTY: SpotQuery = { round: 1, me: 0, hand: [], bids: [-1, -1, -1, -1], play: [] };

// The situation lives in the URL hash, so a spot can be bookmarked or shared.
function readHash(): SpotQuery {
  const h = new URLSearchParams(location.hash.slice(1));
  const nums = (k: string) => (h.get(k) ? h.get(k)!.split(",").filter(Boolean).map(Number) : []);
  const round = Math.min(10, Math.max(1, Number(h.get("r")) || 1));
  const bids = nums("b");
  return {
    round,
    me: Math.min(3, Math.max(0, Number(h.get("s")) || 0)),
    hand: nums("h").slice(0, round),
    bids: bids.length === 4 ? bids : [-1, -1, -1, -1],
    play: nums("p"),
  };
}
function writeHash(q: SpotQuery) {
  const h = `r=${q.round}&s=${q.me}&h=${q.hand}&b=${q.bids}&p=${q.play}`;
  history.replaceState(null, "", `#${h}`);
}

export default function App() {
  const [q, setQ] = useState<SpotQuery>(() => (location.hash ? readHash() : EMPTY));
  const [past, setPast] = useState<SpotQuery[]>([]);
  const [ready, setReady] = useState(() => isLoaded(q.round));
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => writeHash(q), [q]);
  useEffect(() => {
    let live = true;
    setReady(isLoaded(q.round));
    setLoadError(null);
    loadRound(q.round).then(
      () => live && setReady(true),
      (e: Error) => live && setLoadError(e.message),
    );
    return () => {
      live = false;
    };
  }, [q.round]);

  const update = (next: SpotQuery) => {
    setPast((p) => [...p.slice(-200), q]);
    setQ(next);
  };
  const undo = () => {
    if (!past.length) return;
    setQ(past[past.length - 1]);
    setPast(past.slice(0, -1));
  };
  const reset = (keep: Partial<SpotQuery> = {}) => update({ ...EMPTY, round: q.round, me: q.me, ...keep });

  const handFull = q.hand.length === q.round;
  const answer = useMemo(() => (ready && handFull ? spot(q) : null), [ready, handFull, q]);
  const a = answer?.ok ? (answer as SpotAnswer) : null;

  const handCount = (k: number) => q.hand.filter((x) => x === k).length;
  const pickHand = (k: number) => {
    if (q.play.length) return;
    const n = handCount(k);
    if (n < multiplicity(k) && !handFull) update({ ...q, hand: [...q.hand, k], bids: [-1, -1, -1, -1] });
    else if (n > 0) {
      const i = q.hand.lastIndexOf(k);
      update({ ...q, hand: q.hand.filter((_, j) => j !== i), bids: [-1, -1, -1, -1] });
    }
  };
  const play = (x: number) => update({ ...q, play: [...q.play, x] });

  const step = !handFull ? 1 : !a || a.phase === "bidding" ? 2 : 3;

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark">
            <Glyph name="king" size={22} />
          </span>
          <div>
            <h1>SkullKI</h1>
            <p>GTO-Solver · Skull King · 4 Spieler</p>
          </div>
        </div>
        <nav className="steps" aria-label="Fortschritt">
          {["Hand", "Ansage", "Spiel"].map((s, i) => (
            <span key={s} className={`step ${step === i + 1 ? "is-on" : step > i + 1 ? "is-done" : ""}`}>
              <i>{i + 1}</i>
              {s}
            </span>
          ))}
        </nav>
      </header>

      <main>
        {q.play.length === 0 && (
          <Setup
            round={q.round}
            me={q.me}
            onRound={(r) => update({ ...EMPTY, round: r, me: q.me, hand: q.hand.slice(0, r) })}
            onSeat={(s) => update({ ...q, me: s, bids: [-1, -1, -1, -1] })}
          />
        )}

        {q.play.length === 0 && (
          <section className="panel" aria-label="Deine Hand">
            <div className="panel-head">
              <h2>Deine Hand</h2>
              <span className={`counter ${handFull ? "is-full" : ""}`}>
                {q.hand.length}/{q.round}
              </span>
            </div>
            <div className="hand hand-preview">
              <AnimatePresence>
                {[...q.hand].sort(handOrder).map((k, i) => (
                  <motion.div
                    key={`${k}-${i}`}
                    layout
                    initial={{ opacity: 0, y: 12, rotate: -4 }}
                    animate={{ opacity: 1, y: 0, rotate: 0 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                  >
                    <Card kind={k} size="md" onClick={() => pickHand(k)} />
                  </motion.div>
                ))}
              </AnimatePresence>
              {!q.hand.length && <p className="placeholder">Tippe unten die Karten an, die du bekommen hast.</p>}
            </div>
            {!handFull && (
              <CardGrid
                enabled={(k) => handCount(k) < multiplicity(k)}
                selected={handCount}
                onPick={pickHand}
              />
            )}
            {handFull && (
              <button type="button" className="ghost-btn" onClick={() => update({ ...q, hand: [], bids: [-1, -1, -1, -1] })}>
                Hand neu eingeben
              </button>
            )}
          </section>
        )}

        {handFull && !ready && !loadError && (
          <section className="panel skeleton" aria-busy="true">
            <span className="spinner" /> Strategie für Runde {q.round} wird geladen…
          </section>
        )}
        {loadError && <div className="alert">{loadError}</div>}
        {answer && !answer.ok && (
          <div className="alert" role="alert">
            <b>Das geht so nicht:</b> {answer.error}
            {past.length > 0 && (
              <button type="button" className="link-btn" onClick={undo}>
                Rückgängig
              </button>
            )}
          </div>
        )}

        {a && q.play.length === 0 && a.bidAdvice && (
          <section className="panel" aria-label="Ansage">
            <BidAdvice
              advice={a.bidAdvice}
              chosen={q.bids[q.me] >= 0 ? q.bids[q.me] : undefined}
              onChoose={(b) => update({ ...q, bids: q.bids.map((v, i) => (i === q.me ? b : v)) })}
            />
            <BidEntry
              round={q.round}
              me={q.me}
              bids={q.bids}
              onBid={(s, b) => update({ ...q, bids: q.bids.map((v, i) => (i === s ? b : v)) })}
            />
            {a.phase === "bidding" && (
              <p className="hint">Trage alle vier Ansagen ein – dann geht es zum Spiel.</p>
            )}
          </section>
        )}

        {a && a.phase !== "bidding" && <Play a={a} q={q} onPlay={play} />}

        {handFull && (
          <div className="toolbar">
            <button type="button" className="ghost-btn" onClick={undo} disabled={!past.length}>
              ↶ Zurück
            </button>
            {a?.phase === "done" ? (
              <button type="button" className="primary-btn" onClick={() => reset({ round: Math.min(10, q.round + 1) })}>
                Nächste Runde
              </button>
            ) : (
              <button type="button" className="ghost-btn" onClick={() => reset()}>
                Neu
              </button>
            )}
          </div>
        )}
      </main>

      <footer className="foot">
        <p>
          Strategie aus Neural-CFR-Training · Lücke ≤ 0,2 Punkte/Runde gegen trainierte Gegenstrategien.{" "}
          <span className="muted">Bald: gegen die KI spielen · Multiplayer</span>
        </p>
      </footer>
    </div>
  );
}

function Play({ a, q, onPlay }: { a: SpotAnswer; q: SpotQuery; onPlay: (x: number) => void }) {
  const myTurn = a.phase === "playing" && a.toAct === a.me;
  const opts = a.options;
  const best = myTurn ? opts.reduce((x, y) => ((y.p ?? 0) > (x.p ?? 0) ? y : x), opts[0]) : undefined;
  const cardOpt = new Map(opts.filter((o) => o.type === "card").map((o) => [o.value as number, o] as const));
  const hand = [...(a.hand ?? [])].sort(handOrder);
  const rel = (a.toAct - a.me + 4) % 4;
  const tigressMarker = (o: Option) => (o.value === "pirate" ? TIGRESS_PIRATE : TIGRESS_ESCAPE);

  return (
    <section className="panel play" aria-label="Spiel">
      <Table a={a} />

      {a.phase === "done" && a.points && (
        <div className="result">
          <span className="kicker">Runde vorbei</span>
          <ul>
            {a.points.map((pts, s) => (
              <li key={s} className={s === a.me ? "is-me" : ""}>
                <span>{seatLabel(s, a.me)}</span>
                <b className={pts >= 0 ? "pos" : "neg"}>{pts > 0 ? `+${pts}` : pts}</b>
              </li>
            ))}
          </ul>
        </div>
      )}

      {a.phase === "playing" && a.pendingTigress && (
        <div className="decision">
          <span className="kicker">{myTurn ? "Deine Tigress" : `${seatLabel(a.toAct, a.me)} spielt Tigress`}</span>
          <p className="decision-q">Als Pirat oder als Flucht?</p>
          <div className="tig-choices">
            {opts.map((o) => (
              <button
                key={String(o.value)}
                type="button"
                className={`tig-btn ${best === o ? "is-best" : ""}`}
                onClick={() => onPlay(tigressMarker(o))}
              >
                <Glyph name={o.value === "pirate" ? "pirate" : "escape"} size={22} />
                <span>{o.value === "pirate" ? "Pirat" : "Flucht"}</span>
                {o.p !== undefined && <b>{pct(o.p)}%</b>}
              </button>
            ))}
          </div>
        </div>
      )}

      {a.phase === "playing" && !a.pendingTigress && myTurn && best && (
        <div className="decision">
          <span className="kicker">Du bist dran</span>
          <p className="decision-q">
            GTO spielt <b>{cardName(best.value as number)}</b> <span className="mono">{pct(best.p ?? 0)}%</span>
          </p>
        </div>
      )}

      {a.phase === "playing" && !a.pendingTigress && !myTurn && (
        <div className="decision">
          <span className="kicker">
            {seatLabel(a.toAct, a.me)} {POS[rel]} ist dran
          </span>
          <p className="decision-q">Welche Karte wurde gespielt?</p>
          <CardGrid
            enabled={(k) => cardOpt.has(k)}
            count={(k) => cardOpt.get(k)?.n}
            onPick={onPlay}
          />
        </div>
      )}

      {a.phase === "playing" && (
        <div className={`hand hand-live ${myTurn && !a.pendingTigress ? "is-active" : ""}`}>
          {hand.map((k, i) => {
            const o = myTurn && !a.pendingTigress ? cardOpt.get(k) : undefined;
            return (
              <Card
                key={`${k}-${i}`}
                kind={k}
                size="lg"
                p={o?.p}
                best={o !== undefined && o === best}
                disabled={myTurn && !a.pendingTigress && !o}
                onClick={o ? () => onPlay(k) : undefined}
              />
            );
          })}
        </div>
      )}

      <TrickLog a={a} />
      {q.play.length === 0 && a.phase === "playing" && (
        <p className="hint">Ansagen ändern geht, bis die erste Karte gespielt ist.</p>
      )}
    </section>
  );
}
