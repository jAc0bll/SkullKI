export const POS = ["", "links", "gegenüber", "rechts"];

export function seatLabel(seat: number, me: number) {
  if (seat === me) return "Du";
  return `Sitz ${seat + 1}`;
}

export function Setup({
  round,
  me,
  onRound,
  onSeat,
}: {
  round: number;
  me: number;
  onRound: (r: number) => void;
  onSeat: (s: number) => void;
}) {
  return (
    <section className="panel setup" aria-label="Runde und Sitz">
      <div className="field">
        <span className="field-label">Runde</span>
        <div className="coins" role="radiogroup" aria-label="Runde">
          {Array.from({ length: 10 }, (_, i) => i + 1).map((r) => (
            <button
              key={r}
              type="button"
              role="radio"
              aria-checked={r === round}
              className={`coin ${r === round ? "is-on" : ""}`}
              onClick={() => onRound(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="field-label">
          Dein Platz <em>in der Reihenfolge dieser Runde</em>
        </span>
        <div className="seats" role="radiogroup" aria-label="Dein Platz">
          {[0, 1, 2, 3].map((s) => (
            <button
              key={s}
              type="button"
              role="radio"
              aria-checked={s === me}
              className={`seat-pick ${s === me ? "is-on" : ""}`}
              onClick={() => onSeat(s)}
            >
              <strong>{s + 1}.</strong>
              <span>{s === 0 ? "sagt an & spielt aus" : s === 3 ? "zuletzt" : `${s + 1}. am Zug`}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
