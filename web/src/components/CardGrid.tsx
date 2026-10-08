import { ESCAPE, MERMAID, PIRATE, SKULL_KING, SUITS, TIGRESS, cardName, tone } from "../cards";
import { Glyph } from "./Glyph";

// Compact picker for all 61 card kinds: one block per suit (1-7, 8-14) and a
// row of characters. Touch targets stay >= 44px on a phone.
export function CardGrid({
  enabled,
  count,
  selected,
  onPick,
}: {
  enabled: (k: number) => boolean;
  count?: (k: number) => number | undefined;
  selected?: (k: number) => number;
  onPick: (k: number) => void;
}) {
  const chip = (k: number, label: React.ReactNode) => {
    const sel = selected?.(k) ?? 0;
    const n = count?.(k);
    return (
      <button
        key={k}
        type="button"
        className={`chip ${tone(k)} ${sel ? "is-selected" : ""}`}
        disabled={!enabled(k)}
        onClick={() => onPick(k)}
        aria-label={cardName(k)}
        aria-pressed={sel > 0}
      >
        {label}
        {sel > 1 && <span className="chip-count">×{sel}</span>}
        {n !== undefined && n > 1 && !sel && <span className="chip-count chip-count-muted">×{n}</span>}
      </button>
    );
  };
  return (
    <div className="grid-picker">
      {SUITS.map((s, si) => (
        <div className={`grid-suit suit-${si}`} key={si}>
          <div className="grid-suit-label">
            <Glyph name={s.glyph} size={15} />
            <span>{s.name}</span>
          </div>
          <div className="grid-chips">
            {Array.from({ length: 14 }, (_, v) => chip(si * 14 + v, <span className="chip-num">{v + 1}</span>))}
          </div>
        </div>
      ))}
      <div className="grid-specials">
        {[SKULL_KING, PIRATE, TIGRESS, MERMAID, ESCAPE].map((k) =>
          chip(
            k,
            <>
              <Glyph name={toneGlyph(k)} size={18} />
              <span className="chip-name">{cardName(k)}</span>
            </>,
          ),
        )}
      </div>
    </div>
  );
}

function toneGlyph(k: number) {
  return ({ [ESCAPE]: "escape", [MERMAID]: "mermaid", [PIRATE]: "pirate", [TIGRESS]: "tigress", [SKULL_KING]: "king" } as const)[k]!;
}
